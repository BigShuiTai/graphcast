import functools
import dataclasses

from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout

import haiku as hk
import jax

import numpy as np
import xarray as xr

def main(input_file, eval_steps):
    # load model & normalization data
    with open(
      "params/GraphCast_operational"
      " - ERA5-HRES 1979-2021"
      " - resolution 0.25"
      " - pressure levels 13"
      " - mesh 2to6"
      " - precipitation output only.npz", "rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)
    with open("stats/diffs_stddev_by_level.nc", "rb") as f:
        diffs_stddev_by_level = xr.load_dataset(f).compute()
    with open("stats/mean_by_level.nc", "rb") as f:
        mean_by_level = xr.load_dataset(f).compute()
    with open("stats/stddev_by_level.nc", "rb") as f:
        stddev_by_level = xr.load_dataset(f).compute()
    
    state = {}
    params = ckpt.params
    model_config = ckpt.model_config
    task_config = ckpt.task_config
    
    # Build jitted functions, and possibly initialize random weights
    def construct_wrapped_graphcast(
            model_config: graphcast.ModelConfig,
            task_config: graphcast.TaskConfig
        ):
        """Constructs and wraps the GraphCast Predictor."""
        # Deeper one-step predictor.
        predictor = graphcast.GraphCast(model_config, task_config)
        
        # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to
        # from/to float32 to/from BFloat16.
        predictor = casting.Bfloat16Cast(predictor)
        
        # Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from
        # BFloat16 happens after applying normalization to the inputs/targets.
        predictor = normalization.InputsAndResiduals(
            predictor,
            diffs_stddev_by_level=diffs_stddev_by_level,
            mean_by_level=mean_by_level,
            stddev_by_level=stddev_by_level
        )
        
        # Wraps everything so the one-step model can produce trajectories.
        predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
        return predictor
    
    @hk.transform_with_state
    def run_forward(model_config, task_config, inputs, targets_template, forcings):
        predictor = construct_wrapped_graphcast(model_config, task_config)
        return predictor(inputs, targets_template=targets_template, forcings=forcings)
    
    # Jax doesn't seem to like passing configs as args through the jit. Passing it
    # in via partial (instead of capture by closure) forces jax to invalidate the
    # jit cache if you change configs.
    def with_configs(fn):
        return functools.partial(fn, model_config=model_config, task_config=task_config)
    
    # Always pass params and state, so the usage below are simpler
    def with_params(fn):
        return functools.partial(fn, params=params, state=state)
    
    # Our models aren't stateful, so the state is always empty, so just return the
    # predictions. This is requiredy by our rollout code, and generally simpler.
    def drop_state(fn):
        return lambda **kw: fn(**kw)[0]
    
    # load data
    with xr.open_dataset(input_file) as init_ds:
        assert init_ds.sizes["time"] >= 3
        
        if eval_steps == 'auto':
            eval_steps = init_ds.sizes["time"] - 2
        else:
            assert eval_steps in range(1, (init_ds.sizes["time"] - 2) + 1), (
                f"Eval steps should be in [1, {init_ds.sizes['time'] - 2}].")
        
        print("============================")
        print(dataclasses.asdict(task_config))
        print("============================")
        
        # extract eval data
        eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
            init_ds,
            target_lead_times=slice("6h", f"{eval_steps * 6}h"),
            **dataclasses.asdict(task_config)
        )
        
        print("All Examples:  ", init_ds.sizes.mapping)
        print("Eval Inputs:   ", eval_inputs.sizes.mapping)
        print("Eval Targets:  ", eval_targets.sizes.mapping)
        print("Eval Forcings: ", eval_forcings.sizes.mapping)
        print("============================")
    
    assert model_config.resolution in (0, 360. / eval_inputs.sizes["lon"]), (
        "Model resolution doesn't match the data resolution. You likely want to "
        "re-filter the dataset list, and download the correct data.")
    
    print("Inputs:  ", eval_inputs.sizes.mapping)
    print("Targets: ", eval_targets.sizes.mapping)
    print("Forcings:", eval_forcings.sizes.mapping)
    print("============================")
    
    # jit functions
    init_jitted = jax.jit(with_configs(run_forward.init))
    
    run_forward_jitted = drop_state(
        with_params(
            jax.jit(
                with_configs(run_forward.apply)
            )
        )
    )
    
    # start predictions
    predictions = rollout.chunked_prediction_generator(
        run_forward_jitted,
        rng=jax.random.PRNGKey(0),
        inputs=eval_inputs,
        targets_template=eval_targets,
        forcings=eval_forcings
    )
    
    return predictions

if __name__ == '__main__':
    output_dir = 'output'
    predictions = main(input_file="input.nc", eval_steps='auto')
    for i, pred in enumerate(predictions, start=1):
        fcst_hour = i * 6
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [I] Running for GraphCast [+{fcst_hour}h]...")
        pred.attrs["10m_u_component_of_wind_units"] = "m/s"
        pred.attrs["10m_v_component_of_wind_units"] = "m/s"
        pred.attrs["2m_temperature_units"] = "K"
        pred.attrs["geopotential_units"] = "m2/s2"
        pred.attrs["mean_sea_level_pressure_units"] = "Pa"
        pred.attrs["specific_humidity_units"] = "kg/kg"
        pred.attrs["temperature_units"] = "K"
        pred.attrs["total_precipitation_6hr_units"] = "m"
        pred.attrs["u_component_of_wind_units"] = "m/s"
        pred.attrs["v_component_of_wind_units"] = "m/s"
        pred.attrs["vertical_velocity_units"] = "Pa/s"
        pred.attrs['description'] = 'GraphCast Model Inference Data'
        pred.attrs['reference'] = 'https://github.com/google-deepmind/graphcast'
        pred.attrs['code_author'] = 'BigShuiTai'
        pred.attrs['runtime'] = runtime
        pred.attrs['step'] = fcst_hour
        pred.to_netcdf(f'{output_dir}/graphcast_inference_{"%03d" % fcst_hour}.nc')
