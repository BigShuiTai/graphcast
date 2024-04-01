import gc
import numpy as np
import xarray as xr
from datetime import datetime

def main(files, steps):
    # init dataset
    new_dataset = xr.Dataset(
        data_vars = {
            # constants
            "geopotential_at_surface": (["lat", "lon"], np.empty((721, 1440))),
            "land_sea_mask": (["lat", "lon"], np.empty((721, 1440))),
            # surface
            "2m_temperature": (["batch", "time", "lat", "lon"], np.empty((1, steps, 721, 1440))),
            "mean_sea_level_pressure": (["batch", "time", "lat", "lon"], np.empty((1, steps, 721, 1440))),
            "10m_v_component_of_wind": (["batch", "time", "lat", "lon"], np.empty((1, steps, 721, 1440))),
            "10m_u_component_of_wind": (["batch", "time", "lat", "lon"], np.empty((1, steps, 721, 1440))),
            "total_precipitation_6hr": (["batch", "time", "lat", "lon"], np.empty((1, steps, 721, 1440))),
            # isobaric
            "temperature": (["batch", "time", "level", "lat", "lon"], np.empty((1, steps, 13, 721, 1440))),
            "geopotential": (["batch", "time", "level", "lat", "lon"], np.empty((1, steps, 13, 721, 1440))),
            "u_component_of_wind": (["batch", "time", "level", "lat", "lon"], np.empty((1, steps, 13, 721, 1440))),
            "v_component_of_wind": (["batch", "time", "level", "lat", "lon"], np.empty((1, steps, 13, 721, 1440))),
            "vertical_velocity": (["batch", "time", "level", "lat", "lon"], np.empty((1, steps, 13, 721, 1440))),
            "specific_humidity": (["batch", "time", "level", "lat", "lon"], np.empty((1, steps, 13, 721, 1440))),
        },
        coords=dict(
            lon=np.linspace(0, 359.75, 1440),
            lat=np.linspace(-90, 90, 721),
            level=np.array([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]),
            time=np.array([np.timedelta64(i * 21600000000000, 'ns') for i in range(-1, steps - 1)]),
        )
    )
    
    # constants from ERA5 Reanalysis data
    z_sf = xr.open_dataset('geopotential_at_surface.nc')
    lsmsk = xr.open_dataset('land_sea_mask.nc')
    
    # read grib
    habvg_2 = xr.open_mfdataset(files, engine='cfgrib', backend_kwargs={'indexpath': '', 'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 2}}, combine='nested', concat_dim="time", parallel=False)
    habvg_10 = xr.open_mfdataset(files, engine='cfgrib', backend_kwargs={'indexpath': '', 'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10}}, combine='nested', concat_dim="time", parallel=False)
    msl = xr.open_mfdataset(files, engine='cfgrib', backend_kwargs={'indexpath': '', 'filter_by_keys': {'typeOfLevel': 'meanSea'}}, combine='nested', concat_dim="time", parallel=False)
    upper = xr.open_mfdataset(files, engine='cfgrib', backend_kwargs={'indexpath': '', 'filter_by_keys': {'typeOfLevel': 'isobaricInhPa'}}, combine='nested', concat_dim="time", parallel=False)
    
    # process in Dataset
    start_date = datetime.utcfromtimestamp(msl.time.data[0].astype(int) * 1e-9)
    select_times = np.array([np.datetime64(start_date, 'ns') + np.timedelta64(i * 21600000000000, 'ns') for i in range(steps)])
    new_dataset = new_dataset.assign_coords({
        "datetime": (["batch", "time"], np.array([select_times]))
    })
    
    new_dataset["geopotential_at_surface"][:] = z_sf['z'][0].data[::-1].astype(np.float32)
    new_dataset["land_sea_mask"][:] = lsmsk['lsm'][0].data[::-1].astype(np.float32)
    
    new_dataset["2m_temperature"][:,:2] = habvg_2['t2m'].data.astype(np.float32)
    new_dataset["mean_sea_level_pressure"][:,:2] = msl['mslet'].data.astype(np.float32)
    new_dataset["10m_v_component_of_wind"][:,:2] = habvg_10['v10'].data.astype(np.float32)
    new_dataset["10m_u_component_of_wind"][:,:2] = habvg_10['u10'].data.astype(np.float32)
    
    new_dataset["temperature"][:,:2] = upper['t'].data[:,::-1].astype(np.float32)
    new_dataset["geopotential"][:,:2] = (upper['gh'].data[:,::-1] * 9.80665).astype(np.float32)
    new_dataset["u_component_of_wind"][:,:2] = upper['u'].data[:,::-1].astype(np.float32)
    new_dataset["v_component_of_wind"][:,:2] = upper['v'].data[:,::-1].astype(np.float32)
    new_dataset["vertical_velocity"][:,:2] = upper['w'].data[:,::-1].astype(np.float32)
    new_dataset["specific_humidity"][:,:2] = upper['q'].data[:,::-1].astype(np.float32)
    return new_dataset

if __name__ == '__main__':
    files = ['gfs_analysis_p006.grib', 'gfs_analysis_f000.grib']
    steps = 2 + 240 // 6
    ds = main(files, steps)
    gc.collect()
    ds.to_netcdf('input.nc')
