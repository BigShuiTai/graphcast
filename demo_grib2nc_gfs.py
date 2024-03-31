# A demo processing GFS analysis data for GraphCast input
# TODO: too large RAM usage (could anyone fix it?)
import gc
import numpy as np
import xarray as xr
from datetime import datetime

def main(steps):
    # init dataset
    ds_dict = {
        "geopotential_at_surface": np.full((721, 1440), np.nan),
        "land_sea_mask": np.full((721, 1440), np.nan),
        "2m_temperature": np.full((1, steps, 721, 1440), np.nan),
        "mean_sea_level_pressure": np.full((1, steps, 721, 1440), np.nan),
        "10m_v_component_of_wind": np.full((1, steps, 721, 1440), np.nan),
        "10m_u_component_of_wind": np.full((1, steps, 721, 1440), np.nan),
        "total_precipitation_6hr": np.full((1, steps, 721, 1440), np.nan),
        "temperature": np.full((1, steps, 13, 721, 1440), np.nan),
        "geopotential": np.full((1, steps, 13, 721, 1440), np.nan),
        "u_component_of_wind": np.full((1, steps, 13, 721, 1440), np.nan),
        "v_component_of_wind": np.full((1, steps, 13, 721, 1440), np.nan),
        "vertical_velocity": np.full((1, steps, 13, 721, 1440), np.nan),
        "specific_humidity": np.full((1, steps, 13, 721, 1440), np.nan),
    }
    
    # read grib
    habvg_2 = xr.open_mfdataset(['gfs.pgrb2.0p25.p006', 'gfs.pgrb2.0p25.f000'], engine='cfgrib', backend_kwargs={'indexpath': '', 'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 2}}, combine='nested', concat_dim="time", parallel=False)
    habvg_10 = xr.open_mfdataset(['gfs.pgrb2.0p25.p006', 'gfs.pgrb2.0p25.f000'], engine='cfgrib', backend_kwargs={'indexpath': '', 'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10}}, combine='nested', concat_dim="time", parallel=False)
    msl = xr.open_mfdataset(['gfs.pgrb2.0p25.p006', 'gfs.pgrb2.0p25.f000'], engine='cfgrib', backend_kwargs={'indexpath': '', 'filter_by_keys': {'typeOfLevel': 'meanSea'}}, combine='nested', concat_dim="time", parallel=False)
    upper = xr.open_mfdataset(['gfs.pgrb2.0p25.p006', 'gfs.pgrb2.0p25.f000'], engine='cfgrib', backend_kwargs={'indexpath': '', 'filter_by_keys': {'typeOfLevel': 'isobaricInhPa'}}, combine='nested', concat_dim="time", parallel=False)
    gh_surface = xr.open_dataset('geopotential_at_surface.nc') # geopotential_at_surface
    lsmsk = xr.open_dataset('land_sea_mask.nc') # land_sea_mask
    
    input_lats = np.linspace(-90, 90, 721)
    input_lons = np.linspace(0, 359.75, 1440)
    levels = np.array([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])
    start_date = datetime.utcfromtimestamp(msl.time.data[0].astype(int) * 1e-9)
    select_times = np.array([np.datetime64(start_date, 'ns') + np.timedelta64(i * 21600000000000, 'ns') for i in range(steps)])
    valid_times = np.array([np.timedelta64(i * 21600000000000, 'ns') for i in range(-1, steps - 1)])
    
    # process in dict
    ds_dict["geopotential_at_surface"] = gh_surface['z'][0].data[::-1]
    ds_dict["land_sea_mask"] = lsmsk['lsm'][0].data[::-1]
    
    ds_dict["2m_temperature"][:,:2] = habvg_2['t2m'].data
    ds_dict["mean_sea_level_pressure"][:,:2] = msl['mslet'].data
    ds_dict["10m_v_component_of_wind"][:,:2] = habvg_10['v10'].data
    ds_dict["10m_u_component_of_wind"][:,:2] = habvg_10['u10'].data
    
    ds_dict["temperature"][:,:2] = upper['t'].data[:,::-1]
    ds_dict["geopotential"][:,:2] = (upper['gh'].data[:,::-1] * 9.80665)
    ds_dict["u_component_of_wind"][:,:2] = upper['u'].data[:,::-1]
    ds_dict["v_component_of_wind"][:,:2] = upper['v'].data[:,::-1]
    ds_dict["vertical_velocity"][:,:2] = upper['w'].data[:,::-1]
    ds_dict["specific_humidity"][:,:2] = upper['q'].data[:,::-1]
    
    habvg_2.close()
    habvg_10.close()
    msl.close()
    upper.close()
    gh_surface.close()
    lsmsk.close()
    
    del habvg_2, habvg_10, msl, upper, gh_surface, lsmsk
    gc.collect()
    
    new_dataset = xr.Dataset(
        data_vars = {
            "geopotential_at_surface": (["lat", "lon"], ds_dict["geopotential_at_surface"].astype(np.float32)),
            "land_sea_mask": (["lat", "lon"], ds_dict["land_sea_mask"].astype(np.float32)),
            "2m_temperature": (["batch", "time", "lat", "lon"], ds_dict["2m_temperature"].astype(np.float32)),
            "mean_sea_level_pressure": (["batch", "time", "lat", "lon"], ds_dict["mean_sea_level_pressure"].astype(np.float32)),
            "10m_v_component_of_wind": (["batch", "time", "lat", "lon"], ds_dict["10m_v_component_of_wind"].astype(np.float32)),
            "10m_u_component_of_wind": (["batch", "time", "lat", "lon"], ds_dict["10m_u_component_of_wind"].astype(np.float32)),
            "total_precipitation_6hr": (["batch", "time", "lat", "lon"], ds_dict["total_precipitation_6hr"].astype(np.float32)),
            "temperature": (["batch", "time", "level", "lat", "lon"], ds_dict["temperature"].astype(np.float32)),
            "geopotential": (["batch", "time", "level", "lat", "lon"], ds_dict["geopotential"].astype(np.float32)),
            "u_component_of_wind": (["batch", "time", "level", "lat", "lon"], ds_dict["u_component_of_wind"].astype(np.float32)),
            "v_component_of_wind": (["batch", "time", "level", "lat", "lon"], ds_dict["v_component_of_wind"].astype(np.float32)),
            "vertical_velocity": (["batch", "time", "level", "lat", "lon"], ds_dict["vertical_velocity"].astype(np.float32)),
            "specific_humidity": (["batch", "time", "level", "lat", "lon"], ds_dict["specific_humidity"].astype(np.float32)),
        },
        coords=dict(
            lon=input_lons,
            lat=input_lats,
            level=levels,
            time=valid_times,
            datetime=(["batch", "time"], np.array([select_times])),
        )
    )
    return new_dataset

if __name__ == '__main__':
    STEPS = 2 + 240 // 6
    ds = main(STEPS)
    gc.collect()
    ds.to_netcdf('input.nc')
