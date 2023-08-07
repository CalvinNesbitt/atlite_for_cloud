"""
Module for opening data from Transition Zero ERA5 Archive.

For further reference see
https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation
"""

import logging
import os
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from dateutil import parser
from gcsfs import GCSFileSystem
from cloudpathlib import CloudPath, GSClient
import dask
import google.auth


from atlite.gis import maybe_swap_spatial_dims
from atlite.pv.solar_position import SolarPosition

# Null context for running a with statements wihout any context
try:
    from contextlib import nullcontext
except ImportError:
    # for Python verions < 3.7:
    import contextlib

    @contextlib.contextmanager
    def nullcontext():
        yield


logger = logging.getLogger(__name__)

# Model and CRS Settings
crs = 4326

features = {
    "height": ["height"],
    "wind": ["wnd100m", "wnd_azimuth", "roughness"],
    "influx": [
        "influx_toa",
        "influx_direct",
        "influx_diffuse",
        "albedo",
        "solar_altitude",
        "solar_azimuth",
    ],
    "temperature": ["temperature", "soil temperature"],
    "runoff": ["runoff"],
}

static_features = {"height"}

# Loading Credentials for Google Cloud from env variables if available, otherwise
# from default credentials

try:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
except KeyError:
    warnings.warn(
        "No Google Cloud credentials found in environment variables. "
        "Will try using default credentials instead."
    )
    credentials, project_id = google.auth.default()


def _add_height(ds):
    """
    Convert geopotential 'z' to geopotential height following [1].

    References
    ----------
    [1] ERA5: surface elevation and orography, retrieved: 10.02.2019
    https://confluence.ecmwf.int/display/CKB/ERA5%3A+surface+elevation+and+orography
    """
    g0 = 9.80665
    z = ds["z"]
    if "time" in z.coords:
        z = z.isel(time=0, drop=True)
    ds["height"] = z / g0
    ds = ds.drop_vars("z")
    return ds


def _rename_and_clean_coords(ds, add_lon_lat=True):
    """
    Rename 'longitude' and 'latitude' columns to 'x' and 'y' and fix roundings.

    Optionally (add_lon_lat, default:True) preserves latitude and
    longitude columns as 'lat' and 'lon'.
    """
    ds = ds.rename({"longitude": "x", "latitude": "y"})
    # round coords since cds coords are float32 which would lead to mismatches
    ds = ds.assign_coords(
        x=np.round(ds.x.astype(float), 5), y=np.round(ds.y.astype(float), 5)
    )
    ds = maybe_swap_spatial_dims(ds)
    if add_lon_lat:
        ds = ds.assign_coords(lon=ds.coords["x"], lat=ds.coords["y"])

    # Combine ERA5 and ERA5T data into a single dimension.
    # See https://github.com/PyPSA/atlite/issues/190
    # if "expver" in ds.dims.keys():
    #     # expver=1 is ERA5 data, expver=5 is ERA5T data
    #     # This combines both by filling in NaNs from ERA5 data with values from ERA5T.
    #     ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))

    return ds


def get_data_wind(retrieval_params):
    """
    Get wind data for given retrieval parameters.
    """
    ds = retrieve_raw_data(
        variable=[
            "u100",
            "v100",
            "fsr",
        ],
        **retrieval_params,
    )
    ds = _rename_and_clean_coords(ds)

    ds["wnd100m"] = np.sqrt(ds["u100"] ** 2 + ds["v100"] ** 2).assign_attrs(
        units="ms-1", long_name="100 metre wind speed"
    )
    # span the whole circle: 0 is north, π/2 is east, -π is south, 3π/2 is west
    azimuth = np.arctan2(ds["u100"], ds["v100"])
    ds["wnd_azimuth"] = azimuth.where(azimuth >= 0, azimuth + 2 * np.pi)
    ds = ds.drop_vars(["u100", "v100"])
    ds = ds.rename({"fsr": "roughness"})
    return ds


def sanitize_wind(ds):
    """
    Sanitize retrieved wind data.
    """
    ds["roughness"] = ds["roughness"].where(ds["roughness"] >= 0.0, 2e-4)
    return ds


def get_data_influx(retrieval_params):
    """
    Get influx data for given retrieval parameters.
    """
    ds = retrieve_raw_data(
        variable=[
            "fdir",
            "tisr",
            "ssrd",
            "ssr",
        ],
        **retrieval_params,
    )

    ds = _rename_and_clean_coords(ds)

    ds = ds.rename({"fdir": "influx_direct", "tisr": "influx_toa"})
    ds["albedo"] = (
        ((ds["ssrd"] - ds["ssr"]) / ds["ssrd"].where(ds["ssrd"] != 0))
        .fillna(0.0)
        .assign_attrs(units="(0 - 1)", long_name="Albedo")
    )
    ds["influx_diffuse"] = (ds["ssrd"] - ds["influx_direct"]).assign_attrs(
        units="J m**-2", long_name="Surface diffuse solar radiation downwards"
    )
    ds = ds.drop_vars(["ssrd", "ssr"])

    # Convert from energy to power J m**-2 -> W m**-2 and clip negative fluxes
    for a in ("influx_direct", "influx_diffuse", "influx_toa"):
        ds[a] = ds[a] / (60.0 * 60.0)
        ds[a].attrs["units"] = "W m**-2"

    # ERA5 variables are mean values for previous hour, i.e. 13:01 to 14:00 are labelled as "14:00"
    # account by calculating the SolarPosition for the center of the interval for aggregation happens
    # see https://github.com/PyPSA/atlite/issues/158
    # Do not show DeprecationWarning from new SolarPosition calculation (#199)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        time_shift = pd.to_timedelta("-30 minutes")
        sp = SolarPosition(ds, time_shift=time_shift)
    sp = sp.rename({v: f"solar_{v}" for v in sp.data_vars})

    ds = xr.merge([ds, sp])

    return ds


def sanitize_influx(ds):
    """
    Sanitize retrieved influx data.
    """
    for a in ("influx_direct", "influx_diffuse", "influx_toa"):
        ds[a] = ds[a].clip(min=0.0)
    return ds


def get_data_temperature(retrieval_params):
    """
    Get wind temperature for given retrieval parameters.
    """
    ds = retrieve_raw_data(variable=["t2m", "stl4"], **retrieval_params)

    ds = _rename_and_clean_coords(ds)
    ds = ds.rename({"t2m": "temperature", "stl4": "soil temperature"})

    return ds


def get_data_runoff(retrieval_params):
    """
    Get runoff data for given retrieval parameters.
    """
    ds = retrieve_raw_data(variable=["ro"], **retrieval_params)

    ds = _rename_and_clean_coords(ds)
    ds = ds.rename({"ro": "runoff"})

    return ds


def sanitize_runoff(ds):
    """
    Sanitize retrieved runoff data.
    """
    ds["runoff"] = ds["runoff"].clip(min=0.0)
    return ds


def get_data_height(retrieval_params):
    """
    Get height data for given retrieval parameters.
    """
    ds = retrieve_raw_data(variable=["z"], **retrieval_params)

    ds = _rename_and_clean_coords(ds)
    ds = _add_height(ds)

    return ds


def noisy_unlink(path):
    """
    Delete file at given path.
    """
    logger.debug(f"Deleting file {path}")
    try:
        os.unlink(path)
    except PermissionError:
        logger.error(f"Unable to delete file {path}, as it is still in use.")


def retrieve_raw_data(
    path="gs://metdata-era5/atlite-2013/zarr",
    use_caching=True,
    max_cache_size=2**28,
    **retrieval_params,
):
    variable = retrieval_params["variable"]
    time = retrieval_params["time"]
    x = retrieval_params["x"]
    y = retrieval_params["y"]
    mapper = GCSFileSystem(project=project_id, credentials=credentials).get_mapper
    client = GSClient(project=project_id, credentials=credentials)

    # Temporary fix until we make more complete archive
    if parser.parse(str(time[0])).year == 2018:
        path = path.replace("atlite-2013", "atlite-2018")
    path = client.GSPath(path)
    store = mapper(str(path))
    if use_caching is True:
        cache = zarr.LRUStoreCache(store, max_size=max_cache_size)
        ds = xr.open_zarr(store=cache, chunks={})[variable]
    else:
        ds = xr.open_zarr(store=store, chunks={})[variable]

    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        ds = ds.assign_coords(
            longitude=(((ds.longitude + 180) % 360) - 180)
        )  # Change coords from degrees east to west/east
        ds = ds.sortby("longitude")
    # Unpack bounds
    north = max(y)
    south = min(y)
    west = min(x)
    east = max(x)
    start_time, end_time = parser.parse(str(time[0])), parser.parse(str(time[1]))

    # Temporary fix until we make more complete archive: Ensuring start/end time is in 2013 or 2018
    if start_time.year != 2013 and start_time.year != 2018:
        raise ValueError(f"Start time year must be 2013 or 2018, not {start_time.year}")
    if end_time.year != 2013 and end_time.year != 2018:
        raise ValueError(f"End time year must be 2013 or 2018, not {end_time.year}")
    if start_time.year != end_time.year:
        raise ValueError("Start and end time must be in same year, either 2013 or 2018")

    # Subset data
    region_ds = ds.sel(
        latitude=slice(south, north),
        longitude=slice(west, east),
        time=slice(start_time, end_time),
    )
    return region_ds


def get_data(cutout, feature, tmpdir, lock=None, **creation_parameters):
    """
    Retrieve data from TZ google cloud bucket.

    This front-end function downloads data for a specific feature and formats
    it to match the given Cutout.

    Parameters
    ----------
    cutout : atlite.CloudCutout
    feature : str
        Name of the feature data to retrieve. Must be in
        `atlite.datasets.era5_tz.features`
    tmpdir : str/Path
        Directory where the temporary netcdf files are stored.
    **creation_parameters :
        Additional keyword arguments. The only effective argument is 'sanitize'
        (default True) which sets sanitization of the data on or off.

    Returns
    -------
    xarray.Dataset
        Dataset of dask arrays of the retrieved variables.
    """
    coords = cutout.coords

    sanitize = creation_parameters.get("sanitize", True)

    retrieval_params = {
        "x": (coords["x"].min().item(), coords["x"].max().item()),
        "y": (coords["y"].min().item(), coords["y"].max().item()),
        "time": (
            str(coords["time"].min().values),
            str(coords["time"].max().values),
        ),
    }

    func = globals().get(f"get_data_{feature}")
    sanitize_func = globals().get(f"sanitize_{feature}")

    def retrieve_once():
        ds = func({**retrieval_params})
        if sanitize and sanitize_func is not None:
            ds = sanitize_func(ds)
        return ds

    if feature in static_features:
        return retrieve_once().squeeze()

    return retrieve_once()
