"""
Example class for a Cloud based Atlite.
"""

from cloudpathlib import CloudPath
import zarr
from gcsfs import GCSFileSystem
import xarray as xr
from atlite.cutout import Cutout


class CloudCutout(Cutout):
    """
    Atlite style cutout that is underpinned by zarr archive on cloud rather than local data.
    """

    def __init__(
        self,
        path,
        time=(None, None),
        x=(None, None),
        y=(None, None),
        use_caching=False,
        max_cache_size=2**28,
    ):
        """
        Provide an Atlite style cutout that where the data is hosted on the cloud.

        Parameters
        ----------
        path : str | path-like
            Zarr archive from which to load the cutout.
            This will contain raw met data, rather than prepared atlite-style data.
        time : str | slice
            Time range to include in the cutout, e.g. "2011" or
            ("2011-01-05", "2011-01-25")
            This is necessary when building a new cutout.
        x : slice, optional
            Outer longitudinal bounds for the cutout (west, east).
        y : slice, optional
            Outer latitudinal bounds for the cutout (south, north).
        use_caching : bool, default=False
            Whether to use an in memory cache cloud data.
        max_cache_size : float, default=2**28
            The maximum size that the cache may grow to, in number of bytes.
        """
        path = CloudPath(path)
        mapper = GCSFileSystem().get_mapper
        store = mapper(str(path))
        if use_caching is True:
            cache = zarr.LRUStoreCache(store, max_size=max_cache_size)
            data = xr.open_zarr(store=cache, chunks={})
        else:
            data = xr.open_zarr(store=store, chunks={})
        self.path = path
        self.data = data.sel(
            time=slice(time[0], time[1]), x=slice(x[0], x[-1]), y=slice(y[0], y[-1])
        )
