# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2016 - 2023 The Atlite Authors
#
# SPDX-License-Identifier: MIT

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
    Cloud Cutout class.

    This class points the atlite cutout at a zarr store rather than an in disk .
    """

    def __init__(self, path, use_caching=False, max_cache_size=2**28, chunks={}, **cutoutparams):
        """
        Provide an Atlite cutout that where the data is hosted on the cloud.

        Parameters
        ----------
        path : str 
            Zarr archive from which to load the cutout.
        use_caching : boolean 
            Whether to use an in memory cache of our zarr archive 
        """
        path = CloudPath(path)
        mapper = GCSFileSystem().get_mapper
        store = mapper(str(path))
        if use_caching is True:
            cache = zarr.LRUStoreCache(store, max_size=max_cache_size)
            data = xr.open_zarr(store=cache, chunks=chunks)
        else:
            data = xr.open_zarr(store=store, chunks=chunks)
        self.path = path
        self.data = data
