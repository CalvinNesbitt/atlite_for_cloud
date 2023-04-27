# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2016 - 2023 The Atlite Authors
#
# SPDX-License-Identifier: MIT

"""
Base class for Cloud based Atlite.
"""

# There is a binary incompatibility between the pip wheels of netCDF4 and
# rasterio, which leads to the first one to work correctly while the second
# loaded one fails by loading netCDF4 first, we ensure that most of atlite's
# functionality works fine, even when the pip wheels have been used, only for
# resampling the sarah dataset it is important to use conda.
# Refer to
# https://github.com/pydata/xarray/issues/2535,
# https://github.com/rasterio/rasterio-wheels/issues/12

import logging
from cloudpathlib import CloudPath
from warnings import warn

import xarray as xr
from numpy import atleast_1d
from pyproj import CRS

from atlite.datasets import modules as datamodules

from atlite.cutout import Cutout

logger = logging.getLogger(__name__)


class CloudCutout(Cutout):
    """
    Cutout base class.

    This class builds the starting point for most atlite
    functionalities.
    """

    def __init__(self, path, **cutoutparams):
        """
        Provide an Atlite cutout that where the data is hosted on the cloud.

        Parameters
        ----------
        path : str | cloud path
            Zarr archive from which to load the cutout.
        time : str | slice
            Time range to include in the cutout, e.g. "2011" or
            ("2011-01-05", "2011-01-25")
            This is necessary when building a new cutout.
        bounds : GeoSeries.bounds | DataFrame, optional
            The outer bounds of the cutout or as a DataFrame
            containing (min.long, min.lat, max.long, max.lat).
        x : slice, optional
            Outer longitudinal bounds for the cutout (west, east).
        y : slice, optional
            Outer latitudinal bounds for the cutout (south, north).
        dx : float, optional
            Step size of the x coordinate. The default is 0.25.
        dy : float, optional
            Step size of the y coordinate. The default is 0.25.
        dt : str, optional
            Frequency of the time coordinate. The default is 'h'. Valid are all
            pandas offset aliases.
        """
        module = "era5_cloud"
        # name = cutoutparams.get("name", None)
        # cutout_dir = cutoutparams.get("cutout_dir", None)
        # if cutout_dir or name or Path(path).is_dir():
        #     raise ValueError(
        #         "Old style format not supported. You can migrate the old "
        #         "cutout directory using the function "
        #         "`atlite.utils.migrate_from_cutout_directory()`. The argument "
        #         "`cutout_dir` and `name` have been deprecated in favour of `path`."
        #     )

        path = CloudPath(path)

        # Backward compatibility for xs, ys, months and years
        if {"xs", "ys"}.intersection(cutoutparams):
            warn(
                "The arguments `xs` and `ys` have been deprecated in favour of "
                "`x` and `y`",
                DeprecationWarning,
            )
            if "xs" in cutoutparams:
                cutoutparams["x"] = cutoutparams.pop("xs")
            if "ys" in cutoutparams:
                cutoutparams["y"] = cutoutparams.pop("ys")

        if {"years", "months"}.intersection(cutoutparams):
            warn(
                "The arguments `years` and `months` have been deprecated in "
                "favour of `time`",
                DeprecationWarning,
            )
            assert "years" in cutoutparams
            months = cutoutparams.pop("months", slice(1, 12))
            years = cutoutparams.pop("years")
            cutoutparams["time"] = slice(
                f"{years.start}-{months.start}", f"{years.stop}-{months.stop}"
            )

        # Expecting Path to be a cloud store
        data = xr.open_dataset(str(path), engine="zarr", chunks={})
        if cutoutparams:
            warn(
                f'Arguments {", ".join(cutoutparams)} are ignored, since '
                "cutout is already built."
            )

        # Check compatibility of CRS
        modules = atleast_1d(data.attrs.get("module"))
        crs = set(CRS(datamodules[m].crs) for m in modules)
        assert len(crs) == 1, f"CRS of {module} not compatible"

        self.path = path
        self.data = data
