# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2016 - 2024 The Atlite Authors & Calvin Nesbitt
#
# SPDX-License-Identifier: MIT
"""
Example class for a Cloud based Atlite.
"""

import logging

import xarray as xr
from numpy import atleast_1d
from tqdm.dask import TqdmCallback

from atlite.cutout import Cutout
from atlite.data import available_features, get_features, non_bool_dict

logger = logging.getLogger(__name__)


def cloud_cutout_prepare(
    cutout,
    features=None,
):
    """
    Prepare all or a selection of features in a cloud based cutout.

    This function loads the feature data of a cutout, e.g. influx or runoff.
    When not specifying the `feature` argument, all available features will be
    loaded. The function compares the variables which are already included in
    the cutout with the available variables of the modules specified by the
    cutout. It detects missing variables and stores them into the netcdf file
    of the cutout.


    Parameters
    ----------
    cutout : atlite.CloudCutout
    features : str/list, optional
        Feature(s) to be prepared. The default slice(None) results in all
        available features.

    Returns
    -------
    cutout : atlite.CloudCutout
        Cloud cutout with prepared data. The variables are stored in `cutout.data`.
    """
    if cutout.prepared:
        logger.info("Cutout already prepared.")
        return cutout

    modules = atleast_1d(cutout.module)
    features = atleast_1d(features) if features else slice(None)
    prepared = set(atleast_1d(cutout.data.attrs["prepared_features"]))

    # target is series of all available variables for given module and features
    target = available_features(modules).loc[:, features].drop_duplicates()

    for module in target.index.unique("module"):
        missing_vars = target[module]
        if missing_vars.empty:
            continue
        logger.info(f"Calculating with module {module}:")
        missing_features = missing_vars.index.unique("feature")
        ds = get_features(cutout, module, missing_features)
        prepared |= set(missing_features)

        cutout.data.attrs.update(dict(prepared_features=list(prepared)))
        attrs = non_bool_dict(cutout.data.attrs)
        attrs.update(ds.attrs)

        ds = cutout.data.merge(ds[missing_vars.values]).assign_attrs(**attrs)
        cutout.data = ds

    return cutout


class CloudCutout(Cutout):
    """
    Atlite style cutout that is underpinned by zarr archive on cloud rather
    than local data.
    """

    prepare = cloud_cutout_prepare

    def write_netcdf_locally(self, path=None):
        """
        Write the cutout to a local netcdf file.

        Parameters
        ----------
        path : str/Path
            Path to the netcdf file.
        """
        if path is None:
            path = self.path
        logger.info(f"Writing cutout to {path}")
        write_job = self.data.to_netcdf(path, compute=False)
        with TqdmCallback(desc="compute"):
            write_job.compute()

        self.data = xr.open_dataset(self.path, chunks={})
