#!/usr/bin/env python3

# Standard library imports
from typing import Any, Dict

# Third party imports
from bocd import BayesianOnlineChangePointDetection as BOCD, ConstantHazard, StudentT
import bottleneck   
from joblib import Parallel, delayed
import numpy as np
from numpy.typing import NDArray
import pandas as pd


def compute_run_lengths(arr: NDArray, bocd_params: Dict[str, Dict], region: str) -> NDArray:
    """
    Compute run lengths for arr.
    Arguments:
        arr: Array of values.
        bocd_params: Dictionary of BOCD parameters.
        region: Region.
    Returns: Run lengths.
    """
    student_params = {k:v for k, v in bocd_params[region].items() if k != "hazard"}
    bc = BOCD(ConstantHazard(bocd_params[region]["hazard"]), StudentT(**student_params))
    rt_mle = np.empty(arr.shape)
    for i, d in enumerate(arr):
        bc.update(d)
        rt_mle[i] = bc.rt
    return rt_mle

def detect_change(rt_mle: NDArray) -> NDArray:
    """ 
    Detect change points in rt_mle.
    Arguments:
        rt_mle: Run lengths.
        Returns: Indices of change points.
    """
    return np.where(np.diff(rt_mle) < 0)[0]


def make_change_map(
    encoded: NDArray,
    cogs_meta: pd.DataFrame,
    mask: NDArray,
    region: str,
    bocd_params: Dict[str, Dict],
    cores: int
    ) -> pd.DataFrame:
    """
    Make change map for region.
    Arguments:
        encoded: Encoded region scenes.
        cogs_meta: Regional COGs metadata.
        mask: Outlier mask.
        region: Region.
        bocd_params: Dictionary of BOCD parameters.
        cores: Number of cores for making channel-level change map.
    Returns: Regional change map with change summary stats for each pixel.
    """
    timesteps, rows, cols, channels = encoded.shape
    change_map = []
    for channel in range(channels):
        reshaped_encoded = encoded.copy()[:, :, :, channel].reshape(timesteps, rows * cols).T
        reshaped_mask = mask.copy()[:, :, :, channel].reshape(timesteps, rows * cols).T
        channel_change_map = pd.DataFrame(
            Parallel(n_jobs=cores-1)
            (
                delayed(summarize_pixel)
                (reshaped_encoded, unipix, reshaped_mask, bocd_params, cogs_meta, region) for unipix in range(rows * cols)
            )
        )
        channel_change_map.columns = [f"{channel}_{x}" for x in channel_change_map]
        change_map.append(channel_change_map)
    return pd.concat(change_map, axis=1)


def make_outlier_mask(encoded: NDArray, window=10, axis=0, n_stds: float=1.75) -> NDArray:
    """
    Make outlier mask for encoded.
    Arguments:
        encoded: Encoded scenes.
        window: Window size.
        axis: Axis to compute means and stds over.
        n_stds: Number of standard deviations.
    Returns: Outlier mask.
    """
    stds = bottleneck.move_std(encoded, window=window, axis=axis)
    means = bottleneck.move_mean(encoded, window=window, axis=axis)
    return (means - n_stds * stds < encoded) & (encoded < means + n_stds * stds)


def retrieve_change_dates(dates: NDArray, index_changes: NDArray) -> Dict[str, Any]:
    """
    Retrieve change dates from cogs_meta.
    Arguments:
        dates: Scene dates.
        index_changes: Indices of change points.
    Returns: List of change dates.
    """
    return pd.to_datetime(dates[index_changes])


def summarize_pixel(
    reshaped_encoded: NDArray,
    unipix: int,
    reshaped_mask: NDArray,
    bocd_params: Dict[str, Dict],
    cogs_meta: pd.DataFrame,
    region: str
    ) -> Dict[str, Any]:
    """
    Return changepoint summary statistics for one pixel.
    Arguments:
        reshaped_encoded: Two-dimensional array of encoded pixels.
        unipix: Index of pixel.
        bocd_params: Dictionary of BOCD parameters.
        region: Region.
    Returns: Summary stats for one pixel.
    """
    dates = cogs_meta["solar_date"].values[reshaped_mask[unipix, :]]
    vals = reshaped_encoded[unipix, :][reshaped_mask[unipix, :]]
    run_lengths = compute_run_lengths(vals, bocd_params, region)
    index_changes = detect_change(run_lengths)
    change_dates = retrieve_change_dates(dates, index_changes)
    return pd.Series(change_dates).describe(datetime_is_numeric=True).to_dict()
