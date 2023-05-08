#!/usr/bin/env python3

# Standard library imports
from typing import Any, Dict, List

# Third party imports
import bottleneck   
from joblib import Parallel, delayed
import numpy as np
from numpy.typing import NDArray
import pandas as pd

# Local imports
from bocpd import bocd, GaussianUnknownMean, plot_posterior


def parallel_bocd(raveled_scenes, unipix: int, bocd_params: dict, window: int) -> List[int]:
    """
        Run BOCD on a single pixel.
        Arguments:
            raveled_scenes: Raveled scenes.
            unipix: Unipixel index.
            bocd_params: Dictionary of BOCD parameters.
            window: Window size.
        Returns: First and last changepoints if they exist.
    """
    model = GaussianUnknownMean(bocd_params["mean0"], bocd_params["var0"], bocd_params["varx"])
    R, pmean, pvar = bocd(raveled_scenes[:, unipix], model, bocd_params["hazard"])
    predicted_cps = np.where(
        (np.diff((np.argmax(R, axis=1)))<0)
        & (np.max(R, axis=1)[1:] > 1e-1)
        )[0]
    if len(predicted_cps) == 0:
        return [-1, -1]
    return [predicted_cps[0], predicted_cps[-1]]


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

