#!/usr/bin/env python3

# Standard library imports
from pathlib import Path
import re
from typing import Dict, List, Tuple

# Third party imports
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import rasterio as rio
import torch
from torch.utils.data import Dataset


class MLPDataset(Dataset):
    """
    Dataset for training or running inference on pixel-level MLP.
    Let region be blank string to train on all regions.
    Must specify a region when running inference.
    """
    def __init__(
        self,
        images_dir: Path,
        spectral_bands: List[str],
        region: str="",
        train=True,
        norm_params: Dict={},
        slice_frac: float=0.001
    ):
        self.images_dir = images_dir
        self.spectral_bands = spectral_bands
        self.region = region
        self.train = train
        self.norm_params = norm_params
        self.slice_frac = slice_frac
        
        if (not train) and (not region):
            raise ValueError("Must specify region when running inference.")
        
        self.cogs_meta = make_cogs_meta(images_dir)
        if region:
            print(region)
            mask = self.cogs_meta["region"] == region
            self.cogs_meta = self.cogs_meta[mask]
        self.cube = make_cube(self.cogs_meta, slice_frac, train=train)
        self.cube, self.norm_params = normalize_cube(self.cube, spectral_bands, norm_params=norm_params, train=train)
        self.in_features = self.cube.shape[-1]
    def __len__(self) -> int:
        return len(self.cube)

    def __getitem__(self, index) -> torch.Tensor:
        if self.train:
            return torch.from_numpy(self.cube[index])
        return torch.from_numpy(self.cube[index].reshape(-1, self.in_features))


def make_cogs_meta(images_dir: Path) -> pd.DataFrame:
    """
    Make metadata table for COGs in images_dir.
    Arguments:
        images_dir: Path to directory containing COGs.
    Returns: Dataframe of COGs metadata.
    """
    pattern = re.compile(r"_\d{4}-\d{2}-\d{2}")
    cogs_meta = pd.Series(images_dir.glob("*.tif"), name="path").to_frame()
    cogs_meta["filestem"] = cogs_meta["path"].apply(lambda x: x.stem)
    cogs_meta["region"] = cogs_meta["filestem"].apply(lambda x: re.sub(pattern, "", x))
    cogs_meta["event_key"] = cogs_meta["filestem"].apply(lambda x: x.split("_")[0])
    cogs_meta["solar_date"] = cogs_meta["filestem"].apply(lambda x: x.split("_")[1])
    return cogs_meta


def make_cube(
    cogs_meta: pd.DataFrame,
    slice_frac: float,
    train: bool=True,
    qa_vals: List[int]=[]
    ) -> NDArray:
    """
    Make cube of pixels from COGs in images_dir.
    Arguments:
        cogs_meta: Dataframe of COGs metadata.
        spectral_bands: List of spectral bands to include in cube.
        slice_frac: Fraction of pixels to include in cube.
        train: True to sample pixels randomly from COGs.
        qa_vals: List of qa_vals to dummy.
    Returns: Cube of pixels from COGs in images_dir.
    """
    if not qa_vals:
        qa_vals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    cube = []
    for scene_src in cogs_meta["path"].values:
        with rio.open(scene_src) as src_scene:
            arr = src_scene.read()
        bands, rows, cols = arr.shape
        qa_dummies = []
        for qa_val in qa_vals:
            qa_dummies.append((arr[-1, :, :] == qa_val).astype(np.int32))
        qa_dummies.append(arr[-1, :, :] > max(qa_vals))
        qa_dummies = np.stack(qa_dummies, axis=0)
        arr = np.concatenate([arr[:-1, :, :], qa_dummies], axis=0)
        bands, _, _, = arr.shape
        arr = np.moveaxis(arr, 0, -1)
        
        if train:
            arr = arr.reshape(-1, bands)
            slice_indices = (np.random.rand(int(slice_frac * rows * cols)) * rows * cols).astype(int)
            arr = arr[slice_indices]
        cube.append(arr)

    cube = np.concatenate(cube, axis=0) if train else np.stack(cube, axis=0)
    cube = cube.astype(float)

    return cube


def normalize_cube(
    cube: NDArray,
    spectral_bands: List[str],
    norm_params: Dict={},
    train: bool=True,
) -> Tuple[NDArray, Dict[str, float]]:
    """
    Normalize cube of pixels from COGs in images_dir.
    Arguments:
        cube: Cube of pixels from COGs in images_dir.
        spectral_bands: List of spectral bands to include in cube.
        norm_params: Dictionary of normalization parameters; if empty, compute from cube.
        train: True when normalizing training data.
    Returns: Normalized cube of pixels from COGs in images_dir and dictionary of norm params.
    """
    for band_ix, band in enumerate(spectral_bands):
        if band in norm_params:
            # Get the mean and std of the band from norm_params
            min_, std = norm_params[band]["min"], norm_params[band]["std"]
        else:
            # Otherwise, compute the mean and std of the band
            min_ = cube[:, band_ix].mean() if train else cube[:, :, :, band_ix].mean()
            std = cube[:, band_ix].std() if train else cube[:, :, :, band_ix].std()
        # Normalize the band
        if train:
            # Pixels are unraveled in the first dimension in training mode
            cube[:, band_ix] = (cube[:, band_ix] - min_) / std
        else:
            # Spatial dimensions are preserved in inference mode
            cube[:, :, :, band_ix] = (cube[:, :, :, band_ix] - min_) / std
        norm_params[band] = dict(min=min_, std=std)
    return cube.astype(np.float32), norm_params
