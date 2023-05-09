# Local imports
from pathlib import Path
# Third party imports
from joblib import Parallel, delayed
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pyclustering.cluster.kmeans import kmeans
import rasterio as rio


def label_scenes(
    regions: pd.DataFrame,
    cluster_dir: Path,
    encoded_dir: Path,
    cores: int,
    kmeans_instance: kmeans,
    cogs_dir: Path,
    ):
    """
    Label scenes.
    Arguments:
        regions: Regions dataframe.
        cluster_dir: Directory to save cluster labels.
        encoded_dir: Directory of encoded scenes.
        cores: Number of cores to use.
        kmeans_instance: Trained KMeans instance.
        cogs_dir: Directory of COGs.
    """
    for region, region_meta in regions.iterrows():

        cluster_dst = cluster_dir / f"{region}.tif"

        print(f"[Cluster][{region}]: Load region dataset and metadata")
        encoded_src = encoded_dir / f"{region}.npy"
        encoded = np.load(encoded_src)
        
        print(f"[Cluster][{region}]: Prep inputs.")
        # Prep inputs
        timesteps, rows, cols, channels = encoded.shape
        
        print(f"[Cluster][{region}]: Labeling pixels...")
        clusters = Parallel(n_jobs=cores)(
            delayed(parallel_cluster)
            (kmeans_instance, encoded, timestep, rows, cols) for timestep in range(timesteps)
        )
        clusters = np.stack(clusters, axis=0)

        print(f"[Cluster][{region}]: Saving cluster labels...")        
        ref_tif_src = next(cogs_dir.glob(f"{region}*.tif"))
        with rio.open(ref_tif_src) as src_ref_tif:
            profile = src_ref_tif.profile
        profile.update(dict(dtype=rio.uint8, count=timesteps))
        with rio.Env():
            with rio.open(cluster_dst, "w", **profile) as dst:
                for band, index in enumerate(range(timesteps), start=1):
                    dst.write(clusters[index].astype(rio.uint8), band)


def make_cluster_training_set(
    regions: pd.DataFrame,
    encoded_dir: Path,
    start_index: int=10,
    region_sample_size: int=5000,
    ):
    """
    Make training set for clustering algorithm.
    Arguments:
        regions: Regions dataframe.
        change_maps_dir: Directory of change maps.
        encoded_dir: Directory of encoded scenes.
        region_sample_size: Number of pixels to sample from each region.
    """

    component1, component2 = [], []
    for region, _ in regions.iterrows():

        print(f"[Cluster]: Processing {region}...")

        # Load region dataset and metadata
        encoded_src = encoded_dir / f"{region}.npy"
        encoded = np.load(encoded_src)[:, :, :, :]  # chop off first N timesteps
        
        # Create mask to remove outliers that wreak havoc on change detection algo
        s1 = pd.Series(encoded[:, :, :, 0].ravel()).sample(region_sample_size)
        s2 = pd.Series(encoded[:, :, :, 1].ravel()).loc[s1.index]
        component1.append(s1)
        component2.append(s2)
        
    component1 = pd.concat(component1)
    component2 = pd.concat(component2)

    X = component1.to_frame().join(component2.rename(1)).values.astype(np.float64)
    mask = (X[:, 0] < -5.1)  # Remove extreme outliers from training set
    # TODO: Create better way to remove outliers than hardcoding based on observation
    return X[~mask]


def parallel_cluster(kmeans_instance: kmeans, encoded: NDArray, timestep: int, rows: int, cols: int) -> NDArray:
    """
    Label pixels for one scene.
    Arguments:
        kmeans_instance: Trained KMeans instance.
        encoded: Encoded scenes.
        timestep: Timestep pertaining to scene to label.
        rows: Number of rows.
        cols: Number of columns.
    Returns: Labeled pixels for one scene.
    """
    return kmeans_instance.predict(encoded[timestep].reshape(-1, 2)).reshape(rows, cols)