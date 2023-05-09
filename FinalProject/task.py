# Standard library imports
import argparse
from datetime import datetime as dt
import json
import os
import multiprocessing
from pathlib import Path
import warnings

# Third party imports
import geopandas as gpd
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils.metric import type_metric, distance_metric
import rasterio as rio
from scipy.spatial.distance import mahalanobis
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
from download import download_cogs
from mlp import MLP, MLPDataset, train_test_ae_mlp, encode
from clustering import label_scenes, make_cluster_training_set
from bocpd import bocd, GaussianUnknownMean, plot_posterior
from detect import parallel_bocd

# Data dir for users of Docker
DATA_DIR = Path("/work/data")
# Parallelization params
CORES = multiprocessing.cpu_count() - 1
# Download constants
RES = 10
STAC_ENDPOINT = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTIONS = ["sentinel-2-l2a"]
COLLECTION_BANDS = ["blue", "green", "red", "nir08", "swir16", "swir22", "qa"]
OUTPUT_BANDS = ["blue", "green", "red", "nir08", "swir16", "swir22", "ndvi", "qa"]
# Train MLP constants
BATCH_SIZE = 1028
EPOCHS = 30
GAMMA = 0.1
LAT_SPACE_SIZES = [2, 4, 16, 64, 256,]
LRS = [0.5, 0.1]
STEP_SIZES = [5, 10, 15]
TR_VAL_SPLIT = 0.9
WEIGHT_DECAY = 5e-4
# KMeans constants
K = 4
region_sample_size = 5000
# Change detection constants
BOCD_PARAMS = dict(hazard=1/100, mean0=1, var0=2, varx=1,)
WINDOW = 10
START_INDEX = 10
N_STDS = 1.75



def parser():
    parser = argparse.ArgumentParser(description="Run the pipeline.")
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR, help="Path to data directory.")
    parser.add_argument("--cores", type=int, default=1, help="Number of cores to use.")
    # Download args
    parser.add_argument("--skip_download", action="store_true", help="True to skip downloading COGs.")
    parser.add_argument("--stac_endpoint", type=str, default=STAC_ENDPOINT, help="STAC enpoint URL.")
    parser.add_argument("--collections", nargs="+", default=COLLECTIONS, help="List of STAC collections to download from.")
    parser.add_argument("--output_bands", nargs="+", default=OUTPUT_BANDS, help="List of bands to retain.")
    # Train MLP args
    parser.add_argument("--skip_train_mlp", action="store_true", help="True to skip training of MLP.")
    parser.add_argument("--tr_val_split", type=float, default=TR_VAL_SPLIT, help="Fraction of training v. validation set.")
    parser.add_argument("--batch_size", nargs="+", default=BATCH_SIZE, help="MLP batch size.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="MLP epochs.")
    parser.add_argument("--gamma", type=float, default=GAMMA, help="MLP gamma.")
    # MLP inference args
    parser.add_argument("--skip_mlp_inference", action="store_true", help="True to skip MLP inference.")
    # KMeans args
    parser.add_argument("--skip_clustering", action="store_true", help="True to skip KMeans task.")
    # Change detection args
    parser.add_argument("--skip_change_detection", action="store_true", help="True to skip change detection task.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parser()
    data_dir = args.data_dir
    cores = args.cores
    # Primary data dirs
    raw_dir = data_dir / "raw"
    interim_dir = data_dir / "interim"
    processed_dir = data_dir / "processed"
    # Download dir
    cogs_dir = interim_dir / "cogs"
    # MLP training dirs
    models_dir = interim_dir / "models"
    scores_dir = interim_dir / "scores"
    # MLP inference dirs
    encoded_dir = interim_dir / "encoded"
    meta_dir = interim_dir / "meta"
    # Cluster dirs
    cluster_dir = interim_dir / "cluster"
    # Make dirs
    for dir_ in [cogs_dir, models_dir, scores_dir, encoded_dir, meta_dir, cluster_dir]:
        dir_.mkdir(parents=True, exist_ok=True)
        
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Download task
    if not args.skip_download:
        os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "FALSE"
        print("[Download]: Load input region geojson and config.")
        with open(raw_dir / "cfg.json")  as f:
            cfg = json.load(f)
        regions = gpd.read_file(raw_dir / "regions.geojson")
        regions["time_range"] = regions["s2_start"] + "/" + regions["s2_end"]
        download_cogs(
            regions, cogs_dir, cfg,
            stac_endpoint=args.stac_endpoint,
            collections=args.collections,
            output_bands=args.output_bands,
        )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Train MLP task
    if not args.skip_train_mlp:
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Create PyTorch datasets and loaders
        dataset = MLPDataset(cogs_dir, args.output_bands, region="", train=True, slice_frac=0.01)
        tr_val_size = int(args.tr_val_split * len(dataset))
        te_size = len(dataset) - tr_val_size
        tr_size = int(args.tr_val_split * tr_val_size)
        val_size = tr_val_size - tr_size
        # Make datasets
        tr_val_dataset, te_dataset = torch.utils.data.random_split(dataset, [tr_val_size, te_size])
        tr_dataset, val_dataset = torch.utils.data.random_split(tr_val_dataset, [tr_size, val_size])
        # Make loaders
        tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False)
        # Train MLP
        _, in_features = dataset.cube.shape
        tr_val_losses = []
        te_losses = []
        for step_size in STEP_SIZES:
            for lr in LRS:
                print(f"Train model with step_size={step_size} and lr={lr}.")
                model = MLP(
                    in_features=in_features,
                    hidden_neurons=2,
                    )
                model.to(device)
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY, fused=True)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=GAMMA)
                tr_val_losses_, te_losses_ = train_test_ae_mlp(
                    model,
                    tr_loader, val_loader, te_loader,
                    criterion, optimizer, scheduler,
                    lr, step_size, args.epochs,
                    models_dir, scores_dir,
                    device,
                )
                tr_val_losses.extend(tr_val_losses_)
                te_losses.extend(te_losses_)
        tr_val_summary = pd.DataFrame(tr_val_losses)
        te_summary = pd.DataFrame(te_losses)
        tr_val_summary.to_csv(scores_dir / "tr_val_summary.csv")
        te_summary.to_csv(scores_dir / "te_summary.csv")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # MLP inference task
    if not args.skip_mlp_inference:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        regions = gpd.read_file(args.raw_dir / "regions.geojson").set_index("event_key")
        # Find best model and load it
        tr_val_summary = pd.read_csv(args.scores_dir / "tr_val_summary.csv")
        te_summary = pd.read_csv(args.scores_dir / "te_summary.csv")
        optimal_step_size, optimal_lr = te_summary.sort_values(by="te_loss_per_batch").iloc[0][["step_size", "lr"]].values
        model_src = models_dir / f"mlp-autoencoder-{int(optimal_step_size)}-{optimal_lr:.3f}.pt"
        for region, region_meta in regions.iterrows():
            # Load dataset
            dataset = MLPDataset(cogs_dir, args.output_bands, region=region, train=False, slice_frac=0.01)
            scenes, rows, cols, in_features = dataset.cube.shape
            scenes = dataset.cube.shape[0]
            loader = DataLoader(dataset, batch_size=128, shuffle=False)
            # Load model
            in_features = dataset.cube.shape[-1]
            model = MLP(in_features=in_features, hidden_neurons=2)
            model.load_state_dict(torch.load(model_src))
            model.to(device)
            model.eval()
            # Encode: Reduce dimensionality of cube
            encoded = encode(loader, model, scenes, rows, cols, device)
            # Save
            encoded_dst = encoded_dir / f"{region}.npy"
            meta_dst = meta_dir / f"{region}.csv"
            np.save(encoded_dst, encoded)
            dataset.cogs_meta.to_csv(meta_dst, index=False)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Cluster using KMeans and label scenes
    if not args.skip_clustering:
        print(f"[Cluster]: Load regions.")
        regions = gpd.read_file(raw_dir / "regions.geojson").set_index("event_key")

        print(f"[Cluster]: Prepare cluster training set.")
        X = make_cluster_training_set(regions, encoded_dir, region_sample_size=region_sample_size)

        print(f"[Cluster]: Train KMeans.")
        VI = np.linalg.inv(np.cov(X.T))
        metric = distance_metric(type_metric.USER_DEFINED, func=lambda x, y: mahalanobis(x, y, VI))
        initial_centers = kmeans_plusplus_initializer(X, K).initialize()
        kmeans_instance = kmeans(X, initial_centers, metric=metric)
        kmeans_instance.process()
        clusters = kmeans_instance.get_clusters()
        final_centers = kmeans_instance.get_centers()
        label_scenes(regions, cluster_dir, encoded_dir, cores, kmeans_instance, cogs_dir)
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Detect change
    if not args.skip_detection:
        regions = gpd.read_file(args.raw_dir / "regions.geojson").set_index("event_key")
        for region, region_meta in regions.iterrows():
            raster_src = cluster_dir / f"{region}.tif"
            with rio.open(raster_src) as src:
                profile = src.profile
                labeled_scenes = src.read()
            timesteps, rows, cols = labeled_scenes.shape
            raveled_scenes = labeled_scenes.reshape(timesteps, -1)
            changepoints = Parallel(n_jobs=cores)(
                delayed(parallel_bocd)
                (raveled_scenes, unipix, BOCD_PARAMS, WINDOW) for unipix in tqdm(range(rows * cols))
            )
            changepoint_arr = np.stack(changepoints, axis=0)

            meta = pd.read_csv(meta_dir / f"{region}.csv")
            meta["solar_date"] = pd.to_datetime(meta["solar_date"])
            meta["ordinal"] =meta["solar_date"].apply(lambda x: dt.toordinal(x))
            year = meta["solar_date"].dt.year
            month = meta["solar_date"].dt.month
            day = meta["solar_date"].dt.day
            fractional_year = (year + month / 12 + day / 30.5).values

            sds = meta["ordinal"].values[changepoint_arr[:, 0]].reshape(rows, cols)
            eds = meta["ordinal"].values[changepoint_arr[:, 1]].reshape(rows, cols)

            sds_fracyear = fractional_year[changepoint_arr[:, 0]].reshape(rows, cols)
            eds_fracyear = fractional_year[changepoint_arr[:, 1]].reshape(rows, cols)

            ref_cog_src = next(cogs_dir.glob(f"{region}*.tif"))
            with rio.open(ref_cog_src) as src:
                profile = src.profile
                
            profile.update(dict(dtype=np.float32, count=2))
            with rio.open(processed_dir / f"{region}.tif", "w", **profile) as dst:
                dst.write(sds_fracyear.astype(np.float32), 1)
                dst.write(eds_fracyear.astype(np.float32), 2)



