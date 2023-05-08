# Standard library imports
import os
from pathlib import Path
from typing import Dict, List

# Third-party imports
import dask.distributed
import dask.utils
from datacube.utils.cog import write_cog
from dotenv import load_dotenv
import numpy as np
from odc.stac import stac_load
import pandas as pd
import planetary_computer as pc
from pystac_client import Client

# Local imports
from utils import to_float


print("[Download]: Load environment variables from .env file.")
load_dotenv()
USGS_API_KEY = os.environ["USGS_API_KEY"]
USGS_TOKEN_NAME = os.environ["USGS_TOKEN_NAME"]
USGS_USERNAME = os.environ["USGS_USERNAME"]
USGS_PASSWORD = os.environ["USGS_PASSWORD"]
AWS_ACCESS_KEY = os.environ["AWS_ACCESS_KEY"]
AWS_SECRET_KEY = os.environ["AWS_SECRET_KEY"]
NASA_EARTHDATA_S3_ACCESS_KEY = os.environ["NASA_EARTHDATA_S3_ACCESS_KEY"]
NASA_EARTHDATA_S3_SECRET_KEY = os.environ["NASA_EARTHDATA_S3_SECRET_KEY"]
NASA_EARTHDATA_S3_SESSION = os.environ["NASA_EARTHDATA_S3_SESSION"]
NASA_EARTHDATA_USERNAME = os.environ["NASA_EARTHDATA_USERNAME"]
NASA_EARTHDATA_PASSWORD = os.environ["NASA_EARTHDATA_PASSWORD"]

RES = 10
STAC_ENDPOINT = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTIONS = ["sentinel-2-l2a"]
COLLECTION_BANDS = ["blue", "green", "red", "nir08", "swir16", "swir22", "qa"]
OUTPUT_BANDS = ["blue", "green", "red", "nir08", "swir16", "swir22", "ndvi", "qa"]

os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "FALSE"


def download_cogs(
    regions: pd.DataFrame,
    cogs_dir: Path,
    cfg: Dict,
    stac_endpoint: str=STAC_ENDPOINT,
    collections: List[str]=COLLECTIONS,
    output_bands: List[str]=OUTPUT_BANDS,
    ):
    """
    Download COGs for regions.
    Arguments:
        regions: Regions dataframe.
        cogs_dir: Path to COGs directory.
        cfg: Datacube configuration.
        stac_endpoint: STAC endpoint URL.
        collections: STAC collections list.
        output_bands: List of output bands.
    """
    regions["time_range"] = regions["s2_start"] + "/" + regions["s2_end"]

    for index in regions.index.values:
        event_key = regions.loc[index]["event_key"]
        time_range = regions.loc[index]["time_range"]
        bbox_4326 = regions.bounds.loc[index].values.tolist()
        print(f"[{event_key}]: {time_range}.")
        
        print(f"[{event_key}]: Search catalog.")
        catalog = Client.open(stac_endpoint)
        query = catalog.search(
            collections=collections,
            datetime=time_range,
            bbox=bbox_4326,
        )
        items = list(query.get_items())
        print(f"[{event_key}]: Found {len(items)} items.")
        
        items = [item for item in items if item.properties["eo:cloud_cover"] < 30]
        print(f"[{event_key}]: Selected {len(items)} items.")
        
        print(f"[{event_key}]: Load items into data cube.")
        xx = stac_load(
            items,
            bands=COLLECTION_BANDS,
            resolution=RES,
            chunks={"x": 1028, "y": 1028},
            stac_cfg=cfg,
            patch_url=pc.sign,
            crs="utm",
            bbox=bbox_4326,
            fail_on_error=False,   
        )
        nir08 = to_float(xx.nir08)
        red = to_float(xx.red)
        ndvi = ((nir08 - red) / (nir08 + red)).fillna(0) * 10000
        xx["ndvi"] = ndvi
        # xx["qa"] = (xx["qa"] > 0).astype("uint8")
        
        print(f"[{event_key}]: Re-order bands.")
        xx = xx[output_bands].astype(np.int32)
        n_files = len(xx.time.data)
        print(f"[{event_key}]: Write {n_files} TIFs.")

        for i in range(n_files):
            date = xx.isel(time=i).time.dt.strftime("%Y-%m-%d").data
            dst = cogs_dir / f"{event_key}_{date}.tif"
            try:
                arr = xx.isel(time=i).to_array()
                write_cog(geo_im=arr, fname=dst, overwrite=False).compute()
                print(f"[{event_key}]: Wrote {dst.name}.")
            except Exception as e:
                print(f"[{event_key}]: Failed to write {dst.name}.")
                print(f"[{event_key}]: {e}")
    
