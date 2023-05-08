# Final Project: Landcover Change Detection

## Project description

This program runs an end-to-end data pipeline to detect landcover changes using Sentinel-2 satellite imagery. The pipeline consists of the following steps:
* Download satellite imagery for user-selected regions.
* Trains a multilayer perceptron (MLP) to reduce 8-band scenes to two to speed up downstream processes.
* Trains a KMeans classifier to label the 2-channel encoded scenes to simplify the change detection task.
* Change detection: Applies an implementation of Bayesian online changepoint detection (BOCPD) to identify discontinuities in the KMeans-labeled scenes.
* Summarizes change detection outputs into rasters for subsequent analysis.

## Project structure

The project structure is provided below for reference:
```
│   .env
│   1-Download.ipynb
│   2-Train.ipynb
│   3-Infer.ipynb
│   4-Cluster.ipynb
│   5-Detect.ipynb
│   6-Annotate.ipynb
│   bocpd.py
│   clustering.py
│   detect.py
│   Dockerfile
│   download.py
│   readme.md
│   requirements.txt
│   task.py
│   utils.py
│   __init__.py
│
├───figs
│       Change detection plot.png
│       Change pixel.png
│       Euclidean-distance-clusters.png
│       Mahalanobis-distance-clusters.png
│       Manhattan-distance-clusters.png
│       Non-change pixel BOCPD.png
│       Zoomed-in-scatter.png
│
├───mlp
│       datasets.py
│       mlp.py
│       train.py
│       __init__.py
│
└───regions
        af-kharkamar-2022.geojson
        gm-kanifing-2022.geojson
        in-cianjur-2022.geojson
        mg-farafangana-2022.geojson
        tr-islahiye-2023.geojson
        us-baltimore-9999.geojson
```

### Notebooks

There are five notebooks, each numbered in order of execution, that allow the user to interactively explore the program. Each notebook maps to a step in the `task.py` module.

### `.env` file

A `.env` file is required to run the program. An example of one is below. All of the API keys are free to obtain.

```
USGS_API_KEY=your-key
USGS_TOKEN_NAME=your-token
USGS_USERNAME=your-usernam
USGS_PASSWORD=your-password
AWS_ACCESS_KEY=your-aws-access-key
AWS_SECRET_KEY=your-aws-secret-key
NASA_EARTHDATA_S3_ACCESS_KEY=your-nasa-access-key
NASA_EARTHDATA_S3_SECRET_KEY=your-nasa-earthdata-s3-key
NASA_EARTHDATA_S3_SESSION=your-nasa-session-key
NASA_EARTHDATA_USERNAME=your-nasa-earthdata-username
NASA_EARTHDATA_PASSWORD=your-nasa-earthdata-password
PLANETARY_COMPUTER_API_KEY=your-planetary-computer-api-key
PLANETARY_COMPUTER_SERVER=your-planetary-computer-server
```

## Project execution

The project is driven by the `task.py` module, which executes each task. The user may skip any of the tasks in `task.py` using command line arguments.

The user has the following options at the command line.
```
options:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Path to data directory.
  --cores CORES         Number of cores to use.
  --skip_download       True to skip downloading COGs.
  --stac_endpoint STAC_ENDPOINT
                        STAC enpoint URL.
  --collections COLLECTIONS [COLLECTIONS ...]
                        List of STAC collections to download from.
  --output_bands OUTPUT_BANDS [OUTPUT_BANDS ...]
                        List of bands to retain.
  --skip_train_mlp      True to skip training of MLP.
  --tr_val_split TR_VAL_SPLIT
                        Fraction of training v. validation set.
  --batch_size BATCH_SIZE [BATCH_SIZE ...]
                        MLP batch size.
  --epochs EPOCHS       MLP epochs.
  --gamma GAMMA         MLP gamma.
  --skip_mlp_inference  True to skip MLP inference.
  --skip_clustering     True to skip KMeans task.
  --skip_change_detection     True to skip change detection task.
```

## Docker: Local runs

Rather than clone the repository to run the code, the user can opt to pull Docker image for this assignment
from [the DockerHub repo](https://hub.docker.com/repository/docker/pgrjhu/705.603/general).

Pull the image: `$ docker pull pgrjhu/705.603:final`

Instantiate the container:

```
docker run \
-p 8888:8888 \
-dit \
--name final \
pgrjhu/705.603:final
```

Enter the container:
```
docker exec -ti final bash
```
### Running notebooks
In the container, to run the notebooks, you'll need to start Jupyter notebook:
```
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Then open up `localhost:8888` in your browser, enter the key from the terminal, and open the desired notebook.

### Running `.py` file

In `/work`, execute `python task.py` to reproduce the local results.

## Docker: Local runs

To build the container and have Jupyter notebook with GPU (recommended for MLP training):
```
docker run \
    --gpus all \
    --restart=unless-stopped \
    -it \
    -p 8888:8888 -p 8787:8787 -p 8786:8786 -p 5000:5000 \
    [image_id]
```

## Jupyter

To start a Jupyter Lab session, do:
`$ jupyter lab --no-browser --ip=0.0.0.0 --allow-root`
