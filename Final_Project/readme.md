# Change detection algorithm

To build the container and have Jupyter notebook:
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
