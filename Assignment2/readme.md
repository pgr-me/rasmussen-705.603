# Assignment 2

This assignment preprocesses an image for a YOLO classifier. It loads blue-green-red (BGR) images from a directory, converts each to red-green-blue (RGB), resizes it, normalizes it, converts it to a grayscale image, plots the output, and then saves each stage of transformation, from input to final output.

## Data

Input data for this assignment is in the `data/Assignment2/raw/` directory of this repository and output data is in `data/Assignment2/processed`.

## Reproducing the results
* You can run the preprocessing interactively via`module3_prasmus3.ipynb`.
* To run at the command line, execute:
```
$python preprocess.py -i <path-to-source-image-dir> -o <path-to-dst-image-dir> -d <width, height tuple>
```

To view the help menu, execute:
```
$ python preprocess.py -h
usage: preprocess.py [-h] [-i SRC_DIR] [-o DST_DIR] [-d DSIZE [DSIZE ...]]

optional arguments:
  -h, --help            show this help message and exit
  -i SRC_DIR, --src_dir SRC_DIR
  -o DST_DIR, --dst_dir DST_DIR
  -d DSIZE [DSIZE ...], --dsize DSIZE [DSIZE ...]
```

## Additional features include:
* Parameterization of normalization means and standard deviations.
* Parameterization of method of grayscaling.
* Documentation and assertion tests for each function.
* Parameterized plotting function.
* Saving of input, intermediate, and final results.