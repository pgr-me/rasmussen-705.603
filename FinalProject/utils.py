#!/usr/bin/python3

# Third-party imports
from xarray.core.dataset import Dataset


def to_float(xx: Dataset) -> Dataset:
    """
    Converts the data type of the dataset to float32.
    Arguments:
        xx {Dataset} -- Dataset to convert.
    Returns: Converted dataset.
    """
    _xx = xx.astype("float32")
    nodata = _xx.attrs.pop("nodata", None)
    if nodata is None:
        return _xx
    return _xx.where(xx != nodata)
