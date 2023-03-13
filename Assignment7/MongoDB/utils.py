# Standard library imports
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple
# Third party imports
import pandas as pd
import pymongo
from pymongo import MongoClient
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier


def diff_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Compute differences between dater-datee pairs.
    Args:
        df: Dataset to compute differences for.
        cols: Columns to compute differences for.
    Returns: Dataset with difference columns; deletes columns used to compute diffs.
    """
    for col in cols:
        df[f"{col}_diff"] = (
            pd.to_numeric(df[col], errors="coerce")
            - pd.to_numeric(df[f"{col}_o"], errors="coerce")
        ).abs()
        df.drop(axis=1, labels=[col, f"{col}_o"], inplace=True)
    return df


def grid_search(X_tr: pd.DataFrame, y_tr: pd.DataFrame, max_depths: List[int], k_folds: int=5, random_state: int=777) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform a simple grid search over max depths for a decision tree classifier.
    Args:
        X_tr: Training features.
        y_tr: Training labels.
        max_depths: List of max depths to try.
        k_folds: Number of folds to use for cross-validation.
        random_state: Random state for K-folds splitting and decision tree fitting.
    Returns: Tuple of 1) dataframe of train and validation scores by fold by max depth and 2) mean scores by max_depth
    """
    kf = KFold(n_splits=k_folds, random_state=random_state, shuffle=True)
    validation_scores = []
    for max_depth in max_depths:
        for fold, (ix_tr, ix_te) in enumerate(kf.split(X_tr), start=1):
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
            model.fit(X_tr.iloc[ix_tr], y_tr.iloc[ix_tr])
            acc_tr = model.score(X_tr.iloc[ix_tr], y_tr.iloc[ix_tr])
            acc_val = model.score(X_tr.iloc[ix_te], y_tr.iloc[ix_te])
            validation_scores.append(dict(
                fold=fold,
                max_depth=max_depth,
                acc_tr=acc_tr,
                acc_val=acc_val
            ))
    validation_scores = pd.DataFrame(validation_scores)
    validation_summary = validation_scores.drop(axis=1, labels="fold").groupby("max_depth").mean().round(3)
    return validation_scores, validation_summary


def prepare_dataset(
    collection: pymongo.collection.Collection,
    query_cols: List[str],
    diff_cols_: List[str]
) -> pd.DataFrame:
    """
    Prepare dataset for classifier.
    Args:
        collection: PyMongo collection.
        query_cols: Columns to query for.
        diff_cols: Columns to compute differences for; see diff_cols function.
    """
    df = select_data(collection, query_cols)
    df = diff_cols(df, diff_cols_)
    df.dropna(inplace=True)
    for col in df:
        df[col] = df[col].astype(int)
    df["date"] = df.apply(
        lambda x: f"{x.iid}-{x.pid}" if x.iid < x.pid else f"{x.pid}-{x.iid}",
        axis=1
    )
    df.drop_duplicates(subset="date", inplace=True)
    df.rename(columns=dict(iid="dater", pid="datee"), inplace=True)
    df.set_index(["date", "dater", "datee"], inplace=True)
    return df


def select_data(
        collection: pymongo.collection.Collection,
        query_cols: Dict[str, int]
) -> pd.DataFrame:
    """
    Select data from collection.
    Args:
        collection: PyMongo collection.
        query_cols: Columns to select.
    Returns: Dataframe version of dataset.
    """
    return (
        pd.DataFrame(list(collection.find({}, query_cols)))
        .drop(axis=1, labels="_id")
    )


def pymongo_loads(
        csv_src: Path,
        db_name: str,
        collection_name: str,
        host: str="localhost",
        port: int=27017
) -> pymongo.collection.Collection:
    """
    Load CSV into MongoDB collection.
    Args:
        csv_src: Path to CSV.
        db_name: Name of database.
        collection_name: Name of collection.
        host: Host name.
        port: Port.
    Returns: PyMongo collection populated with CSV data.
    """
    client = pymongo.MongoClient(host, port)
    db = client[db_name]
    collection = db.events
    headers = pd.read_csv(csv_src).columns.tolist()
    with open(csv_src) as csvfile:
        reader = csv.DictReader(csvfile)
        db.segment.drop()
        for each in reader:
            row = {}
            for field in headers:
                row[field] = each[field]
            collection.insert_one(row)
    return collection
