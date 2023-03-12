from pathlib import Path
import subprocess
from typing import Any, List, Optional, Tuple, Type

import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier


def execute_cypher(cypher_src: Path, output_dir: Path, output_fn: Optional[str]=None):
    """
    Execute cypher command and save to specified file.
    Args:
        cypher_src: Path to cypher script file.
        output_dir: Path to cypher output directory.
        output_fn: Output filename
    """
    if output_fn is None:
        output_fn = f"{cypher_src.stem}.out"
    output_dst = output_dir / output_fn
    cmd_str = f"/var/lib/neo4j/bin/cypher-shell -f {cypher_src} > {output_dst}"
    subprocess.run(cmd_str, shell=True)

    
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

def parse_cypher_output(
        data_dir: Path,
        fn: str,
        scalar: bool=True,
        dtype: Type=float
) -> Any:
    """
    Parse cypher output.
    Args:
        data_dir: Path to cypher output data directory.
        fn: Filename.
        scalar: True to treat cypher output as a scalar.
        dtype: Data type to cast output to.
    Returns: Cypher output.
    """
    output_path = data_dir / fn
    with open(output_path) as f:
        output = f.readlines()
    if scalar:
        return dtype(output[-1].split("\n")[0])
    return dtype(output)

def prepare_dataset(df: pd.DataFrame, id_cols: List[str], y_col: List[str], attr_cols: List[str], diff_cols: List[str]) -> pd.DataFrame:
    """
    Clean dataframe and subset to desired columns.
    Args:
        df: Raw speed dating dataframe.
        id_cols: List of ID columns (e.g., dater, datee)
        y_col: One-element y column list.
        attr_cols: Other attribute columns.
        diff_cols: Attribute columns to compute difference for; absolute value taken of diff.
    Returns: Prepared, cleaned, subsetted data.
    """
    other_cols = [f"{x}_o" for x in diff_cols]
    df = df[id_cols + y_col + attr_cols + diff_cols + other_cols].dropna()
    for int_col in id_cols + diff_cols + other_cols + y_col:
        df[int_col] = df[int_col].astype(int)

    for diff_col in diff_cols:
        df[f"{diff_col}_diff"] = (df[diff_col] - df[f"{diff_col}_o"]).abs()
    df["date"] = df.apply(lambda frame: str(int(frame.dater)) +"-"+ str(int(frame.datee)) if frame.dater < frame.datee else str(int(frame.datee)) + "-" + str(int(frame.dater)), axis=1)
    df.drop(axis=1, labels=diff_cols+other_cols, inplace=True)
    df.drop_duplicates("date", inplace=True)
    df.set_index(["dater", "datee", "date"], inplace=True)
    return df
