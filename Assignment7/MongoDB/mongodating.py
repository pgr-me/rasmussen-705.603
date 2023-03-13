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
# Local imports
from utils import diff_cols, grid_search, prepare_dataset, pymongo_loads

# File definitions
csv_src = Path("/import/SpeedDatingData.csv")

# Preprocessing inputs
query_cols = {
    "iid": 1,
    "pid": 1,
    "age": 1,
    "age_o": 1,
    "dec": 1,
    "dec_o": 1,
    "attr": 1,
    "attr_o": 1,
    "sinc": 1,
    "sinc_o": 1,
    "intel": 1,
    "intel_o": 1,
    "fun": 1,
    "fun_o": 1,
    "amb": 1,
    "amb_o": 1,
    "shar": 1,
    "shar_o": 1,
    "match": 1,
}
diff_cols_ = ["age", "dec", "attr", "sinc", "intel", "fun", "amb", "shar"]
y_cols = ["match"]

# model inputs
test_size = 0.2
random_state = 777
max_depths = [1, 2, 3, 4, 5, 6, 7, 8]

if __name__ == "__main__":
    print(80 * "~")
    print(f"Load data from {csv_src}")
    collection = pymongo_loads(csv_src, "speeddating", "events")

    print(80 * "~")
    print(f"Prepare data for classifier")
    df = prepare_dataset(collection, query_cols, diff_cols_)

    print(80 * "~")
    print("Split into test and train sets")
    X_tr, X_te, y_tr, y_te = train_test_split(df.drop(axis=1, labels=y_cols), df[y_cols[0]], random_state=random_state, test_size=test_size)

    print(80 * "~")
    print("Conduct cross validation")
    validation_scores, validation_summary = grid_search(X_tr, y_tr, max_depths)
    print("Cross validation results")
    print(validation_summary)

    print(80 * "~")
    print("Select best max depth")
    best_max_depth = validation_summary.index.values[validation_summary["acc_val"].argmax()]
    print(f"Best max depth={best_max_depth}")

    print(80 * "~")
    print("Train model on training data using best max depth")
    best_model = DecisionTreeClassifier(max_depth=best_max_depth)
    best_model.fit(X_tr, y_tr)

    print(80 * "~")
    print("Score model on test data")
    acc_te = round(best_model.score(X_te, y_te), 2)
    print(f"Test accuracy={acc_te}")

    print(80 * "~")
    feature_importances = pd.Series(best_model.feature_importances_, index=list(X_tr)).sort_values(ascending=False)
    print(f"Feature importances:\n{feature_importances}")