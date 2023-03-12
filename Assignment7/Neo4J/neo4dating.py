# Standard library imports
from pathlib import Path
import subprocess
from typing import Any, List, Optional, Tuple, Type

# Third party imports
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier

# Local imports
from utils import execute_cypher, grid_search, parse_cypher_output, prepare_dataset


# Directory and filepath definitions
root_dir = Path("/nosql/neo4j")
scripts_dir = root_dir / "cypher"
data_dir = root_dir / "data"
src = root_dir / "import" / "speeddatingReduced.csv"
schema_viz_cypher_src = scripts_dir / "schema_viz.cypher"
female_male_ratio_cypher_src = scripts_dir / "female_male_ratio.cypher"
match_frac_cypher_src = scripts_dir / "match_frac.cypher"

# Data preprocessing inputs
id_cols = ["dater", "datee"]
y_col = ["match"]
attr_cols = ["int_corr"]
diff_cols = ["age", "race", "attr", "sinc", "intel", "fun", "amb", "shar", "like", "prob", "met"]

# Model inputs
random_state = 777
test_size = 0.2


if __name__ == "__main__":
    print(80 * "~")
    print("Execute cypher queries")
    print("Run Schema visualization query.")
    execute_cypher(schema_viz_cypher_src, data_dir)
    print("Run query to find female: male ratio.")
    execute_cypher(female_male_ratio_cypher_src, data_dir)
    print("Run query to find fraction of dates that resulted in matches.")
    execute_cypher(match_frac_cypher_src, data_dir)
    print(f"Saved outputs to {data_dir}")

    print(80 * "~")
    print(f"Parse cypher outputs written to {data_dir}.")
    schema_viz = parse_cypher_output(data_dir, "schema_viz.out", scalar=False, dtype=str)
    female_male_ratio = parse_cypher_output(data_dir, "female_male_ratio.out")
    match_frac = parse_cypher_output(data_dir, "match_frac.out")

    print(80 * "~")
    print(f"Schema visualization: {schema_viz}")
    print(f"Female: male ratio: {female_male_ratio:2f}")
    print(f"Fraction of dates that resulted in matches: {match_frac:2f}")


    print(80 * "~")
    print(f"Load {src}")
    df_ = pd.read_csv(src)

    print(80 * "~")
    print("Prepare dataset for classifier.")
    df = prepare_dataset(df_, id_cols, y_col, attr_cols, diff_cols)
    print(f"Data has dimensions of {df.shape}")

    print(80 * "~")
    print(f"Split into train and test sets; test size={test_size}.")
    X_tr, X_te, y_tr, y_te = train_test_split(df.drop(axis=1, labels=y_col), df[y_col[0]], random_state=random_state, test_size=test_size)
    max_depths = [2, 4, 6, 8, 10, 12]

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