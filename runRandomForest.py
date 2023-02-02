import argparse
import logging
import time
import numpy as np
import pandas as pd

from typing import List, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from pandas import DataFrame
from argparse import Namespace
from scipy.io.arff import loadarff, MetaData
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.tree._tree import Tree


def create_preprocessor_transformer(meta_data: MetaData) -> ColumnTransformer:
    transformations = []
    for (n, t) in zip(meta_data.names()[:-1], meta_data.types()[:-1]):
        if not t == "numeric":
            transformations.append(("ordinal_%s" % n, OrdinalEncoder(), [n]))
    ct = ColumnTransformer(transformers=transformations, remainder="passthrough")
    logging.info(ct)
    return ct


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trainingFile")
    parser.add_argument("testingFile")
    parser.add_argument("--estimators", default="10")
    parser.add_argument("--log_level", default="WARNING")

    args = parser.parse_args()

    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % args.log_level)
    logging.basicConfig(filename="random_forest.log", level=numeric_level)
    return args


def calculate_node_information_number(trees: List[Tree]) -> Tuple[float, float]:
    n_leaves = np.zeros(len(trees), dtype=int)
    n_internal_nodes = np.zeros(len(trees), dtype=int)
    for i in range(len(trees)):
        n_nodes = trees[i].tree_.node_count
        children_left = trees[i].tree_.children_left
        for x in range(n_nodes):
            if children_left[x] == -1:
                n_leaves[i] += 1
        n_internal_nodes[i] = n_nodes - n_leaves[i]
    return float(np.mean(n_leaves)), float(np.mean(n_internal_nodes))


def run_random_forest(training_data: DataFrame, test_data: DataFrame, trees: int, meta_data: MetaData):

    start_time = time.time()

    # Set up attribute column transformer and target label encoders and then transform both training and test set,
    # we fit on the full data in case of missing values in either the training or test sets
    full_data = pd.concat([training_data, test_data])
    act = create_preprocessor_transformer(meta_data)
    act.fit(full_data[full_data.columns[:-1]])
    transformed_training_attributes = act.transform(training_data[training_data.columns[:-1]])
    transformed_test_attributes = act.transform(test_data[test_data.columns[:-1]])
    tle = LabelEncoder()
    tle.fit(full_data[full_data.columns[-1]])
    transformed_training_targets = tle.transform(training_data[training_data.columns[-1]])
    transformed_test_targets = tle.transform(test_data[test_data.columns[-1]])

    # Run the RF algorithm
    clf = RandomForestClassifier(n_estimators=trees)
    clf = clf.fit(transformed_training_attributes, transformed_training_targets)
    predictions = clf.predict(transformed_test_attributes)
    end_time = time.time()

    # Generate and print statistics
    accuracy = accuracy_score(transformed_test_targets, predictions)
    leaves, internal_nodes = calculate_node_information_number(clf.estimators_)

    print(f"Classification accuracy on test set: {accuracy}")
    print(f"Number of Rules: {leaves}")
    print(f"Average number of terms: {internal_nodes}")
    print(f"Running time (seconds): {end_time - start_time}")


def main():
    args = parse_args()
    train_data = loadarff(args.trainingFile)
    training_df = DataFrame(train_data[0])
    test_data = loadarff(args.testingFile)
    testing_df = DataFrame(test_data[0])
    run_random_forest(training_df, testing_df, int(args.estimators), train_data[1])


if __name__ == "__main__":
    main()
