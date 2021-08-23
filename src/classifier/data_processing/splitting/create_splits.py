import os
import pickle

from sklearn.model_selection import train_test_split, StratifiedKFold
from src.classifier.data_processing.annotate.annotate_sentences import (
    label_with_aggregate_annotation,
)


def create_or_load_indcs(dcfg, label_col, df, annotator_names):
    if dcfg.balance_on_majority or dcfg.k_fold:
        labels_for_strat = label_with_aggregate_annotation(
            dcfg.annotation,
            label_col,
            df,
            annotator_names,
            force_majority=True,
        )[label_col]
    else:
        labels_for_strat = None
    if dcfg.k_fold:
        dest = os.path.join(dcfg.paths.dev_test_indcs, f"num_folds_{dcfg.k_fold}")
        dev_set, test_set = [], []
        if not os.path.isdir(dest) or not os.listdir(dest):
            os.makedirs(dest, exist_ok=True)
            skf = StratifiedKFold(dcfg.k_fold, random_state=42, shuffle=True)
            for fold, (dev_indices, test_indices) in enumerate(
                skf.split(df.index, labels_for_strat)
            ):
                fold_dest = os.path.join(dest, f"fold_{fold}")
                os.makedirs(fold_dest)
                dump_dev_test(fold_dest, dev_indices, test_indices)
                dev_set.append(df.iloc[dev_indices])
                test_set.append(df.iloc[test_indices])
        else:
            for fold in range(dcfg.k_fold):
                fold_dest = os.path.join(dest, f"fold_{fold}")
                dev_indices, test_indices = load_dev_test(fold_dest)
                dev_set.append(df.iloc[dev_indices])
                test_set.append(df.iloc[test_indices])
    else:
        test_size = dcfg.test_split
        dest = os.path.join(dcfg.paths.dev_test_indcs, f"test_size_{test_size}")
        if not os.path.isdir(dest) or not os.listdir(dest):
            os.makedirs(dest, exist_ok=True)
            dev_indices, test_indices = train_test_split(
                df.index,
                test_size=test_size,
                shuffle=True,
                stratify=labels_for_strat,
                random_state=42,
            )
            dump_dev_test(dest, dev_indices, test_indices)
        else:
            dev_indices, test_indices = load_dev_test(dest)

        dev_set, test_set = df.iloc[dev_indices], df.iloc[test_indices]
    return dev_set, test_set


def load_dev_test(dest):
    with open(os.path.join(dest, "dev_indices.pkl"), "rb") as d:
        dev_indices = pickle.load(d)
    with open(os.path.join(dest, "test_indices.pkl"), "rb") as t:
        test_indices = pickle.load(t)
    return dev_indices, test_indices


def dump_dev_test(dest, dev_indices, test_indices):
    with open(os.path.join(dest, "dev_indices.pkl"), "wb") as d:
        pickle.dump(dev_indices, d)
    with open(os.path.join(dest, "test_indices.pkl"), "wb") as t:
        pickle.dump(test_indices, t)


def get_data_splits(dcfg, label_col, df, annotator_names):
    dev_set, test_set = create_or_load_indcs(dcfg, label_col, df, annotator_names)

    return dev_set, test_set
