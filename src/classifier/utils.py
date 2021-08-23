import os
import pickle

import numpy as np
import pandas as pd
import hydra
from omegaconf import OmegaConf


def store_preds(plot_path, name_str, preds_all, labels_all, texts):
    os.makedirs(plot_path, exist_ok=True)
    preds_df = pd.DataFrame()
    preds_df["Y_pred"] = preds_all.tolist()
    preds_df["Y_true"] = labels_all.tolist()
    preds_df["Text"] = texts.tolist()
    preds_df.to_csv(os.path.join(plot_path, f"predictions_{name_str}.csv"))


def build_experiment_name(cfg, fold=None, f_ending=None):
    name = f"{cfg.embedding.name}_embeddings"
    name += f"_{cfg.dev_settings.annotation}_annotated"
    if cfg.pre_processing.tune:
        name += f"_tuned{cfg.pre_processing.epochs}"
    if cfg.pre_processing.use_tfidf:
        name += "_tfidf"
    if not cfg.pre_processing.mean:
        name += "_words"
    else:
        name += "_sentence"

    if fold is not None:
        name += f"_fold-{fold}"
    if f_ending is not None:
        name += f_ending
    return name


def set_path_in_cfg(cfg, split, fold=None):
    OmegaConf.set_struct(cfg, False)  # allows overriding conf
    if split == "dev":
        split_cfg = cfg.dev_settings
    elif split == "test":
        split_cfg = cfg.test_settings
    print(split_cfg)
    if fold is None:
        path = os.path.join(
            get_data_dir(cfg, split_cfg),
            f"{split}_split",
        )
    else:
        path = os.path.join(
            get_data_dir(cfg, split_cfg),
            f"fold_{fold}",
            f"{split}_split",
        )

    print(f"Using {path} for {split} split.")
    if split == "dev":
        cfg.dev_data = path
    else:
        cfg.test_data = path
    return path


def get_splits_dict(cfg, fold=None):
    splits = {}
    for split in ["dev", "test"]:
        p = set_path_in_cfg(cfg, split, fold)
        if p == "":
            raise SystemExit(
                "Paths to both dev and test set must be correctly specified."
            )
        data_dict = pickle.load(open(p, "rb"))
        X, Y, texts = data_dict["X"], data_dict["Y"], data_dict["texts"]
        if -1.0 in Y.unique():
            Y += 1
        splits[f"X_{split}"] = X
        splits[f"Y_{split}"] = Y
        splits[f"texts_{split}"] = texts
        splits[f"texts_{split}"] = texts

    # X_dev, Y_dev, texts_dev, X_test, Y_test, texts_test = splits

    # check if there is any overlap between dev and test set
    # this can happen if different participants gave the same textual response
    common = _common_member(splits["texts_dev"], splits["texts_test"])
    if not common:
        return splits

    elif isinstance(common, list):
        print(f"Old test split length is {len(splits['X_test'])}")
        # if there are overlaps, remove respective elements from the test set
        splits = _remove_elements_by_text(common, splits)
        print(f"New test split length is {len(splits['X_test'])}")
        return splits


def get_data(cfg):
    if cfg.k_fold:
        splits_per_fold = {}
        for fold in range(cfg.k_fold):
            print("\nLOAD FOLD", fold, "\n")
            splits_per_fold[f"fold_{fold}"] = get_splits_dict(cfg, fold)
        return splits_per_fold
    else:
        return get_splits_dict(cfg)


def get_data_dir(cfg, split_settings=None):
    emb_name = cfg.embedding.name
    data_settings = cfg.dev_settings if not split_settings else split_settings
    preprocessed_path = hydra.utils.to_absolute_path(cfg.paths.preprocessed_path)

    if emb_name != "transformer":
        print(data_settings)
        dest_dir = os.path.join(
            preprocessed_path,
            f"emb-{emb_name}"
            f"_tuned-{cfg.pre_processing.tune}{cfg.pre_processing.epochs}"
            f"_uncertainremoved-{cfg.pre_processing.remove_uncertain}"
            f"_annotated-{data_settings.annotation}"
            f"_tfidf-{cfg.pre_processing.use_tfidf}"
            f"_avgemb-{cfg.pre_processing.mean}"
            f"_balanced-{cfg.balance_on_majority}"
            f"_gendered-{data_settings.augment}",
        )
    else:
        dest_dir = os.path.join(
            preprocessed_path,
            f"emb-{emb_name}"
            f"_uncertainremoved-{cfg.pre_processing.remove_uncertain}"
            f"_annotated-{data_settings.annotation}"
            f"_balanced-{cfg.balance_on_majority}"
            f"_gendered-{data_settings.augment}",
        )
    os.makedirs(dest_dir, exist_ok=True)
    return dest_dir


def _common_member(a, b):
    a_set = set(a)
    b_set = set(b)
    if a_set & b_set:
        print("There is at least one duplicate in dev and test split")
        common = list(a_set.intersection(b_set))
        print(common)
        if len(common) > 10:
            SystemExit("There are more than 10 duplicates. Check your " "splits.")
        else:
            return common
    else:
        return False


def _remove_elements_by_text(elements, splits_dict):
    print(f"Removing {len(elements)} cases from test " f"split")
    orig_len = len(splits_dict["X_test"])
    for e in elements:
        idx = splits_dict["texts_test"][splits_dict["texts_test"] == e].index
        abs_idx = [splits_dict["texts_test"].index.get_loc(i) for i in idx]
        splits_dict["X_test"] = np.delete(splits_dict["X_test"], abs_idx, axis=0)
        splits_dict["Y_test"] = splits_dict["Y_test"].drop(idx)
        splits_dict["texts_test"] = splits_dict["texts_test"].drop(idx)

    # check if removal was successful
    assert orig_len - len(elements) == len(splits_dict["X_test"])
    assert (
        len(splits_dict["X_test"])
        == len(splits_dict["Y_test"])
        == len(splits_dict["texts_test"])
    )
    assert not _common_member(splits_dict["texts_dev"], splits_dict["texts_test"])
    return splits_dict
