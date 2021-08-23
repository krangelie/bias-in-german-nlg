import os

import numpy as np
import pandas as pd
import hydra

from src.classifier.data_processing.annotate.metrics import (
    fleiss_kappa,
    get_all_pairwise_kappas,
)


def create_combined_df(data_dir):
    data_dir = hydra.utils.to_absolute_path(data_dir)
    annotations = pd.DataFrame()
    annotator_names = []
    for i, annotation in enumerate(os.listdir(data_dir)):
        annotator = annotation.split("_")[-1].split(".")[0]
        annotator_names += [annotator]
        data = pd.read_csv(os.path.join(data_dir, annotation), index_col=0)
        if "Unsicher" in data.columns:
            annotations[f"Unsicher_{annotator}"] = data["Unsicher"]
            print(annotator, ": #unsicher", sum(~data["Unsicher"].isna()))
            # print(f'{annotator} not sure about {data['Unsicher']} sentences.')
            annotations[annotator] = data["Label"].fillna(98)
            annotations.loc[
                ~annotations[f"Unsicher_{annotator}"].isna(), annotator
            ] = 98
            annotations[annotator] = annotations[annotator].astype("int32")
        if i == 0:
            annotations["Text"] = data["Text"]
            annotations["Gender"] = data["Gender"]
    return annotations, annotator_names


def clean_uncertain_labels(remove_uncertain, annotations, annotator_names):
    if remove_uncertain == "all":
        min_uncertain = 1
    else:
        min_uncertain = 2

    rm_cases = annotations.loc[
        np.sum(annotations[annotator_names] == 98, axis=1) >= min_uncertain,
        annotator_names,
    ].index
    annotations_cleaned = annotations.drop(
        annotations.loc[rm_cases, annotator_names].index
    )

    annotations_cleaned = annotations_cleaned.replace(98, np.nan)
    print(f"Dropping {len(rm_cases)} cases.")
    return annotations_cleaned


def label_with_aggregate_annotation(
    annotation,
    label_col,
    annotations,
    annotator_names,
    force_majority=False,
):
    if annotation == "majority" or force_majority:
        return_df = _get_majority_label(
            annotations,
            annotator_names,
            label_col,
            for_stratification_only=force_majority,
        )
    else:
        not_all_equal_idcs = []
        for i, row in annotations[annotator_names].iterrows():
            e = _all_equal(row)
            if e is False:
                not_all_equal_idcs += [i]
        all_equal_indcs = list(
            set(annotations.index.values.tolist()) - set(not_all_equal_idcs)
        )
        return_df = _get_majority_label(
            annotations.loc[all_equal_indcs, :],
            annotator_names,
            label_col,
            for_stratification_only=force_majority,
        )

        print(
            f"Removed {len(not_all_equal_idcs)} with varying votes. {len(all_equal_indcs)} unanimously labeled sentences remain."
        )

    # Check inter rater reliability
    fleiss_kappa(return_df, annotator_names)
    get_all_pairwise_kappas(return_df, annotator_names)

    return return_df


def _all_equal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)


def _get_majority_label(
    annotations,
    annotator_names,
    label_col,
    for_stratification_only,
):
    annotations[label_col] = annotations[annotator_names].mode(axis="columns")[0]
    if for_stratification_only and 98 in annotations[label_col]:
        from random import choice

        options = annotations[label_col].drop(98)
        annotations.loc[annotations[label_col] == 98, label_col] = choice(
            options
        )  # remove unsicher

    return annotations
