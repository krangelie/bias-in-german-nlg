import os
import random

import hydra.utils
from bidict import bidict

import src.constants as constants


def prepare_texts(params):
    target_texts_all = get_target_texts(params)
    if constants.DEMO not in params.trigger_position:
        (
            neg_demo_neg_target_texts,
            neg_demo_neu_target_texts,
            neg_demo_pos_target_texts,
            pos_demo_neg_target_texts,
            pos_demo_neu_target_texts,
            pos_demo_pos_target_texts,
            neg_names,
            pos_names,
        ) = add_prefix_to_texts(params, target_texts_all)

    else:
        if isinstance(target_texts_all, list):
            neg_target_texts, pos_target_texts, neu_target_texts = target_texts_all
            neg_demo_neg_target_texts = neg_target_texts
            pos_demo_neg_target_texts = neg_target_texts
            pos_demo_pos_target_texts = pos_target_texts
            neg_demo_pos_target_texts = pos_target_texts
            pos_demo_neu_target_texts = neu_target_texts
            neg_demo_neu_target_texts = neu_target_texts
        elif isinstance(target_texts_all, bidict):
            neg_demo_neg_target_texts = target_texts_all[params.neg_demographic][
                "negative"
            ]
            pos_demo_neg_target_texts = target_texts_all[params.pos_demographic][
                "negative"
            ]
            pos_demo_pos_target_texts = target_texts_all[params.pos_demographic][
                "positive"
            ]
            neg_demo_pos_target_texts = target_texts_all[params.neg_demographic][
                "positive"
            ]
            pos_demo_neu_target_texts = target_texts_all[params.pos_demographic][
                "neutral"
            ]
            neg_demo_neu_target_texts = target_texts_all[params.neg_demographic][
                "neutral"
            ]
            print("neg demo neg target text:", neg_demo_neg_target_texts[0])
            print("pos demo pos target text:", pos_demo_pos_target_texts[0])

    return (
        neg_demo_neg_target_texts,
        neg_demo_neu_target_texts,
        neg_demo_pos_target_texts,
        neg_names,
        pos_demo_neg_target_texts,
        pos_demo_neu_target_texts,
        pos_demo_pos_target_texts,
        pos_names,
    )


def add_prefix_to_texts(params, target_texts_all):
    neg_demo_neg_target_texts = []
    pos_demo_neg_target_texts = []
    neg_demo_pos_target_texts = []
    pos_demo_pos_target_texts = []
    neg_demo_neu_target_texts = []
    pos_demo_neu_target_texts = []
    neg_names, pos_names = [], []
    if (
        params.neg_name_file and params.pos_name_file
    ):  # Use names instead of demographic groups.
        neg_name_file = hydra.utils.to_absolute_path(params.neg_name_file)
        pos_name_file = hydra.utils.to_absolute_path(params.pos_name_file)
        assert not isinstance(target_texts_all, dict)
        # Doesn't work for two different genders yet
        neg_target_texts, pos_target_texts, neu_target_texts = target_texts_all
        neg_names = open(neg_name_file, "r").readlines()
        neg_names = [x for x in neg_names if x]
        pos_names = open(pos_name_file, "r").readlines()
        pos_names = [x for x in pos_names if x]
        # If # names is >= batch_size, reset names for each batch_size-th sample.
        # Otherwise, if # names < batch_size, reset names after cycling through all names AND for each batch_size-th sample.
        # Resetting after each batch_size-th sample is just easier for keeping track of loss masking.
        batch_size_mod_number = params.batch_size
        neg_mod_number = min(len(neg_names), params.batch_size)
        pos_mod_number = min(len(pos_names), params.batch_size)
        for idx, l in enumerate(neg_target_texts):
            mod_idx = idx % batch_size_mod_number
            if mod_idx >= neg_mod_number:
                mod_idx = mod_idx % neg_mod_number
            neg_name = neg_names[mod_idx].strip()
            if params.model_type == constants.GPT2:
                neg_demo_neg_target_texts += [neg_name + " " + l]
            elif params.model_type == constants.DIALOGPT:
                neg_demo_neg_target_texts += [l[0] + " " + neg_name + " " + l[1]]

            mod_idx = idx % batch_size_mod_number
            if mod_idx >= pos_mod_number:
                mod_idx = mod_idx % pos_mod_number
            pos_name = pos_names[mod_idx].strip()
            if params.model_type == constants.GPT2:
                pos_demo_neg_target_texts += [pos_name + " " + l]
            elif params.model_type == constants.DIALOGPT:
                pos_demo_neg_target_texts += [l[0] + " " + pos_name + " " + l[1]]

        for idx, l in enumerate(pos_target_texts):
            mod_idx = idx % batch_size_mod_number
            if mod_idx >= neg_mod_number:
                mod_idx = mod_idx % neg_mod_number
            neg_name = neg_names[mod_idx].strip()
            if params.model_type == constants.GPT2:
                neg_demo_pos_target_texts += [neg_name + " " + l]
            elif params.model_type == constants.DIALOGPT:
                neg_demo_pos_target_texts += [l[0] + " " + neg_name + " " + l[1]]

            mod_idx = idx % batch_size_mod_number
            if mod_idx >= pos_mod_number:
                mod_idx = mod_idx % pos_mod_number
            pos_name = pos_names[mod_idx].strip()
            if params.model_type == constants.GPT2:
                pos_demo_pos_target_texts += [pos_name + " " + l]
            elif params.model_type == constants.DIALOGPT:
                pos_demo_pos_target_texts += [l[0] + " " + pos_name + " " + l[1]]

        for idx, l in enumerate(neu_target_texts):
            mod_idx = idx % batch_size_mod_number
            if mod_idx >= neg_mod_number:
                mod_idx = mod_idx % neg_mod_number
            neg_name = neg_names[mod_idx].strip()
            if params.model_type == constants.GPT2:
                neg_demo_neu_target_texts += [neg_name + " " + l]
            elif params.model_type == constants.DIALOGPT:
                neg_demo_neu_target_texts += [l[0] + " " + neg_name + " " + l[1]]

            mod_idx = idx % batch_size_mod_number
            if mod_idx >= pos_mod_number:
                mod_idx = mod_idx % pos_mod_number
            pos_name = pos_names[mod_idx].strip()
            if params.model_type == constants.GPT2:
                pos_demo_neu_target_texts += [pos_name + " " + l]
            elif params.model_type == constants.DIALOGPT:
                pos_demo_neu_target_texts += [l[0] + " " + pos_name + " " + l[1]]

    else:  # Use demographic groups.
        if not isinstance(target_texts_all, dict):
            neg_target_texts, pos_target_texts, neu_target_texts = target_texts_all
            for l in neg_target_texts:
                neg_demo_neg_target_texts += [params.neg_demographic + " " + l]
                pos_demo_neg_target_texts += [params.pos_demographic + " " + l]
            for l in pos_target_texts:
                neg_demo_pos_target_texts += [params.neg_demographic + " " + l]
                pos_demo_pos_target_texts += [params.pos_demographic + " " + l]
            for l in neu_target_texts:
                neg_demo_neu_target_texts += [params.neg_demographic + " " + l]
                pos_demo_neu_target_texts += [params.pos_demographic + " " + l]
        else:
            neg_demo_neg_target_texts = [
                params.neg_demographic + " " + l
                for l in target_texts_all[params.neg_demographic]["negative"]
            ]
            pos_demo_neg_target_texts = [
                params.pos_demographic + " " + l
                for l in target_texts_all[params.pos_demographic]["negative"]
            ]
            neg_demo_pos_target_texts = [
                params.neg_demographic + " " + l
                for l in target_texts_all[params.neg_demographic]["positive"]
            ]
            pos_demo_pos_target_texts = [
                params.pos_demographic + " " + l
                for l in target_texts_all[params.pos_demographic]["positive"]
            ]
            neg_demo_neu_target_texts = [
                params.neg_demographic + " " + l
                for l in target_texts_all[params.neg_demographic]["neutral"]
            ]
            pos_demo_neu_target_texts = [
                params.pos_demographic + " " + l
                for l in target_texts_all[params.pos_demographic]["neutral"]
            ]

    return (
        neg_demo_neg_target_texts,
        neg_demo_neu_target_texts,
        neg_demo_pos_target_texts,
        pos_demo_neg_target_texts,
        pos_demo_neu_target_texts,
        pos_demo_pos_target_texts,
        neg_names,
        pos_names,
    )


def get_target_texts(params):
    file_paths = [
        params.neg_sample_file,
        params.pos_sample_file,
        params.neu_sample_file,
    ]

    if params.flip_gender:
        # for gender flipped text files
        target_texts_all = {}
        path_dict = {}
        for path in file_paths:
            path = hydra.utils.to_absolute_path(path)
            for gendered_text_file in os.listdir(path):
                # save by gender in dict
                split_name = gendered_text_file.split("_")
                if split_name[0] not in path_dict.keys():
                    path_dict[split_name[0]] = {}
                path_dict[split_name[0]][split_name[1]] = os.path.join(
                    path, gendered_text_file
                )
        for gender in path_dict.keys():
            target_texts_all[constants.FILE_NAME_DICT[gender]] = read_texts_from_files(
                params, path_dict[gender]
            )
        print(path_dict)
        print(target_texts_all)
    else:
        target_texts_all = read_texts_from_files(params, file_paths)
    return target_texts_all


def read_texts_from_files(params, file_paths):
    if not isinstance(file_paths, dict):
        target_texts_all = []
        for file in file_paths:
            file = hydra.utils.to_absolute_path(file)
            print(file)
            with open(file, "r") as f:
                target_texts = f.readlines()
                print(target_texts)
                if params.model_type == constants.GPT2:
                    target_texts = [l.strip() for l in target_texts]
                elif params.model_type == constants.DIALOGPT:
                    target_texts = [l.strip().split("\t") for l in target_texts]
                target_texts_all += [target_texts]
        if params.shuffle_seed:
            random.Random(params.shuffle_seed).shuffle(target_texts)
    else:
        target_texts_all = {}
        for valence, file in file_paths.items():
            with open(file, "r") as f:
                target_texts = f.readlines()
                if params.model_type == constants.GPT2:
                    target_texts = [l.strip() for l in target_texts]
                elif params.model_type == constants.DIALOGPT:
                    target_texts = [l.strip().split("\t") for l in target_texts]
                if params.shuffle_seed:
                    random.Random(params.shuffle_seed).shuffle(target_texts)
                target_texts_all[valence] = target_texts
    return target_texts_all


def strip_bias_context(
    neg_demo_neg_target_texts,
    neg_demo_neu_target_texts,
    neg_demo_pos_target_texts,
    pos_demo_neg_target_texts,
    pos_demo_neu_target_texts,
    pos_demo_pos_target_texts,
):
    # When the trigger encapsulates the bias contexts, we strip bias contexts in the target texts.
    for bc in constants.GPT2_BIAS_CONTEXTS:
        pos_demo_pos_target_texts = [
            x.replace(bc, "").strip() for x in pos_demo_pos_target_texts
        ]
        neg_demo_neg_target_texts = [
            x.replace(bc, "").strip() for x in neg_demo_neg_target_texts
        ]
        pos_demo_neg_target_texts = [
            x.replace(bc, "").strip() for x in pos_demo_neg_target_texts
        ]
        neg_demo_pos_target_texts = [
            x.replace(bc, "").strip() for x in neg_demo_pos_target_texts
        ]
        pos_demo_neu_target_texts = [
            x.replace(bc, "").strip() for x in pos_demo_neu_target_texts
        ]
        neg_demo_neu_target_texts = [
            x.replace(bc, "").strip() for x in neg_demo_neu_target_texts
        ]
    return (
        neg_demo_neg_target_texts,
        neg_demo_neu_target_texts,
        neg_demo_pos_target_texts,
        pos_demo_neg_target_texts,
        pos_demo_neu_target_texts,
        pos_demo_pos_target_texts,
    )
