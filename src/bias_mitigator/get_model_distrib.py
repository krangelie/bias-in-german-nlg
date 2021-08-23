"""Script to analyze model's generated distribution of words."""

import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter, OrderedDict

import src.constants as constants


THRESHOLD = 4000  # Max samples to analyze per demographic.


def score_analysis(fi):
    """Run score analysis on files."""
    scores = []
    new_lines = []
    f = pd.read_csv(fi)
    for i, row in f.iterrows():
        score = row["Prediction"]
        sentence = row["Text"]
        new_lines.append(sentence)
        scores.append(score)

    lines = new_lines
    if len(scores) != len(lines):
        print(len(scores), len(lines))
    assert len(scores) == len(lines)
    print("Mean score polarity", np.mean(scores))
    print("Stdev", np.std(scores))
    return list(zip(lines, scores))


def plot_score(score_list, label_list, ratio=False):
    """Plot scores."""
    # Expecting many samples in the neutral area.
    all_score_counts = []
    for score_idx in range(len(score_list)):
        scores = score_list[score_idx]
        scores = scores[:THRESHOLD]
        label = label_list[score_idx]
        score_counts = Counter()
        for c in scores:
            if c == 2:
                score_counts["+"] += 1
            elif c == 0:
                score_counts["-"] += 1
            else:
                score_counts["0"] += 1
        if ratio:
            if len(scores):
                score_counts["+"] /= float(len(scores))
                score_counts["-"] /= float(len(scores))
                score_counts["0"] /= float(len(scores))
        ordered_score_counts = [
            round(score_counts["-"], 3),
            round(score_counts["0"], 3),
            round(score_counts["+"], 3),
        ]
        print(
            "%s: %s samples, [neg, neu, pos] ratio = %s"
            % (label, len(scores), ordered_score_counts)
        )
        all_score_counts.append(ordered_score_counts)
    return all_score_counts


def plot_ratios(ordered_score_counts, label_list, title=None, ratio=True, dest=None):
    """Plot sentiment"""
    width = 0.15

    ind = np.arange(3)
    cat = ["before", "after"]
    colors = sns.color_palette("coolwarm")

    for score_idx in range(len(ordered_score_counts)):
        scores = ordered_score_counts[score_idx]
        label = label_list[score_idx]

        if score_idx >= 2:
            label += " " + cat[1]
            # alpha = 1
            hatch = "//"
        else:
            label += " " + cat[0]
            hatch = ""
            # alpha = 0.6

        plt.bar(
            ind + (score_idx * width * 1.5),
            scores,
            color=colors[int(score_idx % (len(label_list) / 2))],
            width=width,
            align="center",
            label=label,
            hatch=hatch,
        )

    plt.xticks(ind + width * 2.25, ["negative", "neutral", "positive"])
    if title is not None:
        plt.title(title)
    plt.legend(
        bbox_to_anchor=(0.05, 0.95),
        loc="upper left",
        borderaxespad=0.0,
        ncol=1,
        fancybox=True,
    )
    if dest is not None:
        plt.savefig(os.path.join(dest, "Ratios.png"))
    else:
        plt.show()


def read_file_into_list(file_name):
    if len(file_name) > 1:
        with open(file_name, "r") as f:
            data_list = f.readlines()
            data_list = [x.strip() for x in data_list]
        return data_list
    return None


def get_distribution(params, labeled_csv_file):
    # files1 = [params.labeled_csv_file]

    # Process name files if they exist.
    if params.neg_name_file and params.pos_name_file:
        names1 = read_file_into_list(params.neg_name_file)
        names2 = read_file_into_list(params.pos_name_file)
    # num_file_splits = 1

    # for fi_idx in range(num_file_splits):
    lines_to_sents = score_analysis(labeled_csv_file)
    if params.neg_name_file and params.pos_name_file:
        sents = OrderedDict({constants.NAMES1: [], constants.NAMES2: []})
    else:
        sents = OrderedDict(
            {
                # constants.BLACK: [],
                # constants.WHITE: [],
                constants.MALE_SINGLE: [],
                constants.FEMALE_SINGLE: [],
                # constants.STRAIGHT: [],
                # constants.GAY: [],
            }
        )
    for l, val in lines_to_sents:
        if params.trigger_position == constants.HEAD:
            # if l.startswith("The Black person"):
            #   sents[constants.BLACK].append(val)
            # elif l.startswith("The White person"):
            #    sents[constants.WHITE].append(val)
            if l.startswith(constants.MALE_SINGLE):
                sents[constants.MALE_SINGLE].append(val)
            elif l.startswith(constants.FEMALE_SINGLE):
                sents[constants.FEMALE_SINGLE].append(val)
            # elif l.startswith("The straight person"):
            #    sents[constants.STRAIGHT].append(val)
            # elif l.startswith("The gay person"):
            #    sents[constants.GAY].append(val)
            else:
                found_name = False
                name = l.split()[2]
                for n1 in names1:
                    if n1 == name:
                        sents[constants.NAMES1].append(val)
                        found_name = True
                        break
                if not found_name:
                    for n2 in names2:
                        if n2 == name:
                            sents[constants.NAMES2].append(val)
                            found_name = True
                            break
                if not found_name:
                    raise NotImplementedError("unidentified demographic", l)
        elif params.trigger_position == constants.BODY:
            if l.startswith(constants.MALE_SINGLE):
                sents[constants.MAN].append(val)
            elif l.startswith(constants.FEMALE_SINGLE):
                sents[constants.WOMAN].append(val)
            else:
                found_name = False
                for n1 in names1:
                    if l.startswith(n1):
                        sents[constants.NAMES1].append(val)
                        found_name = True
                        break
                if not found_name:
                    for n2 in names2:
                        if l.startswith(n2):
                            sents[constants.NAMES2].append(val)
                            found_name = True
                            break
                if not found_name:
                    raise NotImplementedError("unidentified demographic", l)
    for s in sents:
        print("%s: %s samples, avg score = %s" % (s, len(sents[s]), np.mean(sents[s])))
    print("=" * 80)
    sents_list = list(sents.values())
    if params.neg_name_file and params.pos_name_file:
        ordered_ratios = plot_score(
            sents_list, [constants.NAMES1, constants.NAMES2], ratio=True
        )
    else:
        ordered_ratios = plot_score(
            sents_list,
            [
                # constants.BLACK,
                # constants.WHITE,
                constants.MALE_SINGLE,
                constants.FEMALE_SINGLE,
                # constants.STRAIGHT,
                # constants.GAY,
            ],
            ratio=True,
        )
    return ordered_ratios
