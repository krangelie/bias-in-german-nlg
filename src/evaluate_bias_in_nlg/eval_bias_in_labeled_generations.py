import os
import json

import hydra.utils
import pandas as pd
from scipy.stats import chi2, chi2_contingency
import matplotlib.pyplot as plt

import src.constants as constants
from src.evaluate_bias_in_nlg.bias_eval_helpers import (
    single_file_to_dict,
    mult_files_to_dict,
    plot_regard_ratios,
)


def read_regard_labeled_demo_csv(in_path, demographics, contexts):
    if contexts != "all":
        if contexts == "respect":
            context_list = constants.RESPECT_LIST
        elif contexts == "occupation":
            context_list = constants.OCCUPATION_LIST
        else:
            print("Typo or undefined context list.")
    else:
        context_list = None

    print(f"-- PROCESSING {contexts} CONTEXTS --")

    if in_path.endswith(".csv"):
        demo_dict = single_file_to_dict(in_path, demographics, context_list)
    else:
        demo_dict = mult_files_to_dict(in_path, demographics, context_list)

    print(demographics)

    return demo_dict


def create_contingency_table(demo_dict):
    demographics = [constants.VARIABLE_DICT[k] for k in demo_dict.keys()]
    contingency_table = pd.DataFrame(
        [],
        columns=constants.VALENCE_MAP,
        index=demographics,
    )
    ratios = {}
    sample_num = min([len(value) for key, value in demo_dict.items()])
    for demo, preds in demo_dict.items():
        ratios[demo] = {}
        preds = preds.reset_index()
        counts = preds.loc[:sample_num, "Prediction"].value_counts()
        counts_all = preds["Prediction"].value_counts()
        print("Counts", counts)
        if len(counts != len(constants.VALENCE_MAP)):
            # set_of_preds = preds.loc[:sample_num, "Prediction"]
            set_of_preds = preds["Prediction"]
            print(
                "Attention, not all classes have predictions:",
                set(set_of_preds),
            )
        for valence, value in constants.VALENCE_MAP.items():
            if value in counts.index:
                num = counts_all[value]
            else:
                num = 0
            contingency_table.loc[constants.VARIABLE_DICT[demo], valence] = num
            ratios[demo][valence] = counts_all[value] / len(preds)

    print(contingency_table)
    return contingency_table, ratios


def test_group_independence(contingency_table, out_path, contexts, ratios):

    stat, p, dof, expected = chi2_contingency(contingency_table)
    print("dof=%d" % dof)
    print(expected)
    # interpret test-statistic
    prob = 0.95
    critical = chi2.ppf(prob, dof)

    alpha = 1.0 - prob
    ctxt = f"\nResults for {contexts}\n\n"
    prob_txt = f"\nprobability={prob:.3f}, critical={critical:.3f}, stat={stat:.3f}"
    sign_txt = f"\nalpha={alpha:.3f}, p={p:.3f}"
    print(sign_txt)
    if p <= alpha:
        h_txt = "\nThere is a difference between the distributions (reject H0)"
    else:
        h_txt = "\nThere is no difference between the distributions (fail to reject H0)"
    print(h_txt)

    results_file = open(os.path.join(out_path, "chisquare.txt"), "a")
    result_txt = [
        ctxt,
        prob_txt,
        sign_txt,
        h_txt,
        f"\nN = {contingency_table.sum().sum()}",
    ]
    results_file.writelines(result_txt)
    results_file.close()


def eval_bias_for_context(eval_cfg, axis, context, input_path, output_path):
    demo_dict = read_regard_labeled_demo_csv(
        input_path,
        eval_cfg.demographics,
        context,
    )
    contingency_table, ratios = create_contingency_table(demo_dict)
    test_group_independence(contingency_table, output_path, context, ratios)

    ratios_df = pd.DataFrame([], columns=["Demographic", "Regard", "Prediction"])
    counter = 0
    for demo, valence_dict in ratios.items():
        for valence, value in valence_dict.items():
            ratios_df.loc[counter, :] = {
                "Demographic": demo,
                "Regard": valence,
                "Prediction": value * 100,
            }
            counter += 1

    plot_regard_ratios(demo_dict, context, axis, ratios_df)


def eval_regard_bias(cfg):
    eval_cfg = cfg.run_mode
    output_path = hydra.utils.to_absolute_path(eval_cfg.output_path)
    os.makedirs(output_path, exist_ok=True)
    input_path = hydra.utils.to_absolute_path(eval_cfg.input_path)

    if cfg.run_mode.contexts == "combine":
        fig, ax = plt.subplots(1, 3)
        fig.set_size_inches(7.5, 4)
        fig.suptitle(
            "Regard scores [%]",
            # "Weibchen Sternzeichen Freundlichkeitsprofil Erlangen Mineral",
            # "Vitamin Kneipp Neuzeit empfehlen Klassik erholsame",
            fontsize=15,
        )

        for i, c in enumerate(["all", "occupation", "respect"]):
            eval_bias_for_context(eval_cfg, ax[i], c, input_path, output_path)
        plt.xlabel("")
        # plt.xticks(fontsize=14)
        # plt.ylabel("Regard score [%]", fontsize=15)
        plt.tight_layout()
        os.makedirs(output_path, exist_ok=True)
        dest = os.path.join(output_path, f"ratios_{cfg.run_mode.contexts}_contexts.png")
        fig.savefig(dest)

    else:
        output_path = os.path.join(output_path, f"{cfg.run_mode.contexts}_contexts")
        eval_bias_for_context(
            eval_cfg, None, cfg.run_mode.contexts, input_path, output_path
        )
        os.makedirs(output_path, exist_ok=True)
        plt.xlabel("")
        # plt.xticks(fontsize=14)
        plt.ylabel("Regard score [%]", fontsize=15)
        plt.tight_layout()
        os.makedirs(output_path, exist_ok=True)
        dest = os.path.join(output_path, f"ratios_{cfg.run_mode.contexts}_contexts.png")
        plt.savefig(dest)
