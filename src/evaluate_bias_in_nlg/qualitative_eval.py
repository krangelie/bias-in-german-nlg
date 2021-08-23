import os
import json

from pprint import pprint

import hydra.utils

import src.constants as constants
import src.bias_lexica as bias_lexica

from src.evaluate_bias_in_nlg.bias_eval_helpers import (
    has_context,
    single_file_to_dict,
    mult_files_to_dict,
)

bias_lexica = [
    bias_lexica.sexualization,
    bias_lexica.violence,
    bias_lexica.caretaker,
]
bias_names = ["sexualization", "perpetrator", "caretaker"]


def add_context_labels(demo_dict, text_col):
    for demo, demo_df in demo_dict.items():
        demo_df["OCC context"] = demo_df[text_col].apply(
            has_context, context_list=constants.OCCUPATION_LIST
        )
        demo_dict[demo] = demo_df
    return demo_dict


def add_bias_labels(demo_dict, text_col, out_path):
    for demo, demo_df in demo_dict.items():
        for i, bias_lex in enumerate(bias_lexica):
            demo_df[bias_names[i]] = demo_df[text_col].apply(
                has_context, context_list=bias_lex
            )
        demo_dict[demo] = demo_df
        demo_df.to_csv(
            os.path.join(out_path, f"{demo}_texts_regard_qualbias_labeled.csv")
        )
    return demo_dict


def compute_ratios(demo_dict, text_col):
    ratio_report = {}
    for demo, demo_df in demo_dict.items():
        tmp_report = {}
        for i, bias_lex in enumerate(bias_lexica):
            bias_dict = {}
            abs_count = demo_df[bias_names[i]].sum()
            bias_df = demo_df.loc[demo_df[bias_names[i]] == 1, :]
            bias_dict["total ratio"] = abs_count / len(demo_df)
            bias_dict["total abs"] = f"{abs_count} from {len(demo_df)}"

            occupation_texts = bias_df.loc[bias_df["OCC context"] == 1, text_col]
            all_occ = len(demo_df[demo_df["OCC context"] == 1])
            if len(occupation_texts) > 0:
                bias_dict["occ ratio"] = len(occupation_texts) / all_occ
                bias_dict["occ abs"] = f"{len(occupation_texts)} from {all_occ}"
                print(f"{demo} - {bias_names[i]} - occupation")
                print(occupation_texts.tolist())

            respect_texts = bias_df.loc[bias_df["OCC context"] == 0, text_col]
            all_res = len(demo_df[demo_df["OCC context"] == 0])
            if len(respect_texts) > 0:
                bias_dict["res ratio"] = len(respect_texts) / all_res
                bias_dict["res abs"] = f"{len(respect_texts)} from {all_res}"
                print(f"{demo} - {bias_names[i]} - respect")
                print(respect_texts.tolist())

            neg_texts = bias_df.loc[bias_df["Prediction"] == 0, text_col]
            all_neg = len(demo_df[demo_df["Prediction"] == 0])
            if len(neg_texts) > 0:
                bias_dict["neg ratio"] = len(neg_texts) / all_neg
                bias_dict["neg abs"] = f"{len(neg_texts)} from {all_neg}"
                print(f"{demo} - {bias_names[i]} - neg regard")
                print(neg_texts.tolist())

            neu_texts = bias_df.loc[bias_df["Prediction"] == 1, text_col]
            all_neu = len(demo_df[demo_df["Prediction"] == 1])
            if len(neu_texts) > 0:
                bias_dict["neu ratio"] = len(neu_texts) / all_neu
                bias_dict["neu abs"] = f"{len(neu_texts)} from {all_neu}"

                print(f"{demo} - {bias_names[i]} - neutral regard")
                print(neu_texts.tolist())

            pos_texts = bias_df.loc[bias_df["Prediction"] == 2, text_col]
            all_pos = len(demo_df[demo_df["Prediction"] == 2])
            if len(pos_texts) > 0:
                bias_dict["pos ratio"] = len(pos_texts) / all_pos
                bias_dict["pos abs"] = f"{len(pos_texts)} from {all_pos}"

                print(f"{demo} - {bias_names[i]} - pos regard")
                print(pos_texts.tolist())

            tmp_report[bias_names[i]] = bias_dict
        ratio_report[demo] = tmp_report
    return ratio_report


def eval_qual_bias(cfg):
    eval_cfg = cfg.run_mode
    output_path = hydra.utils.to_absolute_path(eval_cfg.output_path)
    if os.path.isfile(output_path):
        output_path = os.path.dirname(output_path)
    os.makedirs(output_path, exist_ok=True)

    input_path = hydra.utils.to_absolute_path(eval_cfg.input_path)

    demographics = eval_cfg.demographics
    if input_path.endswith(".csv"):
        demo_dict = single_file_to_dict(input_path, demographics)
    else:
        demo_dict = mult_files_to_dict(input_path, demographics)
    if eval_cfg.add_bias_labels:
        demo_dict = add_context_labels(demo_dict, cfg.text_col)
        demo_dict = add_bias_labels(demo_dict, cfg.text_col, output_path)
        print(
            "Bias labels have been added to the dataframe. You might want to manually review "
            "before counting."
        )
        print("File location", output_path)

    else:
        demo_dict = mult_files_to_dict(input_path, eval_cfg.demographics)
    if eval_cfg.compute_ratios:
        if "caretaker" in demo_dict[eval_cfg.demographics[0]].columns:
            ratio_report = compute_ratios(demo_dict, cfg.text_col)
            pprint(ratio_report)
            with open(
                os.path.join(output_path, f"qualitative_bias_ratios.json"), "w"
            ) as outfile:
                json.dump(ratio_report, outfile)
        else:
            "ABORTING - Please generate bias labels first."
