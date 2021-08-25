"""Script to evaluate generated triggers (with regard classifier)."""

import os, sys
import re

import hydra.utils
from transformers import AutoTokenizer
from omegaconf import DictConfig

from src.bias_mitigator.sample_from_gpt2 import sample
from src.bias_mitigator.get_model_distrib import get_distribution, plot_ratios
from src.regard_prediction.inference import predict
import src.constants as constants


def get_trigger_list(trigger_dump_file, triggers=None, trigger_ids=None):
    """Parse trigger_dump_file for final triggers."""
    if triggers is None:
        final_trigger_id = "Final trigger token IDs: "
        final_trigger = "Final trigger: "
        trigger_ids = []
        triggers = []
        if trigger_dump_file:
            with open(trigger_dump_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if final_trigger_id in line:
                        trigger_idx = line.index(final_trigger_id)
                        trigger_val = line[trigger_idx + len(final_trigger_id) :]
                        trigger_val = trigger_val.strip()
                        trigger_val = trigger_val.replace("[", "")
                        trigger_val = trigger_val.replace("]", "")
                        trigger_val_split = trigger_val.split(".")
                        trigger_val = ",".join([v.strip() for v in trigger_val_split])
                        trigger_ids.append(trigger_val)
                    if final_trigger in line:
                        trigger_idx = line.index(final_trigger)
                        trigger_val = line[trigger_idx + len(final_trigger) :]
                        trigger_val = trigger_val.rstrip()
                        triggers.append(trigger_val)
    # Remove eos.
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    triggers = [x.replace(tokenizer.eos_token, "") for x in triggers]
    trigger_ids = [x for x in trigger_ids if x != tokenizer.eos_token_id]
    return triggers, trigger_ids


def get_valid_filename(s):
    s = str(s).strip().replace(" ", "_")
    return re.sub(r"(?u)[^-\w.]", "", s)


def eval_with_tokens(params, trigger_params):
    labeled_csv_file = hydra.utils.to_absolute_path(trigger_params.labeled_csv_file)
    trigger_label_output_dir = hydra.utils.to_absolute_path(
        trigger_params.trigger_label_output_dir
    )
    regard_classifier_dir = hydra.utils.to_absolute_path(
        trigger_params.regard_classifier_dir
    )
    trigger_dump_file = hydra.utils.to_absolute_path(trigger_params.trigger_dump_file)

    trigger_list, trigger_id_list = get_trigger_list(trigger_dump_file)
    trigger_list = trigger_list[:5]
    txt_files = []
    ordered_ratios = []
    for trigger, comma_trigger_list in zip(trigger_list, trigger_id_list):
        print("Trigger", trigger)

        # Sample with trigger.
        fname_list = [get_valid_filename(x) for x in trigger.split()]
        sample_txt_file = "_".join(fname_list) + ".csv"
        sample_txt_file = sample_txt_file.replace("endoftext", "")
        print("txt_file", sample_txt_file)
        sample_location = os.path.join(trigger_label_output_dir, sample_txt_file)
        trigger_params.labeled_csv_file = sample_location.replace(
            ".csv", "_regard_labeled.csv"
        )
        if not os.path.exists(sample_location):
            sample(trigger_params, comma_trigger_list)

        txt_files.append(sample_location)
        # Use regard classifier to classify samples.
        if not os.path.exists(labeled_csv_file):
            predict(
                params,
                eval_model=regard_classifier_dir,
                eval_model_type="transformer",
                embedding_path=trigger_params.embedding,
                sample_file=sample_location,
                eval_dest=trigger_label_output_dir,
                logger=None,
            )
        # Calculate ratios of pos/neu/neg samples for evaluation.
        print("=" * 80)
        ordered_ratios.append(get_distribution(trigger_params, labeled_csv_file))
    ordered_ratios = ordered_ratios[0] if len(trigger_id_list) == 1 else ordered_ratios
    return ordered_ratios, len(trigger_list)


def eval_without_tokens(params, trigger_params, list_length):
    trigger_label_output_dir = hydra.utils.to_absolute_path(
        trigger_params.trigger_label_output_dir
    )
    regard_classifier_dir = hydra.utils.to_absolute_path(
        trigger_params.regard_classifier_dir
    )
    no_trigger_file_name = hydra.utils.to_absolute_path(
        trigger_params.no_trigger_file_name
    )

    txt_files = []
    ordered_ratios = []
    for i in range(list_length):
        # Sample with trigger.
        # fname_list = [get_valid_filename(x) for x in trigger.split()]
        # sample_txt_file = "_".join(fname_list) + ".txt"
        # sample_txt_file = sample_txt_file.replace("endoftext", "")
        # print("txt_file", sample_txt_file)
        # sample_location = trigger_params.trigger_label_output_dir + "/" + sample_txt_file
        sample_txt_file = os.path.join(
            trigger_label_output_dir,
            no_trigger_file_name + ".csv",
        )
        labeled_csv_file = sample_txt_file.replace(".csv", "_regard_labeled.csv")

        if not os.path.exists(sample_txt_file):
            sample(trigger_params)

        txt_files.append(sample_txt_file)

        # Use regard classifier to classify samples.
        if not os.path.exists(labeled_csv_file):
            print(trigger_label_output_dir)
            predict(
                params,
                eval_model=regard_classifier_dir,
                eval_model_type="transformer",
                embedding_path=trigger_params.embedding,
                sample_file=sample_txt_file,
                eval_dest=trigger_label_output_dir,
                logger=None,
            )

        # Calculate ratios of pos/neu/neg samples for evaluation.
        print("=" * 80)
        ordered_ratios.append(get_distribution(trigger_params, labeled_csv_file))

    ordered_ratios = ordered_ratios[0] if list_length == 1 else ordered_ratios
    return ordered_ratios


def evaluate_tokens(params: DictConfig):
    print("Redirecting stdout to 'outputs' folder.")
    orig_stdout = sys.stdout
    f = open("eval_stdout.txt", "a")
    sys.stdout = f
    trigger_params = params.run_mode
    trigger_dump_file = hydra.utils.to_absolute_path(trigger_params.trigger_dump_file)
    assert trigger_dump_file != ""
    print("Params", trigger_params)
    print(trigger_params.trigger_label_output_dir)
    print("Eval triggers")
    after_ratios, list_length = eval_with_tokens(params, trigger_params)
    print("Eval no-trigger baseline")
    before_ratios = eval_without_tokens(params, trigger_params, list_length)

    plot_ratios(
        before_ratios + after_ratios,
        [constants.MALE_SINGLE, constants.FEMALE_SINGLE] * 2,
        dest=hydra.utils.to_absolute_path(trigger_params.results_path),
    )

    sys.stdout = orig_stdout
    f.close()
