"""Sample from LM using trigger + prompts."""

import os

import hydra.utils
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelWithLMHead,
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Tokenizer,
    GPT2LMHeadModel,
)
import re
import numpy as np
import pandas as pd

import src.constants as constants


# Gets the score for the top-k logits to improve quality of samples.
def top_k_logits(logits, k):
    if k == 0:
        return logits
    values = torch.topk(logits, k)[0]
    batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
    return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)


# Generates from the model using optional top-k sampling
def sample_sequence(
    model,
    length,
    batch_size=1,
    context=None,
    device="cuda",
):
    context = (
        torch.tensor(context, device=device, dtype=torch.long)
        .unsqueeze(0)
        .repeat(batch_size, 1)
    )

    with torch.no_grad():
        output = model.generate(
            context,
            do_sample=True,
            max_length=length,
            top_p=0.92,
            top_k=20,
            temperature=0.7,
            num_return_sequences=1,
        )
    return output


def convert_to_bert_format(
    bias_context_to_generations, bert_file, generation_only=False
):
    """Convert to BERT regard classifier format.
    with open(bert_file, "w") as f:
    f.write("\n".join(samples) + "\n")"""

    samples = []
    for bias_context, gen in bias_context_to_generations.items():
        for g in gen:
            if not generation_only:
                samples.append(bias_context + " " + g)
    pd.DataFrame(samples, columns=["Text"]).to_csv(bert_file)
    return samples


def filter_first_sentence(text):
    """Heuristic to only keep the first `sentence` in text."""
    # Cut off the line when we see the first period.
    text = text.replace("\n", " ").replace(" \t", " ")
    if "! " in text:
        period_idx = text.index("! ")
    elif "? " in text:
        period_idx = text.index("? ")
    elif ". " in text:
        period_idx = filter_abbreviations(text)
    else:
        period_idx = len(text)
    sample_end = min(period_idx + 1, len(text))
    text = text[:sample_end]
    return text


def filter_abbreviations(text, prev_period_idx=None, this_period_idx=None):
    if prev_period_idx is None:
        period_idx = text.index(". ")
        sample_end = min(period_idx + 1, len(text))
        span = text[:sample_end]
    else:
        period_idx = this_period_idx
        sample_end = min(period_idx + 1, len(text))
        span = text[prev_period_idx:sample_end]

    if "ca." in span or "z.B." in span:
        if ". " in text[sample_end + 1 :]:
            prev_period_idx = period_idx
            period_idx = text[sample_end + 1 :].index(". ") + sample_end + 1
            period_idx = filter_abbreviations(text, prev_period_idx, period_idx)

    return period_idx


def is_non_sentence(text, prompt):
    # Filter generations that consist only of one punctuation or short word after prompt
    remainder = text.split(prompt)
    if len(remainder) > 1:
        remainder = [remainder[1]]
    if len(remainder[0]) < 5:
        return True
    elif remainder[0].endswith("ca."):
        return True
    else:
        return False


# https://github.com/django/django/blob/master/django/utils/text.py
def get_valid_filename(s):
    """
    Return the given string converted to a string that can be used for a clean
    filename. Remove leading and trailing spaces; convert other spaces to
    underscores; and remove anything that is not an alphanumeric, dash,
    underscore, or dot.
    'johns_portrait_in_2004.jpg'
    """
    s = str(s).strip().replace(" ", "_")
    return re.sub(r"(?u)[^-\w.]", "", s)


def read_file_into_list(file_name):
    if not file_name:
        return []
    with open(file_name, "r") as f:
        data_list = f.readlines()
        data_list = [x.strip() for x in data_list]
        return data_list


def sample(params, trigger_list=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(params.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(params.model_name_or_path)

    model.eval()
    model.to(device)

    # get samples from the model

    if len(params.neg_name_file) > 1 and len(params.pos_name_file) > 1:
        bias_contexts1 = read_file_into_list(params.neg_name_file)
        bias_contexts2 = read_file_into_list(params.pos_name_file)
        generations = 10
    else:
        bias_contexts1 = [params.neg_demographic]
        bias_contexts2 = [params.pos_demographic]
        generations = 100
    respect_contexts = constants.GPT2_RCONTEXTS
    occupation_contexts = constants.GPT2_OCONTEXTS

    if trigger_list is not None:
        if isinstance(trigger_list, list):
            orig_trigger_tokens = np.array(trigger_list)
        else:
            orig_trigger_tokens = np.array(
                [int(t) for t in trigger_list.split(" ") if len(t) > 0]
            )
        trigger = tok.decode(orig_trigger_tokens)
    else:
        trigger = None

    bias_context_to_generations = {}
    if params.trigger_position == constants.HEAD or trigger_list is None:
        contexts = [
            respect_contexts,
            occupation_contexts,
        ]  # lists of phrases like: 'was known for', 'was described as', 'was regarded as', 'was thought of as'
        for context_list in contexts:
            for context in context_list:
                if type(context) == tuple and len(context) == 2:
                    for bias_context1 in bias_contexts1:
                        bias_context_to_generations[
                            " ".join(
                                [context[0], bias_context1, context[1] + tok.eos_token]
                            )
                        ] = []
                    for bias_context2 in bias_contexts2:
                        bias_context_to_generations[
                            " ".join(
                                [context[0], bias_context2, context[1] + tok.eos_token]
                            )
                        ] = []
                else:
                    for bias_context1 in bias_contexts1:  # e.g. white female names
                        bias_context_to_generations[
                            bias_context1 + " " + context
                        ] = []  # would be sth like "Molly was known for"
                    for bias_context2 in bias_contexts2:
                        bias_context_to_generations[bias_context2 + " " + context] = []
    elif params.trigger_position == constants.BODY:
        for bias_context1 in bias_contexts1:
            bias_context_to_generations[bias_context1 + " " + trigger] = []
        for bias_context2 in bias_contexts2:
            bias_context_to_generations[bias_context2 + " " + trigger] = []
    else:
        raise NotImplementedError(
            "params.trigger_position is either head or body", params.trigger_position
        )

    # prepend or append tokens
    for bias_context in bias_context_to_generations:
        # if trigger_list is not None:
        if params.trigger_position == constants.HEAD:
            if trigger:
                trigger_str = trigger + bias_context
                trigger_tokens = np.concatenate(
                    (orig_trigger_tokens, tok.encode(bias_context)), axis=0
                )
            else:
                trigger_str = bias_context
                trigger_tokens = tok.encode(bias_context)
        elif params.trigger_position == constants.BODY:
            trigger_str = bias_context
            trigger_tokens = tok.encode(trigger_str)
        for _ in range(generations):
            out = sample_sequence(
                model=model,
                length=40,
                context=trigger_tokens,
                device=device,
            )
            out = out[:, len(trigger_tokens) :].tolist()
            for i in range(1):
                text = tok.decode(out[i])
                if text.find(tok.eos_token) > 0:
                    text = text[0 : text.find(tok.eos_token)]
                text = text.strip()

                first_sentence = filter_first_sentence(text)
                # if not is_non_sentence(first_sentence, trigger_str):

                if not is_non_sentence(first_sentence, trigger_str):
                    if trigger_list is not None:
                        print("Prompt: ", trigger_str)
                    print("Output: ", text)
                    print("First sentence: ", first_sentence)
                    print("=" * 80)
                    bias_context_to_generations[bias_context].append(first_sentence)

    if trigger_list is not None:
        fname_list = [get_valid_filename(x) for x in trigger.split()]
        bert_file = "_".join(fname_list) + ".csv"

    else:
        no_trigger_file_name = hydra.utils.to_absolute_path(params.no_trigger_file_name)
        bert_file = no_trigger_file_name + ".csv"
    bert_file = bert_file.replace("endoftext", "")
    trigger_label_output_dir = hydra.utils.to_absolute_path(
        params.trigger_label_output_dir
    )
    bert_file = os.path.join(trigger_label_output_dir, bert_file)
    samples = convert_to_bert_format(bias_context_to_generations, bert_file)

    bert_file += ".XYZ"
    # For GPT2, save a version of the bert file with XYZ demographics for the regard classifier.
    all_contexts = respect_contexts + occupation_contexts
    new_bias_context_to_generations = {}
    for bias_context, generations in bias_context_to_generations.items():
        if params.trigger_position == constants.HEAD or trigger_list is not None:
            found = False
            for context in all_contexts:
                try:
                    context_idx = bias_context.index(context)
                    bias_context = bias_context[context_idx:]
                    found = True
                    break
                except ValueError:
                    pass
            if not found:
                print("bias_context:", bias_context)
            assert found
            bias_context = "XYZ " + bias_context
        elif params.trigger_position == constants.BODY:
            bias_context = "XYZ " + trigger
        if bias_context not in new_bias_context_to_generations:
            new_bias_context_to_generations[bias_context] = []
        new_bias_context_to_generations[bias_context].extend(generations)
    convert_to_bert_format(new_bias_context_to_generations, bert_file)

    del model
    del tok
    return samples
