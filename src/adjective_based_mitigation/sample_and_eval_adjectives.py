import os
import random

import hydra.utils
import torch
from torch.nn import functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.classifier.torch_helpers.load_pretrained import load_torch_model
from src.text_generator.regard_text_generator import (
    filter_first_sentence,
    is_non_sentence,
)
from src.text_generator.regard_text_generator import sample_for_prompt
from src.regard_prediction.inference import embed_texts
import src.constants as constants

"""Experimentation with naive trigger approach: Tries out adjectives from a positive adjective 
list and takes the one that reduces negative regard most."""


def sample_adjectives(file_path, n):
    with open(file_path, "r") as f:
        adjectives = [line.rstrip() for line in f]
    random.seed(42)
    adjectives = random.sample(adjectives, n)
    return adjectives


def sample_sentences_for_adj(cfg, adjective, tokenizer, model):
    trigger = f"{cfg.run_mode.demographic} war {adjective}. "
    samples = []
    contexts = cfg.run_mode.contexts
    if contexts == "all":
        context_list = constants.CONTEXT_LIST
    elif contexts == "respect":
        context_list = constants.RESPECT_LIST
    elif contexts == "occupation":
        context_list = constants.OCCUPATION_LIST
    else:
        print("Typo or undefined context list.")
    for context in context_list:
        prompt = trigger + cfg.run_mode.demographic + " " + context
        print(prompt)
        samples_encoded = sample_for_prompt(cfg.gpt, tokenizer, model, prompt)

        for sample_output in samples_encoded:
            sentence = tokenizer.decode(sample_output)
            sentence = sentence.replace(trigger, "")
            sentence = filter_first_sentence(sentence)
            if not is_non_sentence(sentence, prompt):
                samples.append(sentence)
    return samples


def sample_and_classify(cfg, adjective, gpt_tokenizer, gpt_model):
    model_type = cfg.classifier.name
    model_path = hydra.utils.to_absolute_path(cfg.run_mode.regard_classifier)
    model = load_torch_model(model_path, model_type, logger=None)
    model.eval()
    samples = sample_sentences_for_adj(cfg, adjective, gpt_tokenizer, gpt_model)
    print(samples[:2])
    sentence_df, sentences_emb = embed_texts(
        cfg,
        cfg.classifier_mode.embedding,
        model_type,
        pd.DataFrame(samples, columns=[cfg.text_col]),
    )
    outputs = model(torch.Tensor(sentences_emb))
    probs = F.log_softmax(outputs, dim=1)
    preds = torch.argmax(probs, dim=1).detach().numpy()
    return preds


def get_target_ratio(predictions, target_valence):
    # target valence can be "negative", "neutral", or "positive"
    target_count = sum(predictions == constants.VALENCE_MAP[target_valence])
    print("Predictions", set(predictions), f"Num {target_valence}", target_count)
    ratio = target_count / len(predictions)
    return ratio


def get_best_adjectives(adj_ratio_list, criterion, top_k):
    reverse = False if criterion == "min" else True
    adj_ratio_list = sorted(adj_ratio_list, key=(lambda x: x[1]), reverse=reverse)
    return adj_ratio_list[:top_k]


def find_best_adjective(cfg):
    gpt_tokenizer = AutoTokenizer.from_pretrained(cfg.gpt.path)
    gpt_model = AutoModelForCausalLM.from_pretrained(cfg.gpt.path)
    target_valence = cfg.run_mode.target_valence
    criterion = cfg.run_mode.criterion
    top_k = cfg.run_mode.top_k
    adjective_list = hydra.utils.to_absolute_path(cfg.run_mode.adjective_list)
    adjectives = sample_adjectives(adjective_list, cfg.run_mode.adj_samples)
    adj_ratio_tuples = []
    for i, adjective in enumerate(adjectives):
        print(f"-- Evaluating adjective no. {i}: {adjective} --")
        preds = sample_and_classify(cfg, adjective, gpt_tokenizer, gpt_model)
        ratio = get_target_ratio(preds, target_valence)
        adj_ratio_tuples += [(adjective, ratio)]
        print(f"Ratio no. {i} for {adjective}: {ratio}")

    best_adjectives = get_best_adjectives(adj_ratio_tuples, criterion, top_k)
    print(
        f"Top {top_k} adjectives (criterion={criterion}) for {target_valence} regard:"
    )
    print(best_adjectives)
    out_path = hydra.utils.to_absolute_path(cfg.run_mode.out_path)
    os.makedirs(out_path, exist_ok=True)
    pd.DataFrame(
        best_adjectives, columns=["Adjective", f"Ratio_{target_valence}"]
    ).to_csv(
        os.path.join(
            out_path,
            f"Top_{top_k}_adjectives_{criterion}_{target_valence}_"
            f"{cfg.run_mode.demographic}.csv",
        )
    )
