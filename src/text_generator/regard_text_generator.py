import os
import string

import hydra.utils
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GPTNeoForCausalLM,
    GPT2Tokenizer,
)

from src.text_generator.prompt_generator import generate_prompt_list
from src.bias_mitigator.sample_from_gpt2 import filter_first_sentence, is_non_sentence


def sample_for_prompt(sample_params, tokenizer, model, prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    sample_outputs = model.generate(
        input_ids,
        do_sample=sample_params.do_sample,
        max_length=sample_params.max_length,
        top_p=sample_params.top_p,
        top_k=sample_params.top_k,
        temperature=sample_params.temperature,
        num_return_sequences=sample_params.num_return_sequences,
    )
    return sample_outputs


def save_outputs(output_dir, sample_outputs, tokenizer, demo, prompt, trigger):
    os.makedirs(output_dir, exist_ok=True)
    outfile = os.path.join(output_dir, f"{demo}_texts.txt")
    print(f"Storing at {output_dir}")
    with open(outfile, "a") as f:
        for i, sample_output in enumerate(sample_outputs):
            sentence = tokenizer.decode(sample_output)
            if trigger:
                sentence = sentence.replace(trigger, "")
            sentence = filter_first_sentence(sentence)
            if not is_non_sentence(sentence, prompt):
                print(f"{i}: " f"{sentence} ")
                f.write(sentence + "\n")


def sample_for_list_of_prompts(
    sample_params, prompt_list_file, demo, tokenizer, model, output_dir, trigger
):
    with open(prompt_list_file, "r") as f:
        prompts = [line.strip("\n") for line in f.readlines()]
    print(f"Read {len(prompts)} prompts from text file.")

    for prompt in prompts:
        sample_outputs = sample_for_prompt(sample_params, tokenizer, model, prompt)
        save_outputs(output_dir, sample_outputs, tokenizer, demo, prompt, trigger)


def generate_gpt2_texts(cfg):

    print(f"Initializing pretrained model {cfg.gpt.path} - type {cfg.gpt.name}")
    if cfg.gpt.name == "gpt2":
        tokenizer = AutoTokenizer.from_pretrained(cfg.gpt.path)
        model = AutoModelForCausalLM.from_pretrained(cfg.gpt.path)
    else:
        model = GPTNeoForCausalLM.from_pretrained(cfg.gpt.path)
        tokenizer = GPT2Tokenizer.from_pretrained(cfg.gpt.path)

    for demo in cfg.run_mode.demographics:
        if cfg.run_mode.trigger:
            name = (
                f"{demo}_prompts_"
                f"{cfg.run_mode.trigger.translate(str.maketrans('', '', string.punctuation))}.txt"
            )
        else:
            name = f"{demo}_prompts.txt"

        prompt_dir = hydra.utils.to_absolute_path(cfg.run_mode.prompt_dir)
        output_dir = hydra.utils.to_absolute_path(cfg.run_mode.output_dir)

        file_name = os.path.join(prompt_dir, name)
        if not os.path.isfile(file_name):
            generate_prompt_list(prompt_dir, demo, cfg.run_mode.trigger, file_name)
        print(f"Sampling for {demo}")
        sample_for_list_of_prompts(
            cfg.gpt,
            file_name,
            demo,
            tokenizer,
            model,
            output_dir,
            cfg.run_mode.trigger,
        )
    torch.cuda.empty_cache()
    del model
    del tokenizer
