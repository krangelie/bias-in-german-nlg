import os

from src.constants import *


def generate_prompt_list(dest_path, demographics, trigger="", file_name=""):
    os.makedirs(dest_path, exist_ok=True)
    if isinstance(demographics, list):
        demos = [VARIABLE_DICT[demo] for demo in demographics]
    else:
        demos = [VARIABLE_DICT[demographics]]
        demographics = [demographics]

    for i, demo in enumerate(demos):
        print(demo)
        print(f"Writing prompts for {demographics[i]}")
        if file_name == "":
            file_name = os.path.join(dest_path, f"{demographics[i]}_prompts.txt")
        with open(file_name, "a") as f:
            for context in CONTEXT_LIST:
                prompt = demo + " " + context
                if trigger and trigger != "":
                    prompt = trigger + " " + prompt
                f.write(prompt + "\n")
