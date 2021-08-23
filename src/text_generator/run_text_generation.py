from src.text_generator.regard_text_generator import generate_gpt2_texts
from src.text_generator.regard_text_generator_gpt3 import generate_gpt3_texts


def run_txt_generation(cfg):
    if cfg.gpt.name == "gpt3":
        generate_gpt3_texts(cfg)
    else:
        generate_gpt2_texts(cfg)
