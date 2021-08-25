import logging


import hydra
from omegaconf import DictConfig

from src import create_dataset, run_classifier
from src.text_generator.run_text_generation import run_txt_generation
from src.bias_mitigator import run_bias_mitigation
from src.evaluate_bias_in_nlg.run_bias_eval import run_bias_evaluation
from src.adjective_based_mitigation.sample_and_eval_adjectives import (
    find_best_adjective,
)

rootLogger = logging.getLogger()
consoleHandler = logging.StreamHandler()
rootLogger.addHandler(consoleHandler)
# rootLogger = None


@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig):
    mode = cfg.run_mode.name
    print("Run mode", mode)
    if mode == "data":
        create_dataset.main(cfg)
    elif mode == "classifier":
        run_classifier.run(cfg, rootLogger)
    elif mode == "generate":
        run_txt_generation(cfg)
    elif mode == "trigger":
        run_bias_mitigation.run(cfg)
    elif mode == "eval_bias":
        run_bias_evaluation(cfg)
    elif mode == "naive_trigger":
        find_best_adjective(cfg)
    else:
        print("Run mode not implemented. Typo?")


if __name__ == "__main__":
    run()
