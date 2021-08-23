import sys

import torch.cuda

from src.bias_mitigator.create_adv_token import create_adversarial_tokens
from src.bias_mitigator.eval_triggers import evaluate_tokens


def run(params):
    if params.run_mode.find_triggers:
        create_adversarial_tokens(params)
    if params.run_mode.run_trigger_eval:
        evaluate_tokens(params)
