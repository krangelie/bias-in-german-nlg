import time

import numpy as np
import torch


def set_device_and_seeds():
    np.random.seed(0)
    torch.random.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    print("Device: ", device)
    return device


def adjust_params(params):
    params.salience_threshold = float(params.salience_threshold)
    params.use_original_loss = int(params.use_original_loss) == 1
    params.use_salience_loss = int(params.use_salience_loss) == 1
    params.use_dissociation_loss = int(params.use_dissociation_loss) == 1
    params.use_weighted_salience_loss = int(params.use_weighted_salience_loss) == 1
    params.alpha = float(params.alpha)
    params.beta = float(params.beta)
    params.beam_size = int(params.beam_size)
    params.use_weighted_neg = int(params.use_weighted_neg) == 1
    params.num_trigger_tokens = int(params.num_trigger_tokens)
    if params.trigger_masked_phrases:
        params.trigger_masked_phrases = params.trigger_masked_phrases.split(",")
    else:
        params.trigger_masked_phrases = []
    params.debias = int(params.debias)
    assert params.debias in [0, 1, 2]
    # 0 = no debias, 1 = associate neutral, dissociate everything else, 2 = associate positive + neutral, dissociate neg
    params.num_demographics = int(params.num_demographics)
    params.batch_size = int(params.batch_size)
    print("Params", params)
    return params


_start_time = time.time()


def tic():
    global _start_time
    _start_time = time.time()


def tac():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    print("Time passed: {}hour:{}min:{}sec".format(t_hour, t_min, t_sec))
