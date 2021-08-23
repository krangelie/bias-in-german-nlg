import hydra
import torch
from src.classifier.lstm.lstm_classifier import RegardLSTM
from src.classifier.sent_transformer.sbert_classifier import RegardBERT


def load_torch_model(model_path, model_type, logger=None):
    model_path = hydra.utils.to_absolute_path(model_path)
    if logger is not None:
        logger.info(f"Loading pretrained torch model from {model_path}")
    if model_path.endswith("pth"):
        model = torch.load(model_path)
    elif model_path.endswith("ckpt"):
        if model_type == "lstm":
            model = RegardLSTM.load_from_checkpoint(model_path)
        elif model_type == "transformer":
            model = RegardBERT.load_from_checkpoint(model_path)
    return model
