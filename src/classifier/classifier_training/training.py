import numpy as np

from src.classifier.torch_helpers.torch_training import train_torch_model
from src.classifier.non_torch.non_torch_training import train_sklearn


def train_classifier(
    cfg, X_dev_emb, Y_dev, X_test_emb, Y_test, texts_test, logger, seed=42
):
    classes = set(Y_dev)

    if not cfg.classifier.name.startswith(("lstm", "transformer")):
        score = train_sklearn(
            cfg, X_dev_emb, X_test_emb, Y_dev, Y_test, logger, texts_test
        )

    else:
        score = train_torch_model(
            cfg, X_dev_emb, X_test_emb, Y_dev, Y_test, classes, texts_test, seed
        )

    return score
