import os

import hydra.utils
from joblib import dump, load


def save_model(output_path, classifier, logger=None):
    output_path = hydra.utils.to_absolute_path(output_path)
    os.makedirs(output_path, exist_ok=True)

    if logger is not None:
        logger.info(f"Storing model at {output_path}.")
    dump(classifier, os.path.join(output_path, "model.joblib"))


def load_model(model_path, logger=None):
    model_path = hydra.utils.to_absolute_path(model_path)
    if logger is not None:
        logger.info("Loading model from {dest}.")
    classifier = load(model_path)
    return classifier
