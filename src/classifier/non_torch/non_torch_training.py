import os
from datetime import datetime

import hydra.utils
import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.classifier.utils import build_experiment_name

from src.classifier.classifier_training.classifier_utils import get_classifier
from src.classifier.non_torch.eval_non_torch import evaluate_model
from src.classifier.non_torch.save_and_load_model import save_model


def train_sklearn(
    cfg, X_dev_emb, X_test_emb, Y_dev, Y_test, logger, texts_test, seed=None
):
    if cfg.dev_settings.annotation == "unanimous":
        hyperparameters = cfg.classifier.unanimous
    else:
        hyperparameters = cfg.classifier.majority
    model = get_classifier(hyperparameters, cfg.classifier.name, cfg.embedding.n_embed)
    if cfg.classifier_mode.cv_folds:
        skf = StratifiedKFold(n_splits=cfg.classifier_mode.cv_folds)
        scores = []
        for train_index, val_index in skf.split(X_dev_emb, Y_dev):
            X_train = X_dev_emb[train_index]
            Y_train = Y_dev.to_numpy()[train_index]

            model.fit(X_train, Y_train)
            scores.append(
                evaluate_model(
                    cfg.embedding.name,
                    cfg.classifier.name,
                    model,
                    X_test_emb,
                    Y_test,
                    texts_test,
                    cfg.run_mode.plot_path,
                )
            )
        score = np.mean(scores)
        print(
            f"--- Avg. accuracy across {cfg.classifier_mode.cv_folds} folds (cv-score) is: "
            f"{score}, SD={np.std(scores)}---"
        )
        if cfg.classifier_mode:
            timestamp = datetime.now().strftime("%b-%d-%Y-%H-%M-%S")
            out_path = hydra.utils.to_absolute_path(cfg.classifier_mode.out_path)
            output_path = os.path.join(
                out_path,
                cfg.classifier.name,
                build_experiment_name(cfg, f_ending=""),
                timestamp,
            )
            save_model(output_path, model, logger)
    else:
        model.fit(X_dev_emb, Y_dev)
        score = evaluate_model(
            cfg.embedding.name,
            cfg.classifier.name,
            model,
            X_test_emb,
            Y_test,
            texts_test,
            cfg.run_mode.plot_path,
        )
        if cfg.classifier_mode:
            timestamp = datetime.now().strftime("%b-%d-%Y-%H-%M-%S")
            out_path = hydra.utils.to_absolute_path(cfg.classifier_mode.out_path)
            output_path = os.path.join(
                out_path,
                cfg.classifier.name,
                build_experiment_name(cfg, f_ending=""),
                timestamp,
            )
            save_model(output_path, model, logger)
    return score
