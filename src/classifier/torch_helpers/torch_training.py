import os
from datetime import datetime

import hydra.utils
import mlflow.pytorch
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split, StratifiedKFold


from src.classifier.classifier_training.classifier_utils import compute_weight_vector

from src.classifier.classifier_training.classifier_utils import get_classifier
from src.classifier.torch_helpers.eval_torch import evaluate
from src.classifier.visualizers.plots import aggregate_metrics
from src.classifier.torch_helpers.torch_dataloader import get_dataloader
from src.classifier.utils import build_experiment_name


def train_torch_model(
    cfg, X_dev_emb, X_test_emb, Y_dev, Y_test, classes, texts_test, seed=42
):
    print("Dev set size", len(X_dev_emb))
    output_path = hydra.utils.to_absolute_path(cfg.run_mode.plot_path)
    output_path = os.path.join(output_path, cfg.classifier.name)
    if cfg.dev_settings.annotation == "unanimous":
        hyperparameters = cfg.classifier.unanimous
    else:
        hyperparameters = cfg.classifier.majority
    weight_vector = compute_weight_vector(Y_dev, use_torch=True)
    batch_size = hyperparameters.batch_size
    gpu_params = cfg.run_mode.gpu
    model = get_classifier(
        hyperparameters, cfg.classifier.name, cfg.embedding.n_embed, weight_vector
    )
    test_loader = get_dataloader(X_test_emb, Y_test, batch_size, shuffle=False)

    if cfg.classifier_mode.cv_folds:
        skf = StratifiedKFold(n_splits=cfg.classifier_mode.cv_folds)
        accs, result_dicts, confs = [], [], []
        for train_index, val_index in skf.split(X_dev_emb, Y_dev):
            print(f"Num train {len(train_index)}, num val {len(val_index)}")

            X_train, X_val = X_dev_emb[train_index], X_dev_emb[val_index]
            Y_train, Y_val = (
                Y_dev.to_numpy()[train_index],
                Y_dev.to_numpy()[val_index],
            )
            train_loader = get_dataloader(X_train, Y_train, batch_size)
            val_loader = get_dataloader(X_val, Y_val, batch_size, shuffle=False)
            mean_acc, results_dict, conf_matrix_npy = _fit(
                cfg,
                model,
                train_loader,
                val_loader,
                test_loader,
                texts_test,
                classes,
                gpu_params,
                hyperparameters,
                seed,
                output_path,
            )
            accs.append(mean_acc)
            result_dicts.append(results_dict)
            confs.append(conf_matrix_npy)
        aggregate_metrics(result_dicts, confs, output_path)
        print(
            f"--- Avg. accuracy across {cfg.classifier_mode.cv_folds} folds (cv-score) is: "
            f"{np.mean(accs)}, SD={np.std(accs)}---"
        )
        if cfg.classifier_mode.cv_folds == "incremental_train":
            return accs
        else:
            return np.mean(accs)
    else:
        X_train_emb, X_val_emb, Y_train, Y_val = train_test_split(
            X_dev_emb,
            Y_dev,
            test_size=cfg.run_mode.val_split,
            shuffle=True,
            stratify=Y_dev,
            random_state=seed,
        )
        train_loader = get_dataloader(X_train_emb, Y_train, batch_size, shuffle=True)
        val_loader = get_dataloader(X_val_emb, Y_val, batch_size, shuffle=False)
        mean_acc, _, _ = _fit(
            cfg,
            model,
            train_loader,
            val_loader,
            test_loader,
            texts_test,
            classes,
            gpu_params,
            hyperparameters,
            seed,
            output_path,
        )
        return mean_acc


def _fit(
    cfg,
    model,
    train_loader,
    val_loader,
    test_loader,
    texts_test,
    classes,
    gpu_params,
    hyperparameters,
    seed,
    output_path,
):

    early_stopping = EarlyStopping(
        "val_loss", patience=hyperparameters.patience
    )  # change to val_loss
    callbacks = [early_stopping]
    if cfg.run_mode.store_after_training:
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=os.path.join(
                cfg.classifier.model_path,
                build_experiment_name(cfg, f_ending=None),
                datetime.now().strftime("%b-%d-%Y-%H-%M-%S"),
            ),
            filename="{epoch}-{step}-{val_loss:.2f}-{other_metric:.2f}",
            save_top_k=2,
            mode="min",
        )
        callbacks += [checkpoint_callback]
    if gpu_params.use_amp:
        trainer = pl.Trainer(
            precision=gpu_params.precision,
            amp_level=gpu_params.amp_level,
            amp_backend=gpu_params.amp_backend,
            gpus=gpu_params.n_gpus,
            max_epochs=hyperparameters.n_epochs,
            progress_bar_refresh_rate=20,
            callbacks=callbacks,
        )
    else:
        trainer = pl.Trainer(
            gpus=gpu_params.n_gpus,
            max_epochs=hyperparameters.n_epochs,
            progress_bar_refresh_rate=20,
            callbacks=callbacks,
        )
    mlflow.pytorch.autolog()
    # Train the model
    with mlflow.start_run() as run:
        trainer.fit(model, train_loader, val_loader)
        # trainer.test(model, test_loader)
    print(trainer.current_epoch)
    name_str = f"{cfg.embedding.name}_{cfg.classifier.name}"
    if seed != 42:
        output_path += f"_{seed}"
    mean_acc, results_dict, conf_matrix_npy = evaluate(
        model, test_loader, texts_test, classes, name_str, output_path
    )
    return mean_acc, results_dict, conf_matrix_npy


def get_weight_vector(Y, device):
    weight_vector = len(Y) / (len(set(Y)) * np.bincount(Y))
    weight_vector = torch.FloatTensor(weight_vector).to(device)
    return weight_vector
