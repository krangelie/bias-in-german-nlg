from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.classifier.lstm.lstm_classifier import RegardLSTM
from src.classifier.sent_transformer.sbert_classifier import RegardBERT
from src.classifier.torch_helpers.torch_dataloader import get_dataloader


def fit_torch_model(
    model_type,
    X,
    Y,
    path,
    hyperparameters,
    batch_size,
    n_epochs,
    patience,
    gpu_params,
    scoring,
    skf,
    trial,
):
    scores = []

    for train_index, val_index in skf.split(X, Y):
        print(f"Num train {len(train_index)}, num val {len(val_index)}")

        if model_type == "lstm":
            classifier = RegardLSTM(**hyperparameters)
        elif model_type == "transformer":
            classifier = RegardBERT(**hyperparameters)

        X_train, X_val = X[train_index], X[val_index]
        Y_train, Y_val = (
            Y.to_numpy()[train_index],
            Y.to_numpy()[val_index],
        )
        train_loader = get_dataloader(X_train, Y_train, batch_size)
        val_loader = get_dataloader(X_val, Y_val, batch_size, shuffle=False)
        early_stopping = EarlyStopping(
            "val_loss", patience=patience
        )  # change to val_loss
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=path,
            filename="{epoch}-{step}-{val_loss:.2f}-{other_metric:.2f}",
            save_top_k=0,
            mode="min",
        )
        if gpu_params.use_amp:
            trainer = Trainer(
                precision=gpu_params.precision,
                amp_level=gpu_params.amp_level,
                amp_backend=gpu_params.amp_backend,
                gpus=gpu_params.n_gpus,
                checkpoint_callback=True,
                max_epochs=n_epochs,
                progress_bar_refresh_rate=20,
                callbacks=[
                    PyTorchLightningPruningCallback(trial, monitor=scoring),
                    early_stopping,
                    checkpoint_callback,
                ],
            )
        else:
            trainer = Trainer(
                gpus=gpu_params.n_gpus,
                checkpoint_callback=True,
                max_epochs=n_epochs,
                progress_bar_refresh_rate=20,
                callbacks=[
                    PyTorchLightningPruningCallback(trial, monitor=scoring),
                    early_stopping,
                    checkpoint_callback,
                ],
            )

        trainer.logger.log_hyperparams(hyperparameters)
        trainer.fit(classifier, train_loader, val_loader)
        scores.append(trainer.callback_metrics[scoring].item())
    return scores
