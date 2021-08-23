from src.classifier.torch_helpers.torch_training import get_weight_vector


def suggest_sbert(model_params, trial, Y, device, n_embed, n_output):
    n_hidden_lin = trial.suggest_categorical(
        model_params.n_hidden_lin.name, model_params.n_hidden_lin.choices
    )

    n_hidden_lin_2 = trial.suggest_categorical(
        model_params.n_hidden_lin_2.name, model_params.n_hidden_lin_2.choices
    )

    batch_size = trial.suggest_categorical(
        model_params.batch_size.name, model_params.batch_size.choices
    )
    lr = trial.suggest_float(
        model_params.learning_rate.name,
        model_params.learning_rate.lower,
        model_params.learning_rate.upper,
        log=True,
    )
    dropout = trial.suggest_float(
        model_params.dropout.name,
        model_params.dropout.lower,
        model_params.dropout.upper,
        step=model_params.dropout.step,
    )

    weight_vector = get_weight_vector(Y, device)

    hyperparameters = dict(
        n_embed=n_embed,
        n_hidden_lin=n_hidden_lin,
        n_hidden_lin_2=n_hidden_lin_2,
        n_output=n_output,
        lr=lr,
        weight_vector=weight_vector,
        drop_p=dropout,
    )

    # return batch size separately as this is given to dataloader not model
    return hyperparameters, batch_size
