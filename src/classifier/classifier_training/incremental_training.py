import os

import hydra.utils
import pandas as pd

from src.classifier.classifier_training.training import train_classifier


def _get_data_increments(data_len, percentage):
    # creates list of upperbounds to index different amounts of data
    increments = []
    for p in range(percentage, 100 + percentage, percentage):
        upper_bound = int(data_len * (p / 100))
        increments.append(upper_bound)
    return increments


def train_on_increments(
    cfg, X_dev, Y_dev, X_test, Y_test, texts_test, logger, num_runs=5
):

    data = pd.DataFrame(columns=["id", "step", "metric"])
    plot_path = hydra.utils.to_absolute_path(cfg.run_mode.plot_path)
    os.makedirs(plot_path, exist_ok=True)
    counter = 0
    if cfg.classifier_mode.mult_seeds:
        seeds = [21, 42, 84, 168, 336]
        for run in range(num_runs):
            print(f"---- RUN {run} ----")
            data, counter = _run(
                cfg,
                X_dev,
                X_test,
                Y_dev,
                Y_test,
                texts_test,
                counter,
                data,
                logger,
                run,
                seeds[run],
            )
    else:
        data, _ = _run(
            cfg, X_dev, X_test, Y_dev, Y_test, texts_test, counter, data, logger
        )
    print(data)

    data.to_csv(os.path.join(plot_path, "incr_train_data.csv"))
    print(
        "Stored results at",
        plot_path,
        "\nCreate a plot in " "incremental_training_exploration.ipynb",
    )


def _run(
    cfg,
    X_dev,
    X_test,
    Y_dev,
    Y_test,
    texts_test,
    counter,
    data,
    logger,
    run=None,
    seed=42,
):
    print("Full size", len(X_dev))
    increments = _get_data_increments(len(X_dev), cfg.classifier_mode.percentage)
    for upper_bound in increments:
        print(f"{upper_bound} data points")
        X_dev_i, Y_dev_i = X_dev[:upper_bound], Y_dev[:upper_bound]
        scores = train_classifier(
            cfg, X_dev_i, Y_dev_i, X_test, Y_test, texts_test, logger, seed
        )
        if cfg.classifier_mode.cv_folds:
            for fold, score in enumerate(scores):
                data.loc[counter, "id"] = fold
                data.loc[counter, "step"] = upper_bound
                data.loc[counter, "metric"] = score
                counter += 1
        else:
            data.loc[counter, "id"] = run
            data.loc[counter, "step"] = upper_bound
            data.loc[counter, "metric"] = scores
            counter += 1
    return data, counter
