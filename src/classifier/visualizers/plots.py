import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set(style="white")


def plot_label_histogram(label_col, name=""):
    label_col.hist()
    plt.title(name)
    if -1.0 in label_col:
        plt.xticks([-1, 0, 1])
    elif 2.0 in label_col:
        plt.xticks([0, 1, 2])
    plt.show()


def plt_labels_by_gender(annotation, plot_path, X, Y, name=""):
    dest_name = f"labels_by_gender_{annotation}_{name}.png"
    os.makedirs(plot_path, exist_ok=True)
    dest = os.path.join(plot_path, dest_name)
    plt.figure(figsize=(8, 6.5))

    Y = Y.map({-1: "negative", 0: "neutral", 1: "positive"})
    X["Gender representation"] = X["Gender"].map(
        {"F": "female", "N": "none", "M": "male"}
    )

    g = sns.histplot(
        data=X,
        x=Y,
        hue="Gender representation",
        palette="coolwarm",
        multiple="dodge",
        shrink=0.8,
        hue_order=["female", "male", "none"],
    )
    plt.setp(g.get_legend().get_texts(), fontsize="14")  # for legend text
    plt.setp(g.get_legend().get_title(), fontsize="14")  # for legend title

    plt.xlabel("Regard label", fontsize=16)
    plt.ylabel("Count", fontsize=16)
    plt.xticks(fontsize=14)
    plt.savefig(dest)


def aggregate_metrics(results_dicts, conf_matrices_npy, output_path):
    results_all = pd.DataFrame(results_dicts)
    for col in results_all.columns:
        if col != "acc_per_class":
            results_all[col] = results_all[col].astype(float)

    avg_conf = np.array(conf_matrices_npy).mean(axis=0)

    plot_conf_matrix(avg_conf, output_path, "avg_conf")


def plot_conf_matrix(conf_mat, output_path, name):
    labels = ["negative", "neutral", "positive"]
    plot = sns.heatmap(
        conf_mat,
        cmap="coolwarm",
        annot=True,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        annot_kws={"fontsize": 13},
    )
    plot.set_xlabel("True labels", fontsize=15)
    plot.set_ylabel("Predicted labels", fontsize=15)
    plt.savefig(os.path.join(output_path, f"{name}.jpg"))
    plt.close()
