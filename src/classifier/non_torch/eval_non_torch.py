import os
import json
from pprint import pprint

import hydra.utils
from sklearn.metrics import classification_report, plot_confusion_matrix
import matplotlib.pyplot as plt

from src.classifier.utils import store_preds


def evaluate_model(
    embedding_name, model_type, model, X_test, Y_test, texts_test, plot_path
):
    Y_pred = model.predict(X_test)
    report = classification_report(Y_test, Y_pred, output_dict=True)
    pprint(report)

    plot_confusion_matrix(model, X_test, Y_test)

    name_str = f"{embedding_name}_{model_type}"
    plot_path = hydra.utils.to_absolute_path(plot_path)
    os.makedirs(plot_path, exist_ok=True)
    plt.savefig(os.path.join(plot_path, f"conf_matrix_{name_str}.png"))
    with open(os.path.join(plot_path, f"report_{name_str}.json"), "w") as outfile:
        json.dump(report, outfile)

    store_preds(plot_path, name_str, Y_pred, Y_test, texts_test)
    acc = report["accuracy"]  # ["f1-score"]
    print("Storing evaluation results at", plot_path)
    return acc
