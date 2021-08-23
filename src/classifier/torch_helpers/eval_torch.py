import os
import json

import numpy as np
import hydra
import torch
from torch.nn import functional as F
from sklearn.metrics import f1_score

from src.classifier.utils import store_preds
from src.classifier.visualizers.plots import plot_conf_matrix


def evaluate(
    model, test_loader, texts_test, classes, name_str, output_path, plot=False
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model = model.to(device)
    num_correct = 0
    n_classes = len(classes)
    confusion_matrix = torch.zeros(n_classes, n_classes)
    preds_all = []
    labels_all = []

    for inputs, labels in test_loader:
        inputs, labels = inputs.float().to(device), labels.long().to(device)
        test_outputs = model(inputs)
        if isinstance(test_outputs, tuple):
            test_outputs = test_outputs[0]
        probs = F.log_softmax(test_outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        preds_all += preds.float()
        labels_all += labels.float()

    preds_all, labels_all = torch.tensor(preds_all), torch.FloatTensor(labels_all)
    correct_tensor = preds_all.eq(labels_all.view_as(preds_all))
    correct = np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)
    num_per_class = {c: 0 for c in classes}
    for t, p in zip(labels_all, preds_all):
        num_per_class[int(p.item())] += 1
        confusion_matrix[t.long(), p.long()] += 1

    print(f"Confusion matrix: {confusion_matrix}")
    acc_list = confusion_matrix.diag() / confusion_matrix.sum(1)
    acc_per_class = dict(zip(classes, acc_list))
    mean_acc = np.mean(acc_list.cpu().numpy())
    mean_f1_macro = f1_score(
        labels_all.cpu().numpy(),
        preds_all.cpu().numpy(),
        labels=list(classes),
        average="macro",
    )
    mean_f1_micro = f1_score(
        labels_all.cpu().numpy(),
        preds_all.cpu().numpy(),
        labels=list(classes),
        average="micro",
    )
    print(f"Num per class: {num_per_class}")
    print(f"Test Accuracy per class: {acc_per_class}")
    print(f"Test Accuracy averaged: {mean_acc}")
    print(f"Test F1-score macro-averaged: {mean_f1_macro}")
    print(f"Test F1-score micro-averaged: {mean_f1_micro}")
    conf_matrix_npy = confusion_matrix.cpu().numpy()
    if plot:
        os.makedirs(output_path, exist_ok=True)
        plot_conf_matrix(conf_matrix_npy, output_path, f"conf_matrix_{name_str}")

    store_preds(output_path, name_str, preds_all, labels_all, texts_test)
    results_dict = {
        "acc_per_class": str(acc_per_class),
        "mean_acc": str(mean_acc),
        "mean_f1_macro": str(mean_f1_macro),
        "mean_f1_micro": str(mean_f1_micro),
    }
    with open(os.path.join(output_path, f"results_{name_str}.json"), "w") as outfile:
        json.dump(results_dict, outfile)
    np.save(os.path.join(output_path, f"conf_matrix.npy"), conf_matrix_npy)
    return mean_acc, results_dict, conf_matrix_npy
