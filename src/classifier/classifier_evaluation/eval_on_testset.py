import os

import hydra

from src.classifier.torch_helpers.load_pretrained import load_torch_model
from src.classifier.torch_helpers.torch_dataloader import get_dataloader
from src.classifier.torch_helpers.eval_torch import evaluate
from src.classifier.non_torch.save_and_load_model import load_model
from src.classifier.non_torch.eval_non_torch import evaluate_model


def evaluate_on_test_set(cfg, X_test, Y_test, texts_test):
    if cfg.dev_settings.annotation == "majority":
        model_path = cfg.classifier_mode.model_path.majority
    else:
        model_path = cfg.classifier_mode.model_path.unanimous
    model_path = hydra.utils.to_absolute_path(model_path)
    results_path = cfg.classifier_mode.results_path
    results_path = hydra.utils.to_absolute_path(results_path)

    dest = os.path.join(
        results_path,
        cfg.classifier.name,
    )

    os.makedirs(dest, exist_ok=True)
    name_str = f"{cfg.embedding.name}_{cfg.classifier.name}"

    if cfg.classifier.name == "xgb":
        model = load_model(model_path)
        evaluate_model(
            cfg.embedding.name,
            cfg.classifier.name,
            model,
            X_test,
            Y_test,
            texts_test,
            dest,
        )
    else:
        model = load_torch_model(model_path, cfg.classifier.name, logger=None)
        model.to("cpu")
        model.eval()

        test_loader = get_dataloader(
            X_test, Y_test, cfg.classifier_mode.batch_size, shuffle=False
        )

        _, _, _ = evaluate(
            model,
            test_loader,
            texts_test,
            classes=set(Y_test),
            name_str=name_str,
            output_path=dest,
            plot=True,
        )
