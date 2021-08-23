from sklearn.ensemble import RandomForestClassifier
import xgboost


def suggest_xgb(model_params, trial, xgb=None):
    n_estimators = trial.suggest_int(
        model_params.n_estimators.name,
        model_params.n_estimators.lower,
        model_params.n_estimators.upper,
        model_params.n_estimators.step,
    )
    lr = trial.suggest_float(
        model_params.learning_rate.name,
        model_params.learning_rate.lower,
        model_params.learning_rate.upper,
        log=True,
    )
    max_depth = trial.suggest_int(
        model_params.max_depth.name,
        model_params.max_depth.lower,
        model_params.max_depth.upper,
        model_params.max_depth.step,
    )

    classifier = xgboost.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=lr,
        max_depth=max_depth,
        random_state=42,
        use_label_encoder=False,
        tree_method="gpu_hist",
        gpu_id=0,
    )
    return classifier


def suggest_rf(model_params, trial):
    n_estimators = trial.suggest_int(
        model_params.n_estimators.name,
        model_params.n_estimators.lower,
        model_params.n_estimators.upper,
        model_params.n_estimators.step,
    )
    max_depth = trial.suggest_int(
        model_params.max_depth.name,
        model_params.max_depth.lower,
        model_params.max_depth.upper,
        model_params.max_depth.step,
    )

    classifier = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    return classifier
