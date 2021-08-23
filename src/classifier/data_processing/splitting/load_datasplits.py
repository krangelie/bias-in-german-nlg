import os
import pickle


def load_cached(cfg, file_name, logger):
    if os.path.exists(os.path.join(cfg.data_cache, file_name)):

        logger.info(f"Loading {file_name}")
        embedded_splits = pickle.load(
            open(os.path.join(cfg.data_cache, file_name), "rb")
        )
        X_train_emb = embedded_splits["X_train_emb"]
        X_test_emb = embedded_splits["X_test_emb"]
        Y_train = embedded_splits["Y_train"]
        Y_test = embedded_splits["Y_test"]

        return X_train_emb, X_test_emb, Y_train, Y_test

    return None


def load_dev_test(cfg):
    pass
