import os
import pickle

import hydra.utils
import numpy as np

from src.classifier.data_processing.splitting.create_splits import get_data_splits
from src.classifier.utils import get_data_dir
from src.classifier.data_processing.data_augmentation.gendered_prompts import (
    replace_with_gendered_pronouns,
)
from src.classifier.visualizers.plots import plot_label_histogram, plt_labels_by_gender
from src.classifier.data_processing.annotate.annotate_sentences import (
    create_combined_df,
    clean_uncertain_labels,
    label_with_aggregate_annotation,
)
from src.classifier.data_processing.text_embedding.simple_tokenizer import (
    SimpleGermanTokenizer,
)
from src.classifier.data_processing.text_embedding.vectorizer import (
    # TfidfWeights,
    MeanEmbeddingVectorizer,
    WordEmbeddingVectorizer,
)
from src.classifier.data_processing.text_embedding.embedding import get_embedding


def _store_data(data, dest_dir, file_name):
    dest_dir = hydra.utils.to_absolute_path(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)
    pickle.dump(data, open(os.path.join(dest_dir, file_name), "wb"))
    print(f"Saved {file_name} at {dest_dir}.")


def _common_member(a, b):
    a_set = set(a)
    b_set = set(b)
    if a_set & b_set:
        return True
    else:
        return False


def main(cfg):
    # get df with all annotations
    dcfg = cfg.run_mode
    df, annotator_names = create_combined_df(dcfg.paths.raw_data_folder)
    # init tokenizer
    if cfg.embedding.name != "transformer":
        sgt = SimpleGermanTokenizer(
            dcfg.tokenize.to_lower,
            dcfg.tokenize.remove_punctuation,
            dcfg.tokenize.lemmatize,
            dcfg.tokenize.stem,
        )
        # tokenize
        df = sgt.tokenize(df, text_col=cfg.text_col)
        input_col = cfg.token_type
    else:
        # take full sentences for sentence transformer (will be tokenized later)
        input_col = cfg.text_col
    # get stored split indices (independent of pre-processing steps)
    if dcfg.augment:
        df = replace_with_gendered_pronouns(dcfg.augment, cfg.text_col, df)

    # if k_fold: returns lists of splits
    dev_set, test_set = get_data_splits(dcfg, cfg.label_col, df, annotator_names)

    if not dcfg.k_fold:
        preprocess_and_store_splits(cfg, dev_set, test_set, annotator_names)
    else:
        if cfg.specific_fold == -1:
            for fold in range(dcfg.k_fold):
                preprocess_and_store_splits(
                    cfg, dev_set[fold], test_set[fold], annotator_names, fold=fold
                )
        else:
            fold = cfg.specific_fold
            preprocess_and_store_splits(
                cfg, dev_set[fold], test_set[fold], annotator_names, fold=fold
            )


def preprocess_and_store_splits(cfg, dev_set, test_set, annotator_names, fold=None):
    dcfg = cfg.run_mode
    # make sure dev and test split are mutually exclusive
    assert not _common_member(dev_set.index.tolist(), test_set.index.tolist())
    split_names = ["dev_split", "test_split"]

    input_col = cfg.token_type if cfg.embedding.name != "transformer" else cfg.text_col
    # fit TFIDF on dev-set to get IDF-weights
    path_to_tfidf = hydra.utils.to_absolute_path(dcfg.paths.tfidf_weights)
    tfidf_weights = np.load(
        os.path.join(path_to_tfidf, "word2weight_idf.npy"), allow_pickle=True
    ).item()
    max_idf = np.load(
        os.path.join(path_to_tfidf, "max_idf.npy"),
        allow_pickle=True,
    )
    assert isinstance(tfidf_weights, dict)
    # load embedding dictionary or model in case of sentence-transformer
    # if set in config, tune on dev-set
    model = get_embedding(cfg, dev_set[input_col])
    # The following steps are done after splitting to ensure that the same
    # indices are used for dev and test set irrespective of the cleaning
    # procedure
    for i, split in enumerate([dev_set, test_set]):

        # clean out cases where annotators were uncertain
        split = clean_uncertain_labels(
            cfg.pre_processing.remove_uncertain, split, annotator_names
        )

        # annotate
        split = label_with_aggregate_annotation(
            dcfg.annotation,
            cfg.label_col,
            split,
            annotator_names,
        )
        Y_split = split[cfg.label_col]
        if not dcfg.augment:
            plot_label_histogram(
                Y_split,
                name=f"{split_names[i]} balanced on majority, " f"after cleaning",
            )
        else:
            plt_labels_by_gender(
                dcfg.annotation,
                dcfg.paths.plot_path,
                split,
                Y_split,
                name=split_names[i],
            )
        if -1 in Y_split:
            Y_split += 1  # avoid negative labels
        X_split = split[input_col]
        texts = split[cfg.text_col]

        if cfg.embedding.name != "transformer":
            # vectorize (for transformer, embedding will be applied later)
            if cfg.pre_processing.mean:
                vectorizer = MeanEmbeddingVectorizer(
                    model, tfidf_weights, max_idf=max_idf
                )
            else:
                vectorizer = WordEmbeddingVectorizer(
                    model,
                    tfidf_weights,
                    max_idf=max_idf,
                    seq_length=cfg.pre_processing.seq_length,
                )
            X_split = vectorizer.transform(X_split)
        else:
            # get sentence embeddings
            X_split = model.encode(X_split.tolist())

        # store
        if fold is None:
            _store_data(
                {"X": X_split, "Y": Y_split, "texts": texts},
                get_data_dir(cfg),
                split_names[i],
            )
        else:
            dest = os.path.join(get_data_dir(cfg), f"fold_{fold}")
            os.makedirs(dest, exist_ok=True)
            _store_data(
                {"X": X_split, "Y": Y_split, "texts": texts},
                dest,
                split_names[i],
            )
