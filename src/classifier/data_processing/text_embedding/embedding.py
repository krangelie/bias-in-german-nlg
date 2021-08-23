import os

import hydra.utils
from sentence_transformers import SentenceTransformer
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.fasttext import (
    FastText,
    load_facebook_vectors,
    load_facebook_model,
)


def get_embedding(cfg, X=None):
    if cfg.embedding.name != "transformer":
        emb_path = hydra.utils.to_absolute_path(cfg.embedding.path)
    else:
        emb_path = cfg.embedding.path

    if cfg.embedding.name == "w2v":
        embedding = KeyedVectors.load_word2vec_format(
            emb_path, binary=False, no_header=cfg.embedding.no_header
        )

    elif cfg.embedding.name == "fastt":
        if cfg.run_mode.name == "data" and cfg.pre_processing.tune:
            dest_path = os.path.join(
                cfg.embedding.tuned_path, f"{cfg.pre_processing.epochs}_epochs"
            )
            os.makedirs(dest_path, exist_ok=True)
            dest_file = os.path.join(dest_path, "model.bin")

            # if not os.path.isfile(dest_file):
            print(
                f"Tuning {cfg.embedding.name} for {cfg.pre_processing.epochs} epochs."
            )
            embedding = load_facebook_model(emb_path)
            embedding.build_vocab(
                X, update=True
            )  # adds previously unseen words to vocab
            embedding.train(X, total_examples=len(X), epochs=cfg.pre_processing.epochs)
            # embedding.save(dest_file)
            # print(f"Saved finetuned {cfg.embedding.name} as {dest_file}.")

        else:
            embedding = load_facebook_vectors(emb_path)
    elif cfg.embedding.name == "transformer":
        embedding = SentenceTransformer(emb_path)
    else:
        raise SystemExit(f"{cfg.embedding.name} not implemented.")

    return embedding
