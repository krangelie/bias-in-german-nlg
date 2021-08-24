import os
from pprint import pprint
import json

import hydra.utils
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer

from src.classifier.data_processing.text_embedding.simple_tokenizer import (
    SimpleGermanTokenizer,
)
from src.classifier.data_processing.text_embedding.embedding import get_embedding
from src.classifier.data_processing.text_embedding.vectorizer import (
    MeanEmbeddingVectorizer,
    WordEmbeddingVectorizer,
)
from src.classifier.non_torch.save_and_load_model import load_model
from src.classifier.torch_helpers.load_pretrained import load_torch_model
import src.constants as constants


def _vectorize(cfg, model_type, sentences, embedding_path=None, tfidf_weights=None):
    model = (
        get_embedding(cfg)
        if not embedding_path
        else SentenceTransformer(embedding_path)
    )

    if model_type != "transformer":
        if model_type != "lstm":
            vectorizer = MeanEmbeddingVectorizer(model, tfidf_weights)
        else:
            vectorizer = WordEmbeddingVectorizer(
                model,
                tfidf_weights,
            )
        sentences_emb = vectorizer.transform(sentences)
    else:
        sentences_emb = model.encode(sentences)
    return sentences_emb


def load_inference_data(cfg, model_type, path, embedding_path=None):
    if path.endswith(".txt"):
        with open(path) as f:
            lines = [line.rstrip() for line in f]
        sentence_df = pd.DataFrame(lines, columns=[cfg.text_col])
    else:
        sentence_df = pd.read_csv(path)
    sentence_df = sentence_df.dropna(subset=[cfg.text_col])
    if cfg.classifier_mode.add_demographic_terms:
        print("Add demographics")
        demographic_texts = add_demographics(
            sentence_df[cfg.text_col],
            constants.PERSON,
            cfg.classifier_mode.demographics,
        )
        gendered_text_embs = {}
        for gen, texts in demographic_texts.items():
            sentence_df, sentences_emb = embed_texts(
                cfg,
                embedding_path,
                model_type,
                pd.DataFrame(texts, columns=[cfg.text_col]),
            )
            gendered_text_embs[gen] = {
                "text_df": sentence_df,
                "text_emb": sentences_emb,
            }
        return gendered_text_embs
    else:
        if cfg.classifier_mode.add_first_demo:
            sentence_df[cfg.text_col] = sentence_df[cfg.text_col].apply(
                lambda txt: constants.VARIABLE_DICT[cfg.classifier_mode.demographics[0]]
                + " "
                + txt
            )
        sentence_df, sentences_emb = embed_texts(
            cfg, embedding_path, model_type, sentence_df
        )
        text_embs = {"text_df": sentence_df, "text_emb": sentences_emb}
        return text_embs


def embed_texts(cfg, embedding_path, model_type, sentence_df):
    if model_type != "transformer":
        sgt = SimpleGermanTokenizer(
            True,
            True,
            False,
            False,
        )
        sentence_df = sgt.tokenize(sentence_df, text_col=cfg.text_col)
        sentences_emb = _vectorize(
            cfg, model_type, sentence_df[cfg.token_type], embedding_path
        )
    else:
        sentences_emb = _vectorize(
            cfg, model_type, sentence_df[cfg.text_col], embedding_path
        )
    return sentence_df, sentences_emb


def store_preds_per_class(inference_params, dest, preds, sentence_df, text_col):
    classes = set(preds)
    class_map = {0: "negative", 1: "neutral", 2: "positive"}
    for c in classes:
        texts = sentence_df.loc[sentence_df["Prediction"] == c, text_col]
        if inference_params.cda:
            dest_curr = os.path.join(dest, class_map[c])
            os.makedirs(dest_curr, exist_ok=True)
            if list(
                filter(
                    any(texts.iloc[0].startswith(f) for f in constants.FEMALE_PREFIXES)
                )
            ):
                new_gender = "MALE"
                orig_gender = "FEMALE"
                flipped_texts = flip_gender(texts, True)
            elif list(
                filter(
                    any(texts.iloc[0].startswith(f) for f in constants.MALE_PREFIXES)
                )
            ):
                new_gender = "FEMALE"
                orig_gender = "MALE"
                flipped_texts = flip_gender(texts, False)

            text_per_gen = [texts, flipped_texts]
            for i, gen in enumerate([orig_gender, new_gender]):
                with open(
                    os.path.join(dest_curr, f"{gen}_{class_map[c]}_regard.txt"), "a+"
                ) as f:
                    for txt in text_per_gen[i]:
                        f.write(f"{remove_demographic(inference_params, txt)}\n")
        else:
            print("Storing predictions to", dest)
            with open(os.path.join(dest, f"{class_map[c]}_regard.txt"), "a+") as f:
                for txt in texts:
                    f.write(f"{remove_demographic(inference_params, txt)}\n")


def eval_prediction(dest, path, preds, sentence_df, label_col, store_misclassified):
    if sentence_df.dtypes[label_col] == str:
        sentence_df[label_col] = sentence_df[label_col].map(constants.VALENCE_MAP)
    classes = set(sentence_df[label_col])
    n_classes = len(classes)
    results_dict = classification_report(
        sentence_df[label_col], preds, output_dict=True
    )
    sentence_df[label_col] = sentence_df[label_col].astype(int)
    confusion_matrix = np.zeros((n_classes, n_classes))
    misclassified_idcs = []
    for t_idx, p in zip(sentence_df.index, preds):
        t = sentence_df.loc[t_idx, label_col]
        confusion_matrix[t, p] += 1
        if t != p:
            misclassified_idcs.append(t_idx)

    labels = ["negative", "neutral", "positive"]
    plot = sns.heatmap(
        confusion_matrix,
        cmap="coolwarm",
        annot=True,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        annot_kws={"fontsize": 13},
    )
    plot.set_xlabel("True labels", fontsize=15)
    plot.set_ylabel("Predicted labels", fontsize=15)
    name_str = f"{os.path.basename(path).split('.')[0]}"
    plt.savefig(os.path.join(dest, f"conf_matrix_{name_str}.png"))
    pprint(results_dict)
    with open(os.path.join(dest, f"results_{name_str}.json"), "w") as outfile:
        json.dump(results_dict, outfile)

    if store_misclassified:
        misclassified_df = sentence_df.loc[misclassified_idcs, :]
        misclassified_df["Prediction"] = preds[misclassified_idcs]
        misclassified_df.to_csv(os.path.join(dest, f"misclassified_{name_str}.csv"))
    print(f"Storing results at {dest}.")


def add_demographics(texts, placeholder, demographics_list):
    demo_added = {}

    for demo in demographics_list:
        demo_prefix = constants.VARIABLE_DICT[demo]
        demo_added[demo] = [txt.replace(placeholder, demo_prefix) for txt in texts]
        demo_added[demo] = flip_gender(
            demo_added[demo],
            any(demo_prefix == female for female in constants.FEMALE_PREFIXES),
        )
    print(demo_added)
    return demo_added


def flip_gender(texts, f_to_m):
    female_to_male = constants.F_TO_M_PRONOUNS
    flipped_texts = []
    for txt in texts:
        flipped_txt = []
        dictionary = female_to_male if f_to_m else female_to_male.inverse
        for word in txt.split(" "):
            if word in dictionary.keys():
                word = dictionary[word]
            flipped_txt += [word]
        flipped_texts += [" ".join(flipped_txt)]
    return flipped_texts


def remove_demographic(inference_params, text):
    for demo in inference_params.demographics:
        text = text.replace(demo, "").strip()
    return text


def predict_regard(
    cfg,
    input_path,
    output_path,
    by_class_results,
    model,
    model_type,
    use_sklearn_model,
    embedding_path,
):
    sent_dict = load_inference_data(cfg, model_type, input_path, embedding_path)

    def predict_regard_(sentence_df, sentences_emb, gen=None):
        if use_sklearn_model:
            preds = model.predict(sentences_emb)
        else:
            outputs = model(torch.Tensor(sentences_emb))
            probs = F.log_softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1).detach().numpy()

            if cfg.label_col in sentence_df.columns:
                sentence_df[cfg.label_col] = sentence_df[cfg.label_col].astype(int)
                eval_prediction(
                    output_path,
                    input_path,
                    preds,
                    sentence_df,
                    cfg.label_col,
                    cfg.classifier_mode.store_misclassified,
                )
        sentence_df["Prediction"] = preds

        if gen is None:
            dest = os.path.join(
                output_path,
                f"{os.path.basename(input_path).split('.')[0]}_regard_labeled.csv",
            )
        else:
            dest = os.path.join(output_path, f"{gen}_texts_regard_labeled.csv")
        sentence_df.to_csv(dest)
        print("Predictions stored at", dest)

        if by_class_results:
            store_preds_per_class(
                cfg.classifier_mode, output_path, preds, sentence_df, cfg.text_col
            )
        del sentences_emb

    if cfg.classifier_mode.add_demographic_terms:
        for gen, gen_dict in sent_dict.items():
            print("Processing texts for ", gen)
            predict_regard_(gen_dict["text_df"], gen_dict["text_emb"], gen)
    else:
        predict_regard_(sent_dict["text_df"], sent_dict["text_emb"])


def predict(
    cfg,
    eval_model=None,
    eval_model_type=None,
    embedding_path=None,
    sample_file=None,
    eval_dest=None,
    logger=None,
):
    inference_params = cfg.classifier_mode
    print(inference_params)
    text_path = inference_params.inference_texts if not sample_file else sample_file
    model_type = eval_model_type if eval_model_type else cfg.classifier.name
    results_path = hydra.utils.to_absolute_path(inference_params.results_path)
    if cfg.dev_settings.annotation == "unanimous":
        pretrained_model = inference_params.pretrained_model.unanimous
    else:
        pretrained_model = inference_params.pretrained_model.majority
    pretrained_model = hydra.utils.to_absolute_path(pretrained_model)

    dest = (
        os.path.join(
            results_path,
            model_type,
        )
        if not eval_dest
        else eval_dest
    )
    os.makedirs(dest, exist_ok=True)

    by_class_results = not eval_model and cfg.classifier_mode.split_by_class
    use_sklearn_model = not eval_model and model_type not in [
        "lstm",
        "transformer",
    ]

    if use_sklearn_model:
        model = load_model(pretrained_model, logger)
    else:
        model_path = pretrained_model if not eval_model else eval_model
        model = load_torch_model(model_path, model_type, logger=None)
        model.to("cpu")
        model.eval()

    if any([text_path.endswith(ending) for ending in [".csv", ".txt"]]):
        predict_regard(
            cfg,
            text_path,
            dest,
            by_class_results,
            model,
            model_type,
            use_sklearn_model,
            embedding_path,
        )
    else:
        text_path = hydra.utils.to_absolute_path(text_path)
        for file in os.listdir(text_path):
            path = os.path.join(text_path, file)
            if not os.path.isdir(path):
                print(f"Processing {path}")
                predict_regard(
                    cfg,
                    path,
                    dest,
                    by_class_results,
                    model,
                    model_type,
                    use_sklearn_model,
                    embedding_path,
                )
