import os
import logging
from abc import ABC, abstractmethod

from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import numpy as np

# https://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
# Get mean sentence embeddings


class Vectorizer(ABC):
    def __init__(self, word2vec, word2weight=None, max_idf=None, seq_length=0):
        self.log = logging.getLogger(__name__)
        self.word2vec = word2vec  # works also with fasttext embedding
        self.word2weight = word2weight  # tfidf weights
        self.max_idf = max_idf
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = word2vec.vector_size
        self.seq_length = seq_length

    @abstractmethod
    def transform(self):
        pass


class MeanEmbeddingVectorizer(Vectorizer):
    def transform(self, X):
        self.log.info("Creating word vectors and averaging over sentence.")
        if self.word2weight is not None:
            averaged_embs = []
            for words in X:
                word_list = []
                for w in words:
                    if w in self.word2vec:
                        weight = (
                            self.max_idf
                            if w not in self.word2weight
                            else self.word2weight[w]
                        )
                        word_list.append(self.word2vec[w] * weight)
                    else:
                        word_list.append([np.zeros(self.dim)])
                averaged_embs.append(np.mean(word_list, axis=0))
            averaged_embs = np.array(averaged_embs, dtype=object)
        else:
            averaged_embs = np.array(
                [
                    np.mean(
                        [self.word2vec[w] for w in words if w in self.word2vec]
                        or [np.zeros(self.dim)],
                        axis=0,
                    )
                    for words in X
                ],
                dtype=object,
            )

        return averaged_embs


class WordEmbeddingVectorizer(Vectorizer):
    def transform(self, X):
        self.log.info(
            "Creating word vectors and store as list of word vectors per sentence."
        )
        if self.word2weight is not None:
            sentence_list = []
            for words in X:
                word_list = []
                for w in words:
                    if w in self.word2vec:
                        weight = (
                            self.max_idf
                            if w not in self.word2weight
                            else self.word2weight[w]
                        )
                        word_list.append(self.word2vec[w] * weight)
                    else:
                        word_list.append([np.zeros(self.dim)])
                sentence_list.append(word_list)
            word_embs = np.array(sentence_list, dtype=object)

        else:
            word_embs = np.array(
                [
                    [self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)]
                    for words in X
                ],
                dtype=object,
            )

        if self.seq_length != 0:
            word_embs = self.add_padding(word_embs)
        return word_embs

    def add_padding(self, word_embs):
        # word_embs is list of lists with variable length
        # for each of those lists, pad with zero-list of length dim
        # or crop if list is too long

        word_embs_adjusted = []

        for words in word_embs:
            if len(words) >= self.seq_length:
                shortened = words[: self.seq_length]
                word_embs_adjusted.append(np.array(shortened))
            else:
                padded = np.array(
                    words + [np.zeros(self.dim)] * (self.seq_length - len(words))
                )
                word_embs_adjusted.append(padded)

        word_embs_ndarray = np.array(word_embs_adjusted)
        print(word_embs_ndarray.shape)
        return word_embs_ndarray
