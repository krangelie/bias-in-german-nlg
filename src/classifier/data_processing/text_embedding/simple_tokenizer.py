import string
from HanTa import HanoverTagger as ht
from pprint import pprint
from pandas import DataFrame
import nltk


class SimpleGermanTokenizer:
    """Simple tokenization for sentences. Used for non-Transformer-based models."""

    def __init__(
        self,
        to_lower=True,
        use_remove_punctuation=True,
        use_lemmatize=False,
        use_stem=False,
    ):
        self.to_lower = to_lower
        self.use_remove_punctuation = use_remove_punctuation
        self.use_lemmatize = use_lemmatize
        self.use_stem = use_stem
        self.lemmatizer = (
            ht.HanoverTagger("morphmodel_ger.pgz") if use_lemmatize else None
        )
        self.stemmer = Cistem() if use_stem else None

    def tokenize(self, df: DataFrame, text_col="text") -> DataFrame:
        """
        Takes as input a DataFrame with sentences in column 'text_col'.
        Appends at least one new column to the DataFrame containing lists of single word tokens.
        """

        texts = df[text_col]
        if self.to_lower:
            texts = texts.apply(lambda text: text.lower())

        if self.use_remove_punctuation:
            texts = texts.apply(lambda text: self.remove_punctuation(text))

        # Split sentences into list of words
        df["tokenized"] = texts.apply(
            lambda text: nltk.tokenize.word_tokenize(text, language="german")
        )

        if self.use_lemmatize:
            df["lemmata"] = df["tokenized"].apply(lambda tokens: self.lemmatize(tokens))

        if self.use_stem:
            df["stems"] = df["tokenized"].apply(lambda tokens: self.stem(tokens))

        return df

    def remove_punctuation(self, text):
        regular_punct = list(string.punctuation)
        for punc in regular_punct:
            if punc in text:
                text = text.replace(punc, " ")
        return text.strip()

    # https://textmining.wp.hs-hannover.de/Preprocessing.html
    def lemmatize(self, tokenized_wordlist):
        lemmata = self.lemmatizer.tag_sent(
            tokenized_wordlist, taglevel=1
        )  # What is tag level? # return 3-tuple (word, lemma, POS-tag)
        # pprint(lemmata)
        lemmata_only = [entry[1] for entry in lemmata]
        return lemmata_only

    def stem(self, tokenized_wordlist):
        stems = [self.stemmer.stem(token) for token in tokenized_wordlist]
        return stems
