#!/usr/bin/env python3

"""
These functions collectively lowercase, remove punctuation, tokenize, lemmatize / stem, and find top words for corpus by TFIDF.
"""

# Standard library imports
from collections import defaultdict
from pathlib import Path
import string
from typing import Any, Dict, DefaultDict, List, Tuple

# Third part imports
import gensim
from gensim.corpora.dictionary import Dictionary
from gensim.interfaces import TransformedCorpus
import nltk
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd


DATA_DIR = Path("/work/data")
RAW_DIR = DATA_DIR / "raw"
SRC = RAW_DIR / "Musical_instruments_reviews.csv"
PROCESSED_DIR = DATA_DIR / "processed" 
DST = PROCESSED_DIR / "stemming_output.csv"
LIMIT = 15


def tokenize(series: pd.Series) -> List[List[str]]:
    """
    Tokenize a series of sentences / documents.
    Args:
        series: Series of sentences / documents.
    Returns: List of list of strings
    """
    return (
        series
        .str.lower()
        .str.replace(r'[^\w\s]+', '', regex=True)
        .str.strip()
        .apply(word_tokenize)
        .tolist()
    )


def remove_stopwords(sentences: List[List[str]], stopwords: List[str]=stopwords.words("english")) -> List[List[str]]:
    """
    Remove stopwords from list of sentences / documents.
    Args:
        sentences: List of sentences / documents.
        stopwords: List of stopwords.
    Returns: List of sentences / documents minus stopwords.
    """
    return [[word for word in sent if word not in stopwords] for sent in sentences]


def stem(sentences: List[List[str]], language="english") -> List[List[str]]:
    """
    Stem sentences / documents.
    Args:
        sentences: List of sentences / documents.
    Returns: List of stemmed stentences / documents.
    """
    stemmer = SnowballStemmer(language=language)
    return [[stemmer.stem(word) for word in sent] for sent in sentences]


def lemmatize(sentences: List[List[str]]) -> List[List[str]]:
    """
    Lemmatize sentences.
    Args:
        sentences: List of sentences / documents.
    Returns: List of lemmatized sentences / documents.
    """
    lemmatizer = WordNetLemmatizer()
    return [[lemmatizer.lemmatize(word) for word in sent] for sent in sentences]

def make_bow_corpus(documents: List[List[str]], dictionary: Dictionary):
    """
    Make a bag-of-words corpus from a list of documents (sentences) and a GenSim dictionary.
    Args:
        documents: List of list of strings.
        dictionary: GenSim dictionary made from documents.
    Returns: Bag-of-words corpus.
    Based on: https://ithaka.github.io/tdm-notebooks/finding-significant-terms.html
    """
    bow_corpus = []
    for document in documents:
        bow_corpus.append(dictionary.doc2bow(document))
    return bow_corpus


def make_corpus_tfidf(bow_corpus: List[Tuple[int, int]]) -> Tuple:
    """
    Create term frequency inverse document frequency corpus.
    Args:
        bow_corpus: Bag of words corpus.
    Returns: Gensim TFIDF model and corpus TFIDF.
    Based on: https://ithaka.github.io/tdm-notebooks/finding-significant-terms.html
    """
    model = gensim.models.TfidfModel(bow_corpus) 
    corpus_tfidf = model[bow_corpus]
    return model, corpus_tfidf


def make_token_dict(documents: List[List[str]]) -> Dict[str, int]:
    """
    Make Gensim token dictionary.
    Args:
        documents: List of documents.
    Returns: Dictionary of token counts.
    """
    return gensim.corpora.Dictionary(documents)

# For each document, print the ID, most significant/unique word, and TF/IDF score
def find_top_terms_by_doc(corpus_tfidf: TransformedCorpus, dictionary: Dictionary):
    """
    Find the most significant word in terms of TFIDF for each document.
    Args:
        corpus_tfidf: TFIDF matrix for corpus.
        dictionary: Gensim dictionary.
    Returns: Top term for each document by TFIDF score.
    Based on: https://ithaka.github.io/tdm-notebooks/finding-significant-terms.html
    """
    top_terms = []
    for document in corpus_tfidf:
        if len(document) >= 1:
            word_id, score = max(document, key=lambda x: x[1])
            top_terms.append(dict(word_id=word_id, score=score))
    words = pd.DataFrame(dictionary.items(), columns=["word_id", "word"])
    return pd.DataFrame(top_terms, columns=["word_id", "score"]).merge(words, on="word_id").set_index("word_id")


def find_top_terms(top_words_by_doc: pd.DataFrame, limit=5) -> pd.Series:
    """
    Find the top term across documents by taking mean scores for each word.
    Args: 
        top_words_by_doc: Top words produced by find_top_terms_by_doc function.
        limit: Number of top terms to return.
    Returns: Top n terms.
    Based on https://ithaka.github.io/tdm-notebooks/finding-significant-terms.html
    """
    return (
        top_words_by_doc
        .groupby("word")
        .score
        .mean()
        .sort_values(ascending=False)
        .head(limit)
    )

def score_terms(sentences: List[List[str]], lemmatize_: bool=True, limit=5) -> pd.Series:
    """
    Lowercase, remove punctuation, tokenize, lemmatize / stem, and find top words for corpus by TFIDF.
    Args:
        data: List of tokens.
        lemmatize_: True to lemmatize; False to stem.
        limit: Return top n words by TFIDF.
    Returns: List of top n words and associated TFIDF.
    """
    if lemmatize_:
        docs = lemmatize(sentences)
    else:
        docs = stem(sentences)
    docs_dict = make_token_dict(docs)
    docs_bow_corpus = make_bow_corpus(docs, docs_dict)
    docs_model, docs_corpus_tfidf = make_corpus_tfidf(docs_bow_corpus)
    docs_top_terms_by_doc = find_top_terms_by_doc(docs_corpus_tfidf, docs_dict)
    docs_top_terms = find_top_terms(docs_top_terms_by_doc, limit=limit)
    return docs_top_terms

if __name__ == "__main__":
    data = pd.read_csv(SRC)
    stem_top_term_scores = score_terms(nostops, lemmatize_=False, limit=15)
    lem_top_term_scores = score_terms(nostops, lemmatize_=True, limit=15)
    print("Top words from stemming")
    print(stem_top_term_scores)
    print("")
    print("Top words from lemmatizing")
    print(lem_top_term_scores)
    print(f"Save outputs to {PROCESSED_DIR}")
    stem_top_term_scores(PROCESSED_DIR / "stem_top_term_scores.csv")
    lem_top_term_scores(PROCESSED_DIR / "lem_top_term_scores.csv")