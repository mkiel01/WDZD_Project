import re

import numpy as np
import gensim
from gensim.models import Word2Vec, Doc2Vec
import gensim.downloader
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocess_text(text_series):
    # Basic text preprocessing
    processed_text = text_series.apply(lambda x: re.sub(r"\W", " ", x.lower()))
    return processed_text


def vectorize_with_tfidf(text_series, max_features=1000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    text_vectors = vectorizer.fit_transform(text_series).toarray()
    return text_vectors


def vectorize_with_avg_word2vec(data, max_features=100):
    model = Word2Vec(
        sentences=data,
        vector_size=max_features,
        window=5,
        min_count=1,
        workers=4,
    )
    vocab, vectors = model.wv.key_to_index, model.wv.vectors

    def avg_tweet(t):
        return np.array(list(map(lambda s: vectors[vocab[s]], t))).mean(axis=0)

    tweets_vectors = np.array(list(map(avg_tweet, data)))

    return tweets_vectors


def vectorize_with_pretrained_avg_word2vec(data):
    glove_vectors = gensim.downloader.load('glove-twitter-25')
    vocab, vectors = glove_vectors.key_to_index, glove_vectors.vectors

    def word2vec(w):
        if w not in vocab:
            return np.zeros((vectors.shape[1],))
        return vectors[vocab[w]]

    def avg_tweet(t):
        return np.array(list(map(lambda w: word2vec(w), t))).mean(axis=0)

    tweets_vectors = np.array(list(map(avg_tweet, data)))

    return tweets_vectors


def vectorize_with_doc2vec(data, max_features=100):
    def read_corpus(texts):
        for i, text in enumerate(texts):
            tokens = gensim.utils.simple_preprocess(text)
            yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

    model = Doc2Vec(
        documents=list(read_corpus(data)),
        vector_size=max_features,
        window=5,
        min_count=1,
        workers=4,
    )
    return model.dv.vectors
