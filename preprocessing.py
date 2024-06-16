import re

import numpy as np
from gensim.models import Word2Vec
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
