import re
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_text(text_series):
    # Basic text preprocessing
    processed_text = text_series.apply(lambda x: re.sub(r'\W', ' ', x.lower()))
    return processed_text

def vectorize_text(text_series, max_features=1000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    text_vectors = vectorizer.fit_transform(text_series).toarray()
    return text_vectors
