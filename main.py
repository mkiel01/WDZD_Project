import altair as alt
import streamlit as st

from data_loader import load_data
from preprocessing import (preprocess_text, vectorize_with_avg_word2vec,
                           vectorize_with_tfidf)
from tsne import perform_tsne
from ui import adjust_data_size_ui


def main():
    st.title("Twitter Sentiment t-SNE Reduction")

    random_seed = st.number_input(label="Random seed", min_value=0)

    uploaded_file = st.file_uploader("Choose a CSV file with Tweets", type="csv")

    if uploaded_file is None:
        return

    data = load_data(uploaded_file)

    data_len = len(data)
    desired_data_len = adjust_data_size_ui(data_len)
    if desired_data_len < data_len:
        data = data.sample(n=desired_data_len, random_state=random_seed)

    if st.button("Perform t-SNE"):
        text_data = preprocess_text(data["text"])

        # Convert text data to TF-IDF features
        text_vectors = vectorize_with_avg_word2vec(text_data)

        # Map sentiment labels to numerical values
        label_mapping = {0: "negative", 2: "neutral", 4: "positive"}
        data["label"] = data["target"].apply(lambda v: label_mapping.get(v, ""))

        # Determine the perplexity based on the number of samples
        num_samples = text_vectors.shape[0]
        perplexity = min(
            30, num_samples // 3
        )  # Choose a smaller perplexity if the sample size is small

        tsne_results = perform_tsne(
            text_vectors, perplexity=perplexity, random_seed=random_seed
        )
        data["x"], data["y"] = tsne_results[:, 0], tsne_results[:, 1]

        points = (
            alt.Chart(data)
            .mark_point()
            .encode(
                x="x", y="y", color="label", tooltip=["label", "user", "date", "text"]
            )
        ).interactive()

        st.altair_chart(points, use_container_width=True)


if __name__ == "__main__":
    main()
