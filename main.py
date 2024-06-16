import streamlit as st
import matplotlib.pyplot as plt

from data_loader import load_data
from tsne import perform_tsne
from ui import (
    adjust_data_size_ui,
    show_data_button,
)
from preprocessing import preprocess_text, vectorize_text


def main():
    st.title("Twitter Sentiment t-SNE Reduction")

    uploaded_file = st.file_uploader("Choose a CSV file with Tweets", type="csv")

    if uploaded_file is None:
        return

    data = load_data(uploaded_file)

    show_data_button(data, "Show Original Data")

    st.subheader("Adjust Data Size")
    data_size_percentage = adjust_data_size_ui()
    if data_size_percentage < 100:
        data = data.sample(frac=data_size_percentage / 100)

    show_data_button(data, "Show Modified Data")

    if st.button("Perform t-SNE"):
        text_data = preprocess_text(data["text"])

        # Convert text data to TF-IDF features
        text_vectors = vectorize_text(text_data)

        labels = data["target"]

        # Map sentiment labels to numerical values
        # label_mapping = {"positive": 0, "neutral": 1, "negative": 2}

        # Determine the perplexity based on the number of samples
        num_samples = text_vectors.shape[0]
        perplexity = min(
            30, num_samples // 3
        )  # Choose a smaller perplexity if the sample size is small

        st.write("Performing t-SNE...")
        tsne_results = perform_tsne(text_vectors, perplexity=perplexity)

        fig, ax = plt.subplots()
        scatter = ax.scatter(
            tsne_results[:, 0],
            tsne_results[:, 1],
            c=labels,
            cmap="tab10",
            alpha=0.6,
        )
        legend = ax.legend(*scatter.legend_elements(), title="Sentiment")
        ax.add_artist(legend)
        st.pyplot(fig)


if __name__ == "__main__":
    main()
