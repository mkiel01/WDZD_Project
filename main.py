import streamlit as st
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from data_loader import load_data, save_column_changes, remove_columns, add_column
from tsne import perform_tsne
from ui import (
    rename_columns_ui,
    remove_columns_ui,
    add_column_ui,
    adjust_data_size_ui,
    show_data_button,
)
from preprocessing import preprocess_text, vectorize_text


def main():
    # Streamlit interface
    st.title("Twitter Sentiment t-SNE Reduction")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = load_data(uploaded_file)

        # Show Original Data Button
        show_data_button(data, "Show Original Data")

        # Column renaming
        st.subheader("Rename Columns")
        if st.button("Show Rename Columns Options"):
            new_column_names = rename_columns_ui(data)
            if st.button("Save Column Changes"):
                data = save_column_changes(data, new_column_names)

        # Column removal
        st.subheader("Remove Columns")
        if st.button("Show Remove Columns Options"):
            columns_to_remove = remove_columns_ui(data)
            if st.button("Remove Selected Columns"):
                data = remove_columns(data, columns_to_remove)

        # Adding new columns
        st.subheader("Add New Column")
        if st.button("Show Add Column Options"):
            new_col_name, new_col_values = add_column_ui()
            if st.button("Add Column"):
                data = add_column(data, new_col_name, new_col_values)

        # Adjust data size
        st.subheader("Adjust Data Size")
        data_size_percentage = adjust_data_size_ui()
        if data_size_percentage < 100:
            data = data.sample(frac=data_size_percentage / 100)

        # Show Modified Data Button
        show_data_button(data, "Show Modified Data")

        if st.button("Perform t-SNE"):
            # Preprocess text data
            text_data = preprocess_text(data["text"])

            # Convert text data to TF-IDF features
            text_vectors = vectorize_text(text_data)

            labels = data["sentiment"]

            # Map sentiment labels to numerical values
            label_mapping = {"positive": 0, "neutral": 1, "negative": 2}
            label_numbers = labels.map(label_mapping)

            # Determine the perplexity based on the number of samples
            num_samples = text_vectors.shape[0]
            perplexity = min(
                30, num_samples // 3
            )  # Choose a smaller perplexity if the sample size is small

            # Perform t-SNE
            st.write("Performing t-SNE...")
            tsne_results = perform_tsne(text_vectors, perplexity=perplexity)

            # Plotting
            fig, ax = plt.subplots()
            scatter = ax.scatter(
                tsne_results[:, 0],
                tsne_results[:, 1],
                c=label_numbers,
                cmap="tab10",
                alpha=0.6,
            )
            legend = ax.legend(*scatter.legend_elements(), title="Sentiment")
            ax.add_artist(legend)
            st.pyplot(fig)


if __name__ == "__main__":
    main()
