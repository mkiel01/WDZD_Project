import streamlit as st
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from data_loader import load_data, save_column_changes, remove_columns, add_column
from tsne import perform_tsne
from ui import rename_columns_ui, remove_columns_ui, add_column_ui, adjust_data_size_ui

# Streamlit interface
st.title('MNIST Dataset t-SNE Reduction')

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.write("Original Data")
    st.dataframe(data)

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

    st.write("Modified Data")
    st.dataframe(data)

    if st.button("Perform t-SNE"):
        labels = data['label']
        pixels = data.drop('label', axis=1)
        
        # Perform t-SNE
        st.write("Performing t-SNE...")
        tsne_results = perform_tsne(pixels)
        
        # Plotting
        fig, ax = plt.subplots()
        scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='tab10', alpha=0.6)
        legend = ax.legend(*scatter.legend_elements(), title="Classes")
        ax.add_artist(legend)
        st.pyplot(fig)
