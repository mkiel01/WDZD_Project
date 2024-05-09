import streamlit as st
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Function to load data
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    return data

# Function to perform t-SNE
def perform_tsne(data, n_components=2, perplexity=30.0, learning_rate=200.0):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate)
    tsne_results = tsne.fit_transform(data)
    return tsne_results

# Streamlit interface
st.title('MNIST Dataset t-SNE Reduction')

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = load_data(uploaded_file)
    labels = data['label']
    pixels = data.drop('label', axis=1)
    
    # Perform t-SNE
    st.write("Performing t-SNE...")
    tsne_results = perform_tsne(pixels)
    
    # Plotting
    fig, ax = plt.subplots()
    scatter = ax.scatter(tsne_results[:,0], tsne_results[:,1], c=labels, cmap='tab10', alpha=0.6)
    legend = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend)
    st.pyplot(fig)
