from functools import partial

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from data_loader import load_data
from preprocessing import (
    preprocess_text,
    vectorize_with_avg_word2vec,
    vectorize_with_pretrained_avg_word2vec,
    vectorize_with_doc2vec,
    vectorize_with_tfidf,
)
from visualizers import (
    perform_tsne,
    perform_umap,
    perform_pacmap,
)
from metrics import LocalMetric


def remove_text_vectors():
    if "text_vectors" in st.session_state:
        del st.session_state["text_vectors"]


def remove_data():
    if "data" in st.session_state:
        del st.session_state["data"]


def remove_data_and_text_vectors():
    remove_data()
    remove_text_vectors()


def main():
    st.title("Text Datasets Visualization")

    random_seed = st.number_input(label="Random seed", min_value=0)

    option_dataset = st.selectbox(
        "Dataset",
        (
            "load from csv file",
            "url to parquet file",
            "airbnb_embeddings",
            "embedded_movies_small",
        ),
        on_change=remove_data_and_text_vectors,
    )

    match option_dataset:
        case "load from csv file":
            uploaded_file = st.file_uploader(
                "Choose a CSV file", type="csv", on_change=remove_data_and_text_vectors
            )
            data_loader = lambda: load_data(uploaded_file)
        case "url to parquet file":
            url = st.text_input("Parquet URL", on_change=remove_data_and_text_vectors)
            data_loader = lambda: pd.read_parquet(url)
        case "airbnb_embeddings":
            data_loader = lambda: pd.read_parquet(
                "https://huggingface.co/datasets/MongoDB/airbnb_embeddings/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet?download=true"
            )
        case "embedded_movies_small":
            data_loader = lambda: pd.read_parquet(
                "hf://datasets/acloudfan/embedded_movies_small/data/train-00000-of-00001.parquet"
            )

    load_button = st.button("Load dataset")
    if "data" in st.session_state:
        data = st.session_state["data"]
    elif load_button:
        with st.spinner("Loading dataset"):
            data = data_loader()
        st.session_state["data"] = data
    else:
        return

    data_len = len(data)
    start, end = st.slider(
        "Adjust desired data size",
        value=[0, data_len],
        on_change=remove_text_vectors,
    )
    data = data[start:end]

    dataframe_view_select = st.selectbox(
        "View dataframe",
        (
            "head",
            "full",
        ),
    )
    match dataframe_view_select:
        case "head":
            st.write(data.head())
        case "full":
            st.write(data)

    option_vectorizer = st.selectbox(
        "Vectorizer/Embeddings",
        (
            "existing embeddings",
            "TF-IDF",
            "averaged word2vec",
            "pretrained averaged word2vec",
            "doc2vec",
        ),
        on_change=remove_text_vectors,
    )

    text_or_embeddings_col = st.selectbox(
        "Column with embeddings/texts",
        data.columns,
    )

    get_text_or_embeddings = lambda: data[text_or_embeddings_col]

    match option_vectorizer:
        case "existing embeddings":
            vectorizer = lambda: np.array(list(map(np.array, get_text_or_embeddings())))
        case "TF-IDF":
            vectorizer = lambda: vectorize_with_tfidf(
                preprocess_text(get_text_or_embeddings())
            )
        case "averaged word2vec":
            vectorizer = lambda: vectorize_with_avg_word2vec(
                preprocess_text(get_text_or_embeddings)
            )
        case "pretrained averaged word2vec":
            vectorizer = lambda: vectorize_with_pretrained_avg_word2vec(
                preprocess_text(data[text_or_embeddings_col])
            )
        case "doc2vec":
            vectorizer = lambda: vectorize_with_doc2vec(
                preprocess_text(data[text_or_embeddings_col])
            )

    vec_button = st.button("Vectorize")
    if vec_button:
        with st.spinner("Vectorizing"):
            text_vectors = vectorizer()
        st.session_state["text_vectors"] = text_vectors
    elif "text_vectors" in st.session_state:
        text_vectors = st.session_state["text_vectors"]
    else:
        return

    option_visualizer = st.selectbox(
        "Select visualizer",
        ("t-SNE", "UMAP", "PaCMAP"),
    )

    match option_visualizer:
        case "t-SNE":
            perplexity = st.slider(
                "t-SNE: Perplexity",
                value=30,
                min_value=5,
                max_value=50,
                step=1,
            )
        case "UMAP":
            n_neighbors = st.slider(
                "UMAP: Number of Neighbors",
                value=15,
                min_value=2,
                max_value=50,
                step=1,
            )
            min_dist = st.slider(
                "UMAP: Minimum Distance",
                value=0.1,
                min_value=0.001,
                max_value=0.5,
            )
        case "PaCMAP":
            n_neighbors = st.slider(
                "PaCMAP: Number of Neighbors",
                value=10,
                min_value=3,
                max_value=50,
                step=1,
            )

    tooltip_tags = st.multiselect(
        "Tootlip tags",
        data.columns,
    )

    label_column = st.selectbox("Select label column", list(data.columns) + ["nolabel"])

    vis_button = st.button("Visualize")
    if not vis_button:
        return

    match option_visualizer:
        case "t-SNE":
            visualizer = partial(
                perform_tsne,
                perplexity=perplexity,
                random_seed=random_seed,
            )
        case "UMAP":
            visualizer = partial(
                perform_umap,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_seed=random_seed,
            )
        case "PaCMAP":
            visualizer = partial(
                perform_pacmap,
                n_neighbors=n_neighbors,
                random_seed=random_seed,
            )

    with st.spinner("Visualizing"):
        results = visualizer(text_vectors)

    match label_column:
        case "nolabel":
            data["label"] = None
        case _:
            data["label"] = data[label_column]

    data["x"], data["y"] = results[:, 0], results[:, 1]

    click = alt.selection_multi(encodings=["color"])
    brush = alt.selection_interval(encodings=["x", "y"])
    points = (
        alt.Chart(title="Selection chart")
        .mark_point()
        .encode(
            x="x",
            y="y",
            color=alt.condition(brush, "label", alt.value("lightgray")),
            tooltip=tooltip_tags,
        )
        .transform_filter(click)
        .add_selection(brush)
    )
    points_interactive = (
        alt.Chart(title="Interactive chart")
        .mark_point()
        .encode(
            x="x",
            y="y",
            color=alt.condition(brush, "label", alt.value("lightgray")),
            tooltip=tooltip_tags,
        )
        .transform_filter(click)
        .interactive()
    )

    hist = (
        alt.Chart(title="Class distribution")
        .mark_bar()
        .encode(
            x="count()",
            y="label",
            color=alt.condition(click, "label", alt.value("lightgray")),
        )
        .add_selection(click)
        .transform_filter(brush)
    )

    chart = alt.vconcat(points, hist, points_interactive, data=data)

    st.altair_chart(chart)

    # Calculate metrics using LocalMetric class
    local_metric = LocalMetric()
    local_metric.calculate_knn_gain_and_dr_quality(
        text_vectors, results, data["label"].values, option_visualizer
    )

    # Display metrics
    metrics_kg = local_metric.visualize_kg()
    metrics_rnx = local_metric.visualize_rnx()

    mean_L_cf = np.mean(local_metric.L_cf)
    st.write(f"Mean class fidelity (CF): {mean_L_cf:.4f}")

    st.pyplot(metrics_kg)
    st.pyplot(metrics_rnx)


if __name__ == "__main__":
    main()
