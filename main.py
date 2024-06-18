import numpy as np
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


def on_change():
    if "text_vectors" in st.session_state:
        del st.session_state["text_vectors"]


def main():
    st.title("Twitter Sentiment t-SNE Reduction")

    random_seed = st.number_input(label="Random seed", min_value=0)

    uploaded_file = st.file_uploader("Choose a CSV file with Tweets", type="csv")

    if uploaded_file is None:
        return

    data = load_data(uploaded_file)

    data_len = len(data)
    desired_data_len = st.slider(
        "Adjust desired data size",
        min_value=1,
        max_value=data_len,
        value=data_len,
        on_change=on_change,
    )
    if desired_data_len < data_len:
        data = data.sample(n=desired_data_len, random_state=random_seed)

    option_vectorizer = st.selectbox(
        "Select vectorizer",
        (
            "TF-IDF",
            "averaged word2vec",
            "pretrained averaged word2vec",
            "doc2vec",
        ),
        on_change=on_change,
    )

    vec_button = st.button("Vectorize")
    if vec_button:
        text_data = preprocess_text(data["text"])
        match option_vectorizer:
            case "TF-IDF":
                text_vectors = vectorize_with_tfidf(text_data)
            case "averaged word2vec":
                text_vectors = vectorize_with_avg_word2vec(text_data)
            case "pretrained averaged word2vec":
                text_vectors = vectorize_with_pretrained_avg_word2vec(text_data)
            case "doc2vec":
                text_vectors = vectorize_with_doc2vec(text_data)

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
            perplexity = st.number_input(
                "t-SNE: Perplexity",
                value=30,
                min_value=5,
                max_value=50,
            )
            learning_rate = st.number_input(
                "t-SNE: Learning Rate",
                value=200.0,
                min_value=10.0,
                max_value=1000.0,
            )
        case "UMAP":
            n_neighbors = st.number_input(
                "UMAP: Number of Neighbors",
                value=15,
                min_value=2,
                max_value=50,
            )
            min_dist = st.number_input(
                "UMAP: Minimum Distance",
                value=0.1,
                min_value=0.001,
                max_value=0.5,
            )
        case "PaCMAP":
            n_neighbors = st.number_input(
                "PaCMAP: Number of Neighbors",
                value=10,
                min_value=3,
                max_value=50,
            )

    vis_button = st.button("Visualize")
    if not vis_button:
        return

    match option_visualizer:
        case "t-SNE":
            results = perform_tsne(
                text_vectors,
                perplexity=perplexity,
                learning_rate=learning_rate,
                random_seed=random_seed,
            )
        case "UMAP":
            results = perform_umap(
                text_vectors,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_seed=random_seed,
            )
        case "PaCMAP":
            results = perform_pacmap(
                text_vectors,
                n_neighbors=n_neighbors,
                random_seed=random_seed,
            )

    label_mapping = {0: "negative", 2: "neutral", 4: "positive"}
    data["label"] = data["target"].apply(lambda v: label_mapping.get(v, ""))

    # Calculate metrics using LocalMetric class
    local_metric = LocalMetric()
    local_metric.calculate_knn_gain_and_dr_quality(
        text_vectors, results, data["label"].values, option_visualizer
    )

    data["x"], data["y"] = results[:, 0], results[:, 1]

    color = alt.Color("label")
    click = alt.selection_multi(encodings=["color"])
    brush = alt.selection_interval(encodings=["x", "y"])
    points = (
        alt.Chart()
        .mark_point()
        .encode(x="x", y="y", color=alt.condition(brush, "label", alt.value("lightgray")), tooltip=["label", "user", "date", "text"])
    ).transform_filter(click)\
    .add_selection(brush)\
    
    points_interactive = (
        alt.Chart()
        .mark_point()
        .encode(x="x", y="y", color=alt.condition(brush, "label", alt.value("lightgray")), tooltip=["label", "user", "date", "text"])
    ).transform_filter(click)\
    .interactive()

    hist = (
        alt.Chart()
        .mark_bar()
        .encode(x="count()", y="label", color=alt.condition(click, "label", alt.value("lightgray")))
    ).add_selection(click)\
    .transform_filter(brush)

    chart = alt.vconcat(points, hist, points_interactive, data=data)

    st.altair_chart(chart)

    # Display metrics
    metrics_kg = local_metric.visualize_kg()
    metrics_rnx = local_metric.visualize_rnx()

    mean_L_cf = np.mean(local_metric.L_cf)
    st.write(f"Mean class fidelity (CF): {mean_L_cf:.4f}")

    st.pyplot(metrics_kg)
    st.pyplot(metrics_rnx)


if __name__ == "__main__":
    main()
