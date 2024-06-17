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
        (
            "t-SNE",
            "UMAP",
            "PaCMAP",
        ),
    )

    vis_button = st.button("Visualize")
    visualizer = None
    if vis_button:
        match option_visualizer:
            case "t-SNE":
                visualizer = perform_tsne
            case "UMAP":
                visualizer = perform_umap
            case "PaCMAP":
                visualizer = perform_pacmap
    else:
        return

    # Map sentiment labels to numerical values
    label_mapping = {0: "negative", 2: "neutral", 4: "positive"}
    data["label"] = data["target"].apply(lambda v: label_mapping.get(v, ""))
    # Determine the perplexity based on the number of samples
    num_samples = text_vectors.shape[0]
    perplexity = min(
        30, num_samples // 3
    )  # Choose a smaller perplexity if the sample size is small

    results = visualizer(text_vectors, random_seed=random_seed)

    data["x"], data["y"] = results[:, 0], results[:, 1]

    points = (
        alt.Chart(data)
        .mark_point()
        .encode(x="x", y="y", color="label", tooltip=["label", "user", "date", "text"])
    ).interactive()

    st.altair_chart(points, use_container_width=True)


if __name__ == "__main__":
    main()
