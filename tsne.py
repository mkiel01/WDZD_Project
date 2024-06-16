from sklearn.manifold import TSNE


def perform_tsne(data, n_components=2, perplexity=30.0, learning_rate=200.0):
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
    )
    tsne_results = tsne.fit_transform(data)
    return tsne_results
