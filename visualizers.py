from sklearn.manifold import TSNE
from umap import UMAP
from pacmap import PaCMAP


def perform_tsne(
    data,
    n_components=2,
    perplexity=30.0,
    learning_rate=200.0,
    random_seed=0,
):
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        random_state=random_seed,
    )
    results = tsne.fit_transform(data)
    return results


# def perform_umap(
#     data,
#     random_seed=0,
# ):
#     mapper = UMAP(
#         random_state=random_seed,
#     )
#     results = mapper.fit_transform(data)
#     return results


# def perform_pacmap(
#     data,
#     random_seed=0,
# ):
#     mapper = PaCMAP(
#         random_state=random_seed,
#     )
#     results = mapper.fit_transform(data)
#     return results
def perform_umap(
    data,
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    random_seed=0,
):
    mapper = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_seed,
    )
    results = mapper.fit_transform(data)
    return results


def perform_pacmap(
    data,
    n_components=2,
    n_neighbors=5,
    random_seed=0,
):
    mapper = PaCMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        random_state=random_seed,
    )
    results = mapper.fit_transform(data)
    return results