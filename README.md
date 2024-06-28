# Text Datasets Visualizer

This application can visualize text datasets by reducing dimensionality of existing or generated embeddings and plotting them using t-SNE, PaCMAP or UMAP.

## Usage

```
git clone https://github.com/mkiel01/WDZD_Project.git
cd WDZD_Project
docker build . --tag viz-texts
docker run -p 8501:8501 viz-texts
```

Open `localhost:8501` in your browser
