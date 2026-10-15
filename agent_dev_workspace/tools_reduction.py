import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import OneHotEncoder
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def reduce_dimensions_tool(
        method: str = "pca",
        n_components: int = 2,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        random_state: int = 42
    ) -> dict:
        """Reduce dimensionality of the feature matrix using PCA, t-SNE, or UMAP, and plot scatter.

        Required plotting behavior: Creates a 2D scatter plot colored by document categories
        and assigns the figure to session.pending_figure.

        Args:
            method (str): Reduction method ('pca', 'tsne', or 'umap'). Defaults to 'pca'.
            n_components (int): Number of target dimensions. Defaults to 2.
            perplexity (float): Perplexity parameter for t-SNE. Defaults to 30.0.
            n_neighbors (int): Number of neighbors parameter for UMAP. Defaults to 15.
            random_state (int): Random seed for reproducibility. Defaults to 42.

        Returns:
            dict: Summary containing result_id, method, reduced_coordinates, and optional explained_variance_ratio.
        """
        if session.feature_matrix is None:
            return {"status": "error", "message": "No feature_matrix found. Run build_dtm_tool first."}

        matrix = session.feature_matrix
        dense_matrix = matrix.toarray() if hasattr(matrix, "toarray") else np.array(matrix)

        explained_variance_ratio = None

        method_lower = method.lower()
        if method_lower == "pca":
            reducer = PCA(n_components=n_components, random_state=random_state)
            coords = reducer.fit_transform(dense_matrix)
            explained_variance_ratio = reducer.explained_variance_ratio_.tolist()
        elif method_lower == "tsne":
            reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
            coords = reducer.fit_transform(dense_matrix)
        elif method_lower == "umap":
            reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=random_state)
            coords = reducer.fit_transform(dense_matrix)
        else:
            return {"status": "error", "message": f"Unsupported reduction method: {method}"}

        # 建立散佈圖並設置於 session.pending_figure
        labels = session.labels if session.labels is not None else session.dataframe.get("category_name", None)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        if labels is not None:
            sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=labels, palette="tab10", s=100, ax=ax)
        else:
            ax.scatter(coords[:, 0], coords[:, 1], s=100)

        ax.set_title(f"Dimensionality Reduction ({method_lower.upper()})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        plt.tight_layout()

        session.pending_figure = fig

        # 整理輸出
        result_id = session.next_result_id(f"reduce_{method_lower}")
        coords_list = coords.tolist()

        summary = {
            "result_id": result_id,
            "method": method_lower,
            "reduced_coordinates": coords_list,
            "n_samples": len(coords_list)
        }
        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = explained_variance_ratio

        session.artifacts[f"reduced_{method_lower}"] = coords
        session.store_result(
            tool_name="reduce_dimensions_tool",
            args={
                "method": method,
                "n_components": n_components,
                "perplexity": perplexity,
                "n_neighbors": n_neighbors,
                "random_state": random_state
            },
            summary=summary
        )

        return summary

    @tool
    def binarize_labels_tool() -> dict:
        """One-hot encode document category labels.

        Converts categorical category names into a one-hot encoded binary matrix.

        Returns:
            dict: Summary containing result_id, categories, and shape of the binarized matrix.
        """
        if session.dataframe is None or "category_name" not in session.dataframe.columns:
            return {"status": "error", "message": "No dataset with category_name found."}

        categories = session.dataframe["category_name"].values.reshape(-1, 1)
        encoder = OneHotEncoder(sparse_output=False)
        encoded_matrix = encoder.fit_transform(categories)
        category_classes = encoder.categories_[0].tolist()

        result_id = session.next_result_id("binarize_labels")
        summary = {
            "result_id": result_id,
            "categories": category_classes,
            "n_samples": encoded_matrix.shape[0],
            "n_categories": encoded_matrix.shape[1]
        }

        session.artifacts["binarized_labels"] = encoded_matrix
        session.artifacts["binarized_classes"] = category_classes

        session.store_result(
            tool_name="binarize_labels_tool",
            args={},
            summary=summary
        )

        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
