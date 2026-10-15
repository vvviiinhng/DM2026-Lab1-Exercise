import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def cosine_similarity_tool(doc1_idx: int, doc2_idx: int) -> dict:
        """Compute the cosine similarity between two documents in the feature matrix.

        Uses raw feature counts from session.feature_matrix. Returns 0.0 if either document
        has zero norm (e.g. empty text).

        Args:
            doc1_idx (int): Index of the first document.
            doc2_idx (int): Index of the second document.

        Returns:
            dict: Summary containing result_id, doc1_idx, doc2_idx, and similarity score.
        """
        if session.feature_matrix is None:
            return {"status": "error", "message": "No feature_matrix found. Run build_dtm_tool first."}

        matrix = session.feature_matrix
        dense_matrix = matrix.toarray() if hasattr(matrix, "toarray") else np.array(matrix)

        n_docs = dense_matrix.shape[0]
        if not (0 <= doc1_idx < n_docs and 0 <= doc2_idx < n_docs):
            return {"status": "error", "message": f"Document index out of bounds. Must be between 0 and {n_docs - 1}."}

        vec1 = dense_matrix[doc1_idx].reshape(1, -1)
        vec2 = dense_matrix[doc2_idx].reshape(1, -1)

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            sim_score = 0.0
        else:
            sim_score = float(cosine_similarity(vec1, vec2)[0, 0])

        result_id = session.next_result_id("cosine_sim")
        summary = {
            "result_id": result_id,
            "doc1_idx": doc1_idx,
            "doc2_idx": doc2_idx,
            "similarity": sim_score
        }

        session.store_result(
            tool_name="cosine_similarity_tool",
            args={"doc1_idx": doc1_idx, "doc2_idx": doc2_idx},
            summary=summary
        )

        return summary

    @tool
    def feature_correlation_matrix_tool(top_n: int = 20) -> dict:
        """Compute Pearson correlation matrix across top terms by variance and render a heatmap.

        Operates on global DTM. Selects up to top_n terms with highest variance.
        Sets the figure on session.pending_figure.

        Args:
            top_n (int): Maximum number of top variance terms to include. Defaults to 20.

        Returns:
            dict: Summary containing result_id, top_terms, and the correlation_matrix list.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No feature_matrix found. Run build_dtm_tool first."}

        matrix = session.feature_matrix
        dense_matrix = matrix.toarray() if hasattr(matrix, "toarray") else np.array(matrix)
        feature_names = np.array(session.feature_names)

        # 1. 計算特徵變異數並取前 top_n 高變異數特徵
        variances = np.var(dense_matrix, axis=0)
        n_features = len(feature_names)
        actual_top_n = min(top_n, n_features)

        # 按變異數由高到低排序索引
        top_indices = np.argsort(variances)[::-1][:actual_top_n]
        top_terms = feature_names[top_indices].tolist()
        top_matrix = dense_matrix[:, top_indices]

        # 2. 計算 Pearson 相關係數矩陣 (rowvar=False 表示 columns 是變數)
        if top_matrix.shape[0] < 2:
            corr_matrix = np.eye(actual_top_n)
        else:
            corr_matrix = np.corrcoef(top_matrix, rowvar=False)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # 3. 繪製熱圖並設置於 session.pending_figure
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, xticklabels=top_terms, yticklabels=top_terms, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Feature Correlation Matrix (Top Variance Terms)")
        plt.tight_layout()

        session.pending_figure = fig

        # 4. 整理輸出
        result_id = session.next_result_id("feature_corr")
        corr_list = corr_matrix.tolist()

        summary = {
            "result_id": result_id,
            "top_terms": top_terms,
            "correlation_matrix": corr_list
        }

        session.store_result(
            tool_name="feature_correlation_matrix_tool",
            args={"top_n": top_n},
            summary=summary
        )

        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
