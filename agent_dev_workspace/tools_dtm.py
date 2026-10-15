from typing import List, Optional, Union, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer


def make_tools(session: Any) -> list:
    @tool
    def build_dtm_tool(
        ngram_range: List[int] = [1, 1],
        max_features: Optional[int] = None,
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
    ) -> Dict[str, Any]:
        """Builds a document-term matrix (DTM) from the loaded dataset.

        Args:
            ngram_range: List of two ints [min_n, max_n] specifying the n-gram range.
            max_features: Maximum number of terms/features to include in the vocabulary.
            min_df: Minimum document frequency for terms (int count or float ratio).
            max_df: Maximum document frequency for terms (int count or float ratio).

        Returns:
            Dict containing result_id, n_documents, vocabulary_size, non_zero, total_elements, and sparsity_pct.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded in session."}

        # Convert list to tuple for CountVectorizer
        ngram_tuple = (ngram_range[0], ngram_range[1])

        # Initialize and fit CountVectorizer
        vectorizer = CountVectorizer(
            ngram_range=ngram_tuple,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
        )

        texts = session.dataframe["text"].fillna("").tolist()
        feature_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out().tolist()

        # Update session state
        session.feature_matrix = feature_matrix
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)

        # Compute sparsity metrics
        n_rows, n_cols = feature_matrix.shape
        non_zero = int(feature_matrix.nnz)
        total_elements = int(n_rows * n_cols)
        sparsity_pct = 100.0 * (1.0 - non_zero / total_elements) if total_elements > 0 else 0.0

        res_id = session.next_result_id("dtm")
        summary = {
            "result_id": res_id,
            "n_documents": n_rows,
            "vocabulary_size": n_cols,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
        }

        session.store_result("build_dtm_tool", {"ngram_range": ngram_range, "max_features": max_features, "min_df": min_df, "max_df": max_df}, summary)

        return summary

    @tool
    def term_frequency_tool() -> Dict[str, Any]:
        """Aggregate term frequencies across all documents in the global document-term matrix.

        Args:

        Returns:
            Dict containing result_id, vocabulary_size, and term_frequencies dict.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No feature_matrix found. Run build_dtm_tool first."}

        matrix = session.feature_matrix
        feature_names = session.feature_names

        # 彙整各 Term 在全體 Document 的總次數 (column sum)
        counts = np.array(matrix.sum(axis=0)).flatten()
        term_freqs = {term: int(count) for term, count in zip(feature_names, counts)}

        res_id = session.next_result_id("term_freq")

        # 同時建立包含 full_report 的 DataFrame 以利衍生工具讀取
        report_df = pd.DataFrame({
            "term": feature_names,
            "frequency": counts
        }).sort_values(by="frequency", ascending=False)

        summary = {
            "result_id": res_id,
            "vocabulary_size": len(feature_names),
            "term_frequencies": term_freqs
        }

        session.store_result(
            tool_name="term_frequency_tool",
            args={},
            summary=summary,
            full_report=report_df
        )

        return summary

    @tool
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20) -> Dict[str, Any]:
        """Render a heatmap of a slice of the raw document-term matrix.

        Slices the global feature_matrix by [0:n_documents, 0:n_terms] without sorting.
        Sets the heatmap figure on session.pending_figure.

        Args:
            n_terms (int): Number of terms/columns to include. Defaults to 20.
            n_documents (int): Number of documents/rows to include. Defaults to 20.

        Returns:
            Dict containing result_id, sliced_matrix list, terms list, and document_indices list.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No feature_matrix found. Run build_dtm_tool first."}

        matrix = session.feature_matrix
        feature_names = np.array(session.feature_names)

        n_docs_total, n_terms_total = matrix.shape
        actual_n_docs = min(n_documents, n_docs_total)
        actual_n_terms = min(n_terms, n_terms_total)

        # 依照 positional slice 切片 [0:actual_n_docs, 0:actual_n_terms]
        sliced_sparse = matrix[0:actual_n_docs, 0:actual_n_terms]
        sliced_dense = sliced_sparse.toarray() if hasattr(sliced_sparse, "toarray") else np.array(sliced_sparse)
        sliced_terms = feature_names[0:actual_n_terms].tolist()
        doc_indices = list(range(actual_n_docs))

        # 繪製 Heatmap 並設置在 session.pending_figure
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(sliced_dense, xticklabels=sliced_terms, yticklabels=doc_indices, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
        ax.set_title(f"DTM Heatmap (First {actual_n_docs} Docs x First {actual_n_terms} Terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Document Index")
        plt.tight_layout()

        session.pending_figure = fig

        res_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": res_id,
            "sliced_matrix": sliced_dense.tolist(),
            "terms": sliced_terms,
            "document_indices": doc_indices
        }

        session.store_result(
            tool_name="dtm_heatmap_tool",
            args={"n_terms": n_terms, "n_documents": n_documents},
            summary=summary
        )

        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
