import pandas as pd
import numpy as np
from scipy import stats
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def variance_filter_tool(threshold: float = 0.0) -> dict:
        """Filter terms in the document-term matrix based on variance.

        Identifies near-constant terms whose variance across all documents falls below
        the specified threshold.

        Args:
            threshold (float): Minimum variance required for a term to be kept.
                Defaults to 0.0.

        Returns:
            dict: Summary containing result_id, threshold, kept_terms, and removed_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No document-term matrix found. Run build_dtm_tool first."}

        matrix = session.feature_matrix
        dense_matrix = matrix.toarray() if hasattr(matrix, "toarray") else np.array(matrix)
        variances = np.var(dense_matrix, axis=0)

        df_report = pd.DataFrame({
            "term": session.feature_names,
            "variance": variances
        }).sort_values(by="variance", ascending=False).reset_index(drop=True)

        kept_mask = df_report["variance"] >= threshold
        kept_terms = df_report[kept_mask]["term"].tolist()
        removed_terms = df_report[~kept_mask]["term"].tolist()

        result_id = session.next_result_id("variance_filter")
        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
            "total_terms": len(session.feature_names),
            "n_kept": len(kept_terms),
            "n_removed": len(removed_terms)
        }

        session.store_result(
            tool_name="variance_filter_tool",
            args={"threshold": threshold},
            summary=summary,
            full_report=df_report
        )

        return summary

    @tool
    def pearson_filter_tool(target_class: str, threshold: float = 0.0) -> dict:
        """Compute Pearson correlation between each term count and a target class.

        Uses One-vs-Rest encoding for the target class (1 for target_class, 0 otherwise).

        Args:
            target_class (str): Name of the target category to correlate with.
            threshold (float): Minimum absolute correlation required to keep a term. Defaults to 0.0.

        Returns:
            dict: Summary containing result_id, target_class, threshold, correlations, kept_terms, removed_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No document-term matrix found. Run build_dtm_tool first."}
        if session.dataframe is None or "category_name" not in session.dataframe.columns:
            return {"status": "error", "message": "No dataset or category_name column found in session."}

        matrix = session.feature_matrix
        dense_matrix = matrix.toarray() if hasattr(matrix, "toarray") else np.array(matrix)
        
        y = (session.dataframe["category_name"] == target_class).astype(float).values

        correlations = []
        for i in range(dense_matrix.shape[1]):
            x = dense_matrix[:, i]
            if np.std(x) == 0 or np.std(y) == 0:
                r = 0.0
            else:
                r, _ = stats.pearsonr(x, y)
                if np.isnan(r):
                    r = 0.0
            correlations.append(float(r))

        df_report = pd.DataFrame({
            "term": session.feature_names,
            "pearson_r": correlations
        }).sort_values(by="pearson_r", ascending=False).reset_index(drop=True)

        kept_mask = df_report["pearson_r"].abs() >= threshold
        kept_terms = df_report[kept_mask]["term"].tolist()
        removed_terms = df_report[~kept_mask]["term"].tolist()

        result_id = session.next_result_id("pearson_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
            "total_terms": len(session.feature_names),
            "n_kept": len(kept_terms)
        }

        session.store_result(
            tool_name="pearson_filter_tool",
            args={"target_class": target_class, "threshold": threshold},
            summary=summary,
            full_report=df_report
        )

        return summary

    @tool
    def spearman_filter_tool(target_class: str, threshold: float = 0.0) -> dict:
        """Compute Spearman rank correlation between each term count and a target class.

        Uses One-vs-Rest encoding for the target class (1 for target_class, 0 otherwise).

        Args:
            target_class (str): Name of the target category to correlate with.
            threshold (float): Minimum absolute correlation required to keep a term. Defaults to 0.0.

        Returns:
            dict: Summary containing result_id, target_class, threshold, correlations, kept_terms, removed_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No document-term matrix found. Run build_dtm_tool first."}
        if session.dataframe is None or "category_name" not in session.dataframe.columns:
            return {"status": "error", "message": "No dataset or category_name column found in session."}

        matrix = session.feature_matrix
        dense_matrix = matrix.toarray() if hasattr(matrix, "toarray") else np.array(matrix)
        
        y = (session.dataframe["category_name"] == target_class).astype(float).values

        correlations = []
        for i in range(dense_matrix.shape[1]):
            x = dense_matrix[:, i]
            if np.std(x) == 0 or np.std(y) == 0:
                rho = 0.0
            else:
                rho, _ = stats.spearmanr(x, y)
                if np.isnan(rho):
                    rho = 0.0
            correlations.append(float(rho))

        df_report = pd.DataFrame({
            "term": session.feature_names,
            "spearman_r": correlations
        }).sort_values(by="spearman_r", ascending=False).reset_index(drop=True)

        kept_mask = df_report["spearman_r"].abs() >= threshold
        kept_terms = df_report[kept_mask]["term"].tolist()
        removed_terms = df_report[~kept_mask]["term"].tolist()

        result_id = session.next_result_id("spearman_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
            "total_terms": len(session.feature_names),
            "n_kept": len(kept_terms)
        }

        session.store_result(
            tool_name="spearman_filter_tool",
            args={"target_class": target_class, "threshold": threshold},
            summary=summary,
            full_report=df_report
        )

        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
