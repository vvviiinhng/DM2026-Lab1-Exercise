import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from langchain_core.tools import tool

# 確保 nltk 相關資源可用
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def make_tools(session):
    @tool
    def list_files_tool(subdirectory: str = "") -> dict:
        """List files and directories in a repository subdirectory.

        Args:
            subdirectory (str): Relative directory path to inspect. Defaults to "".

        Returns:
            dict: Summary containing result_id, subdirectory, files list, and directories list.
        """
        target_dir = os.path.join(".", subdirectory) if subdirectory else "."
        if not os.path.exists(target_dir) or not os.path.isdir(target_dir):
            return {"status": "error", "message": f"Directory '{subdirectory}' does not exist."}

        entries = os.listdir(target_dir)
        files = [e for e in entries if os.path.isfile(os.path.join(target_dir, e))]
        directories = [e for e in entries if os.path.isdir(os.path.join(target_dir, e))]

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "subdirectory": subdirectory,
            "files": sorted(files),
            "directories": sorted(directories)
        }

        session.store_result(
            tool_name="list_files_tool",
            args={"subdirectory": subdirectory},
            summary=summary
        )

        return summary

    @tool
    def inspect_data_tool(n_rows: int = 5) -> dict:
        """Inspect the first n rows and column schema of the active dataset in session.dataframe.

        Args:
            n_rows (int): Number of rows to peek. Defaults to 5.

        Returns:
            dict: Summary containing result_id, shape, columns list, and head records.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded in session. Run load_dataset_tool first."}

        df = session.dataframe
        head_records = df.head(n_rows).to_dict(orient="records")

        result_id = session.next_result_id("inspect_data")
        summary = {
            "result_id": result_id,
            "shape": list(df.shape),
            "columns": list(df.columns),
            "head": head_records
        }

        session.store_result(
            tool_name="inspect_data_tool",
            args={"n_rows": n_rows},
            summary=summary
        )

        return summary

    @tool
    def check_missing_tool() -> dict:
        """Check for missing or empty text values in session.dataframe.

        Args:

        Returns:
            dict: Summary containing result_id, missing_text_count, and missing_per_column dict.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded in session. Run load_dataset_tool first."}

        df = session.dataframe
        missing_per_col = df.isna().sum().to_dict()

        # 特殊檢查 text 欄位是否為空字串或純空白
        missing_text_count = 0
        if "text" in df.columns:
            empty_text_mask = df["text"].isna() | (df["text"].astype(str).str.strip() == "")
            missing_text_count = int(empty_text_mask.sum())

        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "missing_text_count": missing_text_count,
            "missing_per_column": {col: int(cnt) for col, cnt in missing_per_col.items()}
        }

        session.store_result(
            tool_name="check_missing_tool",
            args={},
            summary=summary
        )

        return summary

    @tool
    def check_duplicates_tool(drop: bool = False) -> dict:
        """Check for (and optionally drop) duplicate documents in session.dataframe.

        When drop=True, drops all copies of duplicated rows using keep=False.

        Args:
            drop (bool): Whether to remove duplicate rows from session.dataframe. Defaults to False.

        Returns:
            dict: Summary containing result_id, duplicate_count, and rows_remaining.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded in session. Run load_dataset_tool first."}

        df = session.dataframe
        # 以 text 欄位為主或整行比對
        target_cols = ["text"] if "text" in df.columns else None
        dup_mask = df.duplicated(subset=target_cols, keep=False)
        duplicate_count = int(dup_mask.sum())

        if drop and duplicate_count > 0:
            if target_cols:
                session.dataframe = df.drop_duplicates(subset=target_cols, keep=False).reset_index(drop=True)
            else:
                session.dataframe = df.drop_duplicates(keep=False).reset_index(drop=True)

        rows_remaining = len(session.dataframe)

        result_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": result_id,
            "duplicate_count": duplicate_count,
            "rows_remaining": rows_remaining
        }

        session.store_result(
            tool_name="check_duplicates_tool",
            args={"drop": drop},
            summary=summary
        )

        return summary

    @tool
    def sample_data_tool(n: int = 4, random_state: int = 42) -> dict:
        """Sample n random documents from session.dataframe using a fixed seed.

        Args:
            n (int): Number of rows to sample. Defaults to 4.
            random_state (int): Random seed for reproducibility. Defaults to 42.

        Returns:
            dict: Summary containing result_id, sampled_indices, and sample_preview records.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded in session. Run load_dataset_tool first."}

        df = session.dataframe
        n_sample = min(n, len(df))
        sampled_df = df.sample(n=n_sample, random_state=random_state)

        result_id = session.next_result_id("sample_data")
        summary = {
            "result_id": result_id,
            "sampled_indices": list(sampled_df.index),
            "sample_preview": sampled_df.to_dict(orient="records")
        }

        session.store_result(
            tool_name="sample_data_tool",
            args={"n": n, "random_state": random_state},
            summary=summary
        )

        return summary

    @tool
    def tokenize_tool() -> dict:
        """Tokenize text column in session.dataframe into lowercased unigrams using NLTK.

        Stores tokens list in session.dataframe['unigrams'] and session.artifacts['tokens'].

        Args:

        Returns:
            dict: Summary containing result_id, total_documents, and sample_tokens.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded in session. Run load_dataset_tool first."}

        df = session.dataframe
        if "text" not in df.columns:
            return {"status": "error", "message": "'text' column not found in session.dataframe."}

        tokens_list = []
        for text in df["text"]:
            if pd.isna(text) or not str(text).strip():
                tokens_list.append([])
            else:
                tokens = word_tokenize(str(text).lower())
                tokens_list.append(tokens)

        session.dataframe["unigrams"] = tokens_list
        session.artifacts["tokens"] = tokens_list

        result_id = session.next_result_id("tokenize")
        summary = {
            "result_id": result_id,
            "total_documents": len(df),
            "sample_tokens": tokens_list[:3]
        }

        session.store_result(
            tool_name="tokenize_tool",
            args={},
            summary=summary
        )

        return summary

    @tool
    def describe_data_tool() -> dict:
        """Compute text character length stats (overall and per category) and generate a boxplot.

        Sets the boxplot figure on session.pending_figure.

        Args:

        Returns:
            dict: Summary containing result_id, overall_stats, and per_category_stats.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded in session. Run load_dataset_tool first."}

        df = session.dataframe.copy()
        if "text" not in df.columns:
            return {"status": "error", "message": "'text' column not found in session.dataframe."}

        df["text_length"] = df["text"].apply(lambda x: len(str(x)) if pd.notna(x) and str(x).strip() else 0)

        # 全局長度數據統計
        overall_stats = df["text_length"].describe().to_dict()

        # 分類統計
        per_category_stats = {}
        cat_col = "category_name" if "category_name" in df.columns else ("category" if "category" in df.columns else None)

        if cat_col:
            for cat_val, group in df.groupby(cat_col):
                per_category_stats[str(cat_val)] = group["text_length"].describe().to_dict()

            # 繪製 Boxplot 並設置 pending_figure
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=df, x=cat_col, y="text_length", ax=ax)
            ax.set_title("Text Length Distribution by Category")
            ax.set_xlabel("Category")
            ax.set_ylabel("Text Length (characters)")
            plt.tight_layout()
            session.pending_figure = fig

        result_id = session.next_result_id("describe_data")
        summary = {
            "result_id": result_id,
            "overall_stats": overall_stats,
            "per_category_stats": per_category_stats
        }

        session.store_result(
            tool_name="describe_data_tool",
            args={},
            summary=summary
        )

        return summary

    return [
        list_files_tool,
        inspect_data_tool,
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        tokenize_tool,
        describe_data_tool,
    ]
