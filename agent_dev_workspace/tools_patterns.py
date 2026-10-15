import tempfile
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: str = "tfidf",
        algorithm: str = "fpgrowth",
        min_sup: int = 1,
        k: int = 10
    ) -> dict:
        """Mine frequent itemsets/patterns for a single document category.

        Builds a category-local DTM with English stopword removal, applies vocabulary
        filtering (variance, tfidf, or term_frequency), converts to a transactional
        database, and runs PAMI algorithms (FPGrowth, FAE top-k, or MaxFPGrowth).

        Args:
            category_name (str): The target category label to filter session.dataframe by.
            filtering_method (str): Filtering approach ('variance', 'tfidf', or 'term_frequency').
                Defaults to 'tfidf'.
            algorithm (str): Mining algorithm ('fpgrowth', 'topk', or 'maxfpgrowth').
                Defaults to 'fpgrowth'.
            min_sup (int): Minimum support count threshold for 'fpgrowth' and 'maxfpgrowth'.
                Defaults to 1.
            k (int): Number of top patterns for 'topk' algorithm. Defaults to 10.

        Returns:
            dict: Summary containing result_id, category_name, filtering_method, algorithm, and patterns list.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns or "category_name" not in session.dataframe.columns:
            return {"status": "error", "message": "No dataset with text and category_name columns found."}

        # 1. 篩選該類別的文章並建立 Category-local DTM (去掉英文停用詞)
        cat_df = session.dataframe[session.dataframe["category_name"] == category_name]
        texts = cat_df["text"].dropna().tolist()
        texts = [t for t in texts if t.strip() != ""]

        if not texts:
            return {
                "category_name": category_name,
                "patterns": [],
                "note": f"No non-empty text found for category {category_name}"
            }

        vectorizer = CountVectorizer(stop_words="english")
        try:
            dtm_matrix = vectorizer.fit_transform(texts)
        except ValueError:
            return {
                "category_name": category_name,
                "patterns": [],
                "note": "Vocabulary empty after stop-word removal"
            }

        feature_names = np.array(vectorizer.get_feature_names_out())
        dense_dtm = dtm_matrix.toarray()
        ascii_mask = np.array([t.isascii() for t in feature_names])
        feature_names = feature_names[ascii_mask]
        dense_dtm = dense_dtm[:, ascii_mask]

        if dense_dtm.shape[1] == 0:
            return {
                "category_name": category_name,
                "patterns": [],
                "note": "No terms found in DTM"
            }

        # 2. 應用選定的 filtering_method
        if filtering_method == "variance":
            scores = np.var(dense_dtm, axis=0)
            if len(scores) > 0:
                p5, p95 = np.percentile(scores, 5), np.percentile(scores, 95)
                mask = (scores >= p5) & (scores <= p95)
            else:
                mask = np.array([], dtype=bool)
        elif filtering_method == "term_frequency":
            scores = np.sum(dense_dtm, axis=0)
            if len(scores) > 0:
                p5, p95 = np.percentile(scores, 5), np.percentile(scores, 95)
                mask = (scores >= p5) & (scores <= p95)
            else:
                mask = np.array([], dtype=bool)
        elif filtering_method == "tfidf":
            transformer = TfidfTransformer()
            tfidf_matrix = transformer.fit_transform(dtm_matrix)
            scores = np.mean(tfidf_matrix.toarray(), axis=0)
            if len(scores) > 0:
                p20 = np.percentile(scores, 20)
                mask = scores >= p20
            else:
                mask = np.array([], dtype=bool)
        else:
            return {"status": "error", "message": f"Unknown filtering_method: {filtering_method}"}

        survived_features = feature_names[mask]
        survived_dtm = dense_dtm[:, mask]

        # 3. 如果無任何詞彙存活
        if len(survived_features) == 0:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived filtering"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        # 4. 轉成 Transactional Database
        binary_df = pd.DataFrame((survived_dtm >= 1).astype(int), columns=survived_features)

        with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", delete=False, suffix=".txt") as tmp_db:
            db_path = tmp_db.name

        try:
            from PAMI.extras.convert import DF2DB
            converter = DF2DB.DF2DB(binary_df)
            converter.convert2TransactionalDatabase(db_path, ">=", 1)

            patterns = []
            # 5. 根據 algorithm 執行 PAMI 探勘
            if algorithm == "fpgrowth":
                from PAMI.frequentPattern.basic import FPGrowth
                obj = FPGrowth.FPGrowth(iFile=db_path, minSup=min_sup, sep="\t")
                obj.startMine()
                df_res = obj.getPatternsAsDataFrame()
                if df_res is not None and not df_res.empty:
                    for _, row in df_res.iterrows():
                        patterns.append({"pattern": str(row["Patterns"]), "support": int(row["Support"])})

            elif algorithm == "topk":
                from PAMI.frequentPattern.topk import FAE
                obj = FAE.FAE(iFile=db_path, k=k, sep="\t")
                obj.startMine()
                df_res = obj.getPatternsAsDataFrame()
                if df_res is not None and not df_res.empty:
                    for _, row in df_res.iterrows():
                        patterns.append({"pattern": str(row["Patterns"]), "support": int(row["Support"])})

            elif algorithm == "maxfpgrowth":
                from PAMI.frequentPattern.maximal import MaxFPGrowth
                obj = MaxFPGrowth.MaxFPGrowth(iFile=db_path, minSup=min_sup, sep="\t")
                obj.startMine()
                df_res = obj.getPatternsAsDataFrame()
                if df_res is not None and not df_res.empty:
                    for _, row in df_res.iterrows():
                        patterns.append({"pattern": str(row["Patterns"]), "support": int(row["Support"])})
            else:
                return {"status": "error", "message": f"Unknown algorithm: {algorithm}"}

        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

        # 6. 保存與儲存結果
        result_id = session.next_result_id("mine_patterns")
        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "patterns": patterns[:20]
        }

        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = patterns

        session.store_result(
            tool_name="mine_patterns_tool",
            args={
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "min_sup": min_sup,
                "k": k
            },
            summary=summary
        )

        return summary

    return [mine_patterns_tool]
