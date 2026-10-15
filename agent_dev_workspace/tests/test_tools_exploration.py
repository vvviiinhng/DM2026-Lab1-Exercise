import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools


def test_cosine_similarity_tool():
    # 1. 建立 Session 與載入 premade load_dataset_tool
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # 2. 載入測試資料集並構建 DTM
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # 3. 取得 cosine_similarity_tool
    exp_tools = make_exploration_tools(session)
    cosine_sim_tool = next(t for t in exp_tools if t.name == "cosine_similarity_tool")

    # Known Answer #16: 測試不同文檔之間的餘弦相似度
    res_0_1 = cosine_sim_tool.invoke({"doc1_idx": 0, "doc2_idx": 1})
    assert res_0_1["similarity"] == pytest.approx(0.9847, abs=1e-3)

    res_0_3 = cosine_sim_tool.invoke({"doc1_idx": 0, "doc2_idx": 3})
    assert res_0_3["similarity"] == pytest.approx(0.0909, abs=1e-3)

    res_0_7 = cosine_sim_tool.invoke({"doc1_idx": 0, "doc2_idx": 7})
    assert res_0_7["similarity"] == 0.0


def test_feature_correlation_matrix_tool():
    # 1. 獨立建立 Session 避免狀態影響
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    exp_tools = make_exploration_tools(session)
    feature_corr_tool = next(t for t in exp_tools if t.name == "feature_correlation_matrix_tool")

    # Known Answer #19: 測試特徵相關性矩陣
    res = feature_corr_tool.invoke({"top_n": 20})

    expected_terms = ["alpha", "gamma", "delta", "beta", "always"]
    assert res["top_terms"] == expected_terms

    corr_matrix = np.array(res["correlation_matrix"])
    # Check alpha vs beta (indices 0 and 3)
    assert corr_matrix[0, 3] == pytest.approx(0.8885, abs=1e-3)
    # Check gamma vs delta (indices 1 and 2)
    assert corr_matrix[1, 2] == pytest.approx(0.8307, abs=1e-3)

    # 驗證圖表已設置於 session.pending_figure
    assert session.pending_figure is not None
