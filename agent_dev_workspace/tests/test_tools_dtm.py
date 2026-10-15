import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool_known_answer_4():
    # 建立獨立的 SessionState
    session = SessionState()

    # 載入 premade_tools 中的 load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # 載入 sample_fixture.csv
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 取得要測試的 build_dtm_tool
    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")

    # 執行 build_dtm_tool
    result = build_dtm_tool.invoke({})

    # 驗證傳回值與 Known Answer 4 是否相符
    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]
    
    # 檢查傳回的 summary 數值與 session 屬性
    assert result["non_zero"] == 21
    assert result["total_elements"] == 40
    assert pytest.approx(result["sparsity_pct"], abs=1e-3) == 47.5

    # 檢查 artifact 是否包含 count_vectorizer
    assert "count_vectorizer" in session.artifacts
