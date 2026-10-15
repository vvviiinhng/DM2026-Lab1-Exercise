import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools as make_patterns_tools


def test_mine_patterns_tool_known_answer_17_and_20():
    session = SessionState()

    # 1. 載入資料集
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 2. 取得 mine_patterns_tool
    patterns_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")

    # ----- 測試 Known Answer 17: FPGrowth with tfidf -----
    res_catA_fp = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert len(res_catA_fp["patterns"]) == 1
    assert "alpha" in res_catA_fp["patterns"][0]["pattern"]
    assert res_catA_fp["patterns"][0]["support"] == 3

    res_catB_fp = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert len(res_catB_fp["patterns"]) == 1
    assert "gamma" in res_catB_fp["patterns"][0]["pattern"]
    assert res_catB_fp["patterns"][0]["support"] == 4

    # ----- 測試 Zero Terms Case (variance filtering on sample fixture) -----
    res_zero = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "variance",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_zero["patterns"] == []
    assert "no terms survived filtering" in res_zero["note"]

    # ----- 測試 Known Answer 20: topk and maxfpgrowth -----
    res_topk = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert len(res_topk["patterns"]) == 1
    assert "alpha" in res_topk["patterns"][0]["pattern"]
    assert res_topk["patterns"][0]["support"] == 3

    res_maxfp = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    # MaxFPGrowth 對單一項目會傳回空列表，這為正常行為
    assert res_maxfp["patterns"] == []
