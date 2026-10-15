import pytest
import pandas as pd
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools as make_data_tools


def test_list_files_tool():
    session = SessionState()
    tools = make_data_tools(session)
    list_files_tool = next(t for t in tools if t.name == "list_files_tool")

    # Known Answer #11: 檢查 subdirectory="newdataset"
    res = list_files_tool.invoke({"subdirectory": "newdataset"})
    assert res["directories"] == []
    assert res["files"] == ["Reddit-stock-sentiment.csv"]


def test_check_missing_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_data_tools(session)
    check_missing_tool = next(t for t in tools if t.name == "check_missing_tool")

    # Known Answer #2: exactly 1 row has missing/empty text
    res = check_missing_tool.invoke({})
    assert res["missing_text_count"] == 1


def test_check_duplicates_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_data_tools(session)
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")

    # Known Answer #3: exactly 1 duplicate (rows 5 and 6 are exact copies)
    res_check = check_duplicates_tool.invoke({"drop": False})
    assert res_check["duplicate_count"] == 2  # duplicate mask with keep=False finds both rows

    # Dropping with keep=False removes both copies, leaving 6 rows
    res_drop = check_duplicates_tool.invoke({"drop": True})
    assert res_drop["rows_remaining"] == 6


def test_sample_data_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_data_tools(session)
    sample_data_tool = next(t for t in tools if t.name == "sample_data_tool")

    # Known Answer #13: n=4, random_state=42 returns indices [1, 5, 0, 7]
    res = sample_data_tool.invoke({"n": 4, "random_state": 42})
    assert res["sampled_indices"] == [1, 5, 0, 7]


def test_tokenize_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_data_tools(session)
    tokenize_tool = next(t for t in tools if t.name == "tokenize_tool")

    # Known Answer #10: NLTK tokenize to lowercased unigrams
    res = tokenize_tool.invoke({})
    tokens = session.dataframe["unigrams"].tolist()

    assert tokens[0] == ["always", "alpha", "alpha", "alpha", "beta"]
    assert tokens[3] == ["always", "gamma", "gamma", "gamma", "delta"]
    assert tokens[7] == []  # empty text


def test_describe_data_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_data_tools(session)
    describe_data_tool = next(t for t in tools if t.name == "describe_data_tool")

    # Known Answer #18: text length stats
    res = describe_data_tool.invoke({})
    overall = res["overall_stats"]

    assert overall["count"] == 8
    assert overall["mean"] == pytest.approx(19.875, abs=1e-3)
    assert overall["min"] == 0
    assert overall["max"] == 30

    per_cat = res["per_category_stats"]
    assert per_cat["catA"]["mean"] == pytest.approx(17.25, abs=1e-3)
    assert per_cat["catB"]["mean"] == pytest.approx(22.5, abs=1e-3)

    assert session.pending_figure is not None


def test_inspect_data_tool():
    # 針對沒有官方 Known Answer 的 inspect_data_tool 做基礎邏輯測試
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_data_tools(session)
    inspect_data_tool = next(t for t in tools if t.name == "inspect_data_tool")

    res = inspect_data_tool.invoke({"n_rows": 3})
    assert res["shape"] == [8, 3]  # text, category, category_name
    assert len(res["head"]) == 3
