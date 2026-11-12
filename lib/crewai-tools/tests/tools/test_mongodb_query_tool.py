import json
from unittest.mock import MagicMock

from crewai_tools import MongoDBQueryTool
import pytest


@pytest.fixture
def tool():
    t = MongoDBQueryTool(
        connection_string="mongodb://localhost:27017", database_name="db"
    )
    t._db = MagicMock()
    return t


def _run(tool, **kwargs):
    return tool._run(**kwargs)


def test_requires_connection_string(monkeypatch):
    monkeypatch.delenv("MONGODB_URI", raising=False)
    with pytest.raises(ValueError, match="MONGODB_URI"):
        MongoDBQueryTool(database_name="db")


def test_uses_env_connection_string(monkeypatch):
    monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017")
    assert MongoDBQueryTool(database_name="db")._client is not None


def test_list_collections(tool):
    tool._db.list_collection_names.return_value = ["b", "a"]
    assert json.loads(_run(tool, action="list_collections")) == ["a", "b"]


def test_find_applies_filter_sort_and_limit_cap(tool):
    cursor = MagicMock()
    cursor.sort.return_value = cursor
    cursor.limit.return_value = [{"_id": 1, "n": 2}]
    coll = tool._db.__getitem__.return_value
    coll.find.return_value = cursor

    out = json.loads(
        _run(
            tool,
            action="find",
            collection="c",
            filter={"n": 2},
            sort={"n": -1},
            limit=10_000,
        )
    )

    assert out == [{"_id": 1, "n": 2}]
    assert coll.find.call_args.args[0] == {"n": 2}
    assert coll.find.call_args.kwargs["max_time_ms"] == tool.max_time_ms
    cursor.sort.assert_called_once_with([("n", -1)])
    cursor.limit.assert_called_once_with(tool.max_limit)


def test_find_default_limit(tool):
    cursor = MagicMock()
    cursor.limit.return_value = []
    tool._db.__getitem__.return_value.find.return_value = cursor
    _run(tool, action="find", collection="c")
    cursor.limit.assert_called_once_with(tool.default_limit)


def test_count(tool):
    tool._db.__getitem__.return_value.count_documents.return_value = 7
    assert json.loads(_run(tool, action="count", collection="c")) == {"count": 7}


def test_aggregate_appends_limit(tool):
    coll = tool._db.__getitem__.return_value
    coll.aggregate.return_value = [{"_id": "x"}]
    out = json.loads(
        _run(
            tool,
            action="aggregate",
            collection="c",
            pipeline=[{"$match": {"a": 1}}],
            limit=5,
        )
    )
    assert out == [{"_id": "x"}]
    assert coll.aggregate.call_args.args[0] == [{"$match": {"a": 1}}, {"$limit": 5}]


@pytest.mark.parametrize("stage", ["$out", "$merge"])
def test_aggregate_rejects_write_stages(tool, stage):
    result = _run(tool, action="aggregate", collection="c", pipeline=[{stage: "other"}])
    assert result.startswith("Error:") and stage in result
    tool._db.__getitem__.return_value.aggregate.assert_not_called()


def test_rejects_nested_forbidden_operator(tool):
    pipeline = [{"$unionWith": {"coll": "o", "pipeline": [{"$merge": {"into": "x"}}]}}]
    assert "$merge" in _run(tool, action="aggregate", collection="c", pipeline=pipeline)


def test_find_rejects_where_and_function(tool):
    assert "$where" in _run(tool, action="find", collection="c", filter={"$where": "1"})
    assert "$function" in _run(
        tool,
        action="find",
        collection="c",
        filter={"$expr": {"$function": {"body": "x", "args": [], "lang": "js"}}},
    )


def test_collection_required(tool):
    assert "collection" in _run(tool, action="find")


def test_sample_schema(tool):
    tool._db.__getitem__.return_value.aggregate.return_value = [
        {"_id": 1, "name": "a", "meta": {"k": 1}},
        {"_id": 2, "name": None, "meta": {"k": "x"}},
    ]
    out = json.loads(_run(tool, action="sample_schema", collection="c"))
    assert out["sampled_documents"] == 2
    assert out["fields"]["name"] == ["NoneType", "str"]
    assert out["fields"]["meta.k"] == ["int", "str"]


def test_driver_errors_are_reported_not_swallowed(tool):
    tool._db.list_collection_names.side_effect = RuntimeError("boom")
    assert _run(tool, action="list_collections") == "Error: RuntimeError: boom"
