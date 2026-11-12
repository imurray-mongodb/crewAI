import json
from unittest.mock import MagicMock, patch

from crewai_tools import MongoDBVectorSearchConfig, MongoDBVectorSearchTool
import pytest


@pytest.fixture
def mongodb_vector_search_tool():
    tool = MongoDBVectorSearchTool(
        connection_string="foo", database_name="bar", collection_name="test"
    )
    tool._embed_texts = lambda x: [[0.1]]
    yield tool


def test_successful_query_execution(mongodb_vector_search_tool):
    with patch.object(mongodb_vector_search_tool._coll, "aggregate") as mock_aggregate:
        mock_aggregate.return_value = [dict(text="foo", score=0.1, _id=1)]

        results = json.loads(mongodb_vector_search_tool._run(query="sandwiches"))

        assert len(results) == 1
        assert results[0]["text"] == "foo"
        assert results[0]["_id"] == 1


def test_provide_config():
    query_config = MongoDBVectorSearchConfig(limit=10)
    tool = MongoDBVectorSearchTool(
        connection_string="foo",
        database_name="bar",
        collection_name="test",
        query_config=query_config,
        vector_index_name="foo",
        embedding_model="bar",
    )
    tool._embed_texts = lambda x: [[0.1]]
    with patch.object(tool._coll, "aggregate") as mock_aggregate:
        mock_aggregate.return_value = [dict(text="foo", score=0.1, _id=1)]

        tool._run(query="sandwiches")
        assert mock_aggregate.mock_calls[-1].args[0][0]["$vectorSearch"]["limit"] == 10

        mock_aggregate.return_value = [dict(text="foo", score=0.1, _id=1)]


def test_cleanup_on_deletion(mongodb_vector_search_tool):
    with patch.object(mongodb_vector_search_tool, "_client") as mock_client:
        mongodb_vector_search_tool.__del__()

        mock_client.close.assert_called_once()


def test_create_search_index(mongodb_vector_search_tool):
    with patch(
        "crewai_tools.tools.mongodb_vector_search_tool.vector_search.create_vector_search_index"
    ) as mock_create_search_index:
        mongodb_vector_search_tool.create_vector_search_index(dimensions=10)
        kwargs = mock_create_search_index.mock_calls[0].kwargs
        assert kwargs["dimensions"] == 10
        assert kwargs["similarity"] == "cosine"


def test_add_texts(mongodb_vector_search_tool):
    with patch.object(mongodb_vector_search_tool._coll, "bulk_write") as bulk_write:
        mongodb_vector_search_tool.add_texts(["foo"])
        args = bulk_write.mock_calls[0].args
        assert "ReplaceOne" in str(args[0][0])
        assert "foo" in str(args[0][0])


@pytest.fixture
def query_tool(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    tool = MongoDBVectorSearchTool(
        connection_string="foo",
        database_name="bar",
        collection_name="test",
        allowed_actions=["find", "count", "aggregate", "sample_schema"],
    )
    yield tool


def test_query_actions_disabled_by_default(mongodb_vector_search_tool):
    result = mongodb_vector_search_tool._run(action="find", filter={})
    assert result == "Error: action 'find' is not enabled on this tool."


def test_query_actions_do_not_require_embedding_credentials(query_tool):
    assert query_tool._openai_client is None


def test_vector_search_still_requires_query(mongodb_vector_search_tool):
    assert "query" in mongodb_vector_search_tool._run(action="vector_search")


def test_find_caps_limit_and_applies_sort(query_tool):
    cursor = MagicMock()
    cursor.sort.return_value = cursor
    cursor.limit.return_value = [{"_id": 1}]
    with patch.object(query_tool._coll, "find", return_value=cursor) as find:
        out = json.loads(
            query_tool._run(
                action="find", filter={"a": 1}, sort={"a": -1}, limit=10_000
            )
        )
    assert out == [{"_id": 1}]
    assert find.call_args.args[0] == {"a": 1}
    assert find.call_args.kwargs["max_time_ms"] == query_tool.max_time_ms
    cursor.sort.assert_called_once_with([("a", -1)])
    cursor.limit.assert_called_once_with(query_tool.max_limit)


def test_count(query_tool):
    with patch.object(query_tool._coll, "count_documents", return_value=3):
        assert json.loads(query_tool._run(action="count")) == {"count": 3}


def test_aggregate_appends_limit(query_tool):
    with patch.object(query_tool._coll, "aggregate", return_value=[{"x": 1}]) as agg:
        out = json.loads(
            query_tool._run(action="aggregate", pipeline=[{"$match": {}}], limit=5)
        )
    assert out == [{"x": 1}]
    assert agg.call_args.args[0] == [{"$match": {}}, {"$limit": 5}]


@pytest.mark.parametrize("stage", ["$out", "$merge"])
def test_aggregate_rejects_write_stages(query_tool, stage):
    with patch.object(query_tool._coll, "aggregate") as agg:
        result = query_tool._run(action="aggregate", pipeline=[{stage: "x"}])
    assert result.startswith("Error:") and stage in result
    agg.assert_not_called()


def test_rejects_nested_forbidden_operators(query_tool):
    nested = [{"$unionWith": {"coll": "o", "pipeline": [{"$merge": {"into": "x"}}]}}]
    assert "$merge" in query_tool._run(action="aggregate", pipeline=nested)
    assert "$where" in query_tool._run(action="find", filter={"$where": "1"})


def test_sample_schema(query_tool):
    docs = [{"_id": 1, "n": "a", "m": {"k": 1}}, {"_id": 2, "n": None, "m": {"k": "x"}}]
    with patch.object(query_tool._coll, "aggregate", return_value=docs):
        out = json.loads(query_tool._run(action="sample_schema"))
    assert out["sampled_documents"] == 2
    assert out["fields"]["n"] == ["NoneType", "str"]
    assert out["fields"]["m.k"] == ["int", "str"]


def test_query_action_errors_are_reported(query_tool):
    with patch.object(query_tool._coll, "count_documents", side_effect=RuntimeError("boom")):
        assert query_tool._run(action="count") == "Error: RuntimeError: boom"
