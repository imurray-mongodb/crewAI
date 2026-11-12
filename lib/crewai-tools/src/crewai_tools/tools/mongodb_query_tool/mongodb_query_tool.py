"""Read-only, general-purpose MongoDB query tool for CrewAI agents."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import logging
import os
from typing import Any, Literal

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field, PrivateAttr


try:
    import pymongo  # noqa: F401

    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

logger = logging.getLogger(__name__)

# Operators that can write data or execute server-side JavaScript. The tool is
# read-only, so any query or pipeline containing one of these is rejected.
FORBIDDEN_OPERATORS: frozenset[str] = frozenset(
    {"$out", "$merge", "$function", "$accumulator", "$where"}
)

_MAX_SCHEMA_DEPTH = 3


class MongoDBQueryToolSchema(BaseModel):
    """Input for MongoDBQueryTool."""

    action: Literal["find", "aggregate", "count", "list_collections", "sample_schema"] = Field(
        ...,
        description=(
            "Operation to run: 'find' (query documents), 'aggregate' (run a "
            "pipeline), 'count' (count matching documents), 'list_collections' "
            "(list collection names) or 'sample_schema' (infer field names and "
            "types from a sample of documents)."
        ),
    )
    collection: str | None = Field(
        default=None,
        description="Collection to query. Not needed for 'list_collections'.",
    )
    filter: dict[str, Any] | None = Field(  # noqa: A003
        default=None,
        description="MongoDB query filter for 'find' and 'count', e.g. {\"status\": \"active\"}.",
    )
    projection: dict[str, Any] | None = Field(
        default=None,
        description="Fields to include or exclude for 'find', e.g. {\"name\": 1}.",
    )
    sort: dict[str, int] | None = Field(
        default=None,
        description="Sort for 'find'; 1 ascending, -1 descending, e.g. {\"created\": -1}.",
    )
    pipeline: list[dict[str, Any]] | None = Field(
        default=None,
        description="Aggregation pipeline stages for 'aggregate'. Write stages ($out, $merge) are not allowed.",
    )
    limit: int | None = Field(
        default=None,
        description="Maximum documents to return (capped by the tool's max_limit).",
    )


def find_forbidden_operator(value: Any) -> str | None:
    """Return the first forbidden operator found anywhere in ``value``, if any."""
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in FORBIDDEN_OPERATORS:
                return key
            found = find_forbidden_operator(nested)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = find_forbidden_operator(item)
            if found:
                return found
    return None


def _describe_schema(doc: dict[str, Any], prefix: str, depth: int, out: dict[str, set[str]]) -> None:
    for key, value in doc.items():
        path = f"{prefix}{key}"
        out.setdefault(path, set()).add(type(value).__name__)
        if isinstance(value, dict) and depth < _MAX_SCHEMA_DEPTH:
            _describe_schema(value, f"{path}.", depth + 1, out)


class MongoDBQueryTool(BaseTool):
    """Read-only queries against a MongoDB database (find, aggregate, count, schema)."""

    name: str = "MongoDB Query Tool"
    description: str = (
        "Query a MongoDB database (read-only). Actions: find documents, run an "
        "aggregation pipeline, count documents, list collections, or sample a "
        "collection's schema. Call 'list_collections' and 'sample_schema' first "
        "when you don't know the data's shape."
    )
    args_schema: type[BaseModel] = MongoDBQueryToolSchema

    connection_string: str | None = Field(
        default=None,
        description="MongoDB connection string. Defaults to the MONGODB_URI environment variable.",
    )
    database_name: str = Field(..., description="Name of the MongoDB database to query.")
    default_limit: int = Field(default=20, ge=1, description="Documents returned when no limit is given.")
    max_limit: int = Field(default=100, ge=1, description="Hard cap on documents returned per call.")
    max_time_ms: int = Field(default=30_000, ge=1, description="Server-side time limit per operation, in ms.")
    sample_size: int = Field(default=20, ge=1, description="Documents sampled by 'sample_schema'.")
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="MONGODB_URI",
                description="MongoDB connection string (used when connection_string is not passed)",
                required=False,
            ),
        ]
    )
    package_dependencies: list[str] = Field(default_factory=lambda: ["pymongo"])

    _client: Any = PrivateAttr(default=None)
    _db: Any = PrivateAttr(default=None)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not MONGODB_AVAILABLE:
            raise ImportError(
                "pymongo is required for MongoDBQueryTool. "
                "Install it with `uv add crewai-tools --extra mongodb`."
            )
        uri = self.connection_string or os.environ.get("MONGODB_URI")
        if not uri:
            raise ValueError(
                "Provide connection_string or set the MONGODB_URI environment variable."
            )

        from pymongo import MongoClient
        from pymongo.driver_info import DriverInfo

        try:
            tools_version = version("crewai-tools")
        except PackageNotFoundError:
            tools_version = "unknown"
        self._client = MongoClient(
            uri, driver=DriverInfo(name="CrewAI", version=tools_version)
        )
        self._db = self._client[self.database_name]

    def _clamp_limit(self, limit: int | None) -> int:
        return max(1, min(limit or self.default_limit, self.max_limit))

    def _run(self, **kwargs: Any) -> str:
        from bson import json_util

        args = MongoDBQueryToolSchema(**kwargs)
        try:
            result = self._dispatch(args)
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            logger.error("MongoDBQueryTool failed: %s", e)
            return f"Error: {type(e).__name__}: {e}"
        return json_util.dumps(result)

    def _dispatch(self, args: MongoDBQueryToolSchema) -> Any:
        if args.action == "list_collections":
            return sorted(self._db.list_collection_names())

        if not args.collection:
            raise ValueError(f"'collection' is required for action '{args.action}'.")
        coll = self._db[args.collection]

        if args.action == "find":
            self._reject_forbidden(args.filter, args.projection)
            cursor = coll.find(
                args.filter or {}, args.projection, max_time_ms=self.max_time_ms
            )
            if args.sort:
                cursor = cursor.sort(list(args.sort.items()))
            return list(cursor.limit(self._clamp_limit(args.limit)))

        if args.action == "count":
            self._reject_forbidden(args.filter)
            return {
                "count": coll.count_documents(
                    args.filter or {}, maxTimeMS=self.max_time_ms
                )
            }

        if args.action == "aggregate":
            if not args.pipeline:
                raise ValueError("'pipeline' is required for action 'aggregate'.")
            self._reject_forbidden(args.pipeline)
            pipeline = [*args.pipeline, {"$limit": self._clamp_limit(args.limit)}]
            return list(coll.aggregate(pipeline, maxTimeMS=self.max_time_ms))

        # sample_schema
        docs = coll.aggregate(
            [{"$sample": {"size": self.sample_size}}], maxTimeMS=self.max_time_ms
        )
        fields: dict[str, set[str]] = {}
        count = 0
        for doc in docs:
            count += 1
            _describe_schema(doc, "", 1, fields)
        return {
            "sampled_documents": count,
            "fields": {path: sorted(types) for path, types in sorted(fields.items())},
        }

    @staticmethod
    def _reject_forbidden(*values: Any) -> None:
        for value in values:
            found = find_forbidden_operator(value)
            if found:
                raise ValueError(
                    f"Operator '{found}' is not allowed: MongoDBQueryTool is read-only."
                )

    def __del__(self) -> None:
        client = getattr(self, "_client", None)
        if client is not None:
            try:
                client.close()
            except Exception as e:
                logger.error("Error closing MongoDB client: %s", e)
