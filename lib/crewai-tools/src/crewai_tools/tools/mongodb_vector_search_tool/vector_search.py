from collections.abc import Iterable
from importlib.metadata import version
from logging import getLogger
import os
from typing import Any, Literal

from crewai.tools import BaseTool, EnvVar
from openai import AzureOpenAI, Client
from pydantic import BaseModel, Field

from crewai_tools.tools.mongodb_vector_search_tool.utils import (
    create_vector_search_index,
    find_forbidden_operator,
)


try:
    import pymongo  # noqa: F401

    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

logger = getLogger(__name__)


class MongoDBVectorSearchConfig(BaseModel):
    """Configuration for MongoDB vector search queries."""

    limit: int | None = Field(default=4, description="number of documents to return.")
    pre_filter: dict[str, Any] | None = Field(
        default=None,
        description="List of MQL match expressions comparing an indexed field",
    )
    post_filter_pipeline: list[dict[str, Any]] | None = Field(
        default=None,
        description="Pipeline of MongoDB aggregation stages to filter/process results after $vectorSearch.",
    )
    oversampling_factor: int = Field(
        default=10,
        description="Multiple of limit used when generating number of candidates at each step in the HNSW Vector Search",
    )
    include_embeddings: bool = Field(
        default=False,
        description="Whether to include the embedding vector of each result in metadata.",
    )


MongoDBAction = Literal["vector_search", "find", "count", "aggregate", "sample_schema"]


class MongoDBToolSchema(BaseModel):
    """Input for MongoDBTool."""

    query: str | None = Field(
        default=None,
        description="The query to search retrieve relevant information from the MongoDB database. Pass only the query, not the question. Required for the 'vector_search' action.",
    )
    action: MongoDBAction = Field(
        default="vector_search",
        description=(
            "'vector_search' (default) semantically searches with `query`. "
            "'find', 'count', 'aggregate' and 'sample_schema' query the collection "
            "directly and are only available when enabled on the tool."
        ),
    )
    filter: dict[str, Any] | None = Field(  # noqa: A003
        default=None,
        description="MongoDB query filter for 'find' and 'count'.",
    )
    projection: dict[str, Any] | None = Field(
        default=None, description="Fields to include or exclude for 'find'."
    )
    sort: dict[str, int] | None = Field(
        default=None,
        description="Sort for 'find'; 1 ascending, -1 descending.",
    )
    pipeline: list[dict[str, Any]] | None = Field(
        default=None,
        description="Aggregation pipeline for 'aggregate'. Write stages are not allowed.",
    )
    limit: int | None = Field(
        default=None,
        description="Maximum documents to return for 'find' and 'aggregate' (capped by the tool's max_limit).",
    )


class MongoDBVectorSearchTool(BaseTool):
    """Tool to perfrom a vector search the MongoDB database."""

    name: str = "MongoDBVectorSearchTool"
    description: str = "A tool to perfrom a vector search on a MongoDB database for relevant information on internal documents."

    args_schema: type[BaseModel] = MongoDBToolSchema
    query_config: MongoDBVectorSearchConfig | None = Field(
        default=None, description="MongoDB Vector Search query configuration"
    )
    embedding_model: str = Field(
        default="text-embedding-3-large",
        description="Text OpenAI embedding model to use",
    )
    vector_index_name: str = Field(
        default="vector_index", description="Name of the Atlas Search vector index"
    )
    text_key: str = Field(
        default="text",
        description="MongoDB field that will contain the text for each document",
    )
    embedding_key: str = Field(
        default="embedding",
        description="Field that will contain the embedding for each document",
    )
    database_name: str = Field(..., description="The name of the MongoDB database")
    collection_name: str = Field(..., description="The name of the MongoDB collection")
    connection_string: str = Field(
        ...,
        description="The connection string of the MongoDB cluster",
    )
    dimensions: int = Field(
        default=1536,
        description="Number of dimensions in the embedding vector",
    )
    allowed_actions: list[MongoDBAction] = Field(
        default_factory=lambda: ["vector_search"],
        description=(
            "Actions the agent may run. Defaults to vector search only. Add "
            "'find', 'count', 'aggregate' or 'sample_schema' for read-only "
            "queries on the configured collection."
        ),
    )
    max_limit: int = Field(
        default=100, ge=1, description="Hard cap on documents returned by 'find' and 'aggregate'."
    )
    max_time_ms: int = Field(
        default=30_000, ge=1, description="Server-side time limit for the read actions, in ms."
    )
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="BROWSERBASE_API_KEY",
                description="API key for Browserbase services",
                required=False,
            ),
            EnvVar(
                name="BROWSERBASE_PROJECT_ID",
                description="Project ID for Browserbase services",
                required=False,
            ),
        ]
    )
    package_dependencies: list[str] = Field(default_factory=lambda: ["pymongo"])

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not MONGODB_AVAILABLE:
            import click

            if click.confirm(
                "You are missing the 'mongodb' crewai tool. Would you like to install it?"
            ):
                import subprocess

                subprocess.run(["uv", "add", "pymongo"], check=True)  # noqa: S607

            else:
                raise ImportError("You are missing the 'mongodb' crewai tool.")

        self._openai_client: AzureOpenAI | Client | None = None
        if "vector_search" in self.allowed_actions:
            if "AZURE_OPENAI_ENDPOINT" in os.environ:
                self._openai_client = AzureOpenAI()
            elif "OPENAI_API_KEY" in os.environ:
                self._openai_client = Client()
            else:
                raise ValueError(
                    "OPENAI_API_KEY environment variable is required for MongoDBVectorSearchTool and it is mandatory to use the tool."
                )

        from pymongo import MongoClient
        from pymongo.driver_info import DriverInfo

        self._client: MongoClient[dict[str, Any]] = MongoClient(
            self.connection_string,
            driver=DriverInfo(name="CrewAI", version=version("crewai-tools")),
        )
        self._coll = self._client[self.database_name][self.collection_name]

    def create_vector_search_index(
        self,
        *,
        dimensions: int,
        relevance_score_fn: str = "cosine",
        auto_index_timeout: int = 15,
    ) -> None:
        """Convenience function to create a vector search index.

        Args:
            dimensions: Number of dimensions in embedding.  If the value is set and
                the index does not exist, an index will be created.
            relevance_score_fn: The similarity score used for the index
                Currently supported: 'euclidean', 'cosine', and 'dotProduct'
            auto_index_timeout: Timeout in seconds to wait for an auto-created index
               to be ready.
        """
        create_vector_search_index(
            collection=self._coll,
            index_name=self.vector_index_name,
            dimensions=dimensions,
            path=self.embedding_key,
            similarity=relevance_score_fn,
            wait_until_complete=auto_index_timeout,
        )

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        batch_size: int = 100,
        **kwargs: Any,
    ) -> list[str]:
        """Add texts, create embeddings, and add to the Collection and index.

        Important notes on ids:
            - If _id or id is a key in the metadatas dicts, one must
                pop them and provide as separate list.
            - They must be unique.
            - If they are not provided, the VectorStore will create unique ones,
                stored as bson.ObjectIds internally, and strings in Langchain.
                These will appear in Document.metadata with key, '_id'.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of unique ids that will be used as index in VectorStore.
                See note on ids.
            batch_size: Number of documents to insert at a time.
                Tuning this may help with performance and sidestep MongoDB limits.

        Returns:
            List of ids added to the vectorstore.
        """
        from bson import ObjectId

        _metadatas = metadatas or [{} for _ in texts]
        ids = [str(ObjectId()) for _ in range(len(list(texts)))]

        result_ids = []
        texts_batch = []
        metadatas_batch = []
        size = 0
        i = 0
        for j, (text, metadata) in enumerate(zip(texts, _metadatas, strict=False)):
            size += len(text) + len(metadata)
            texts_batch.append(text)
            metadatas_batch.append(metadata)
            if (j + 1) % batch_size == 0 or size >= 47_000_000:
                batch_res = self._bulk_embed_and_insert_texts(
                    texts_batch, metadatas_batch, ids[i : j + 1]
                )
                result_ids.extend(batch_res)
                texts_batch = []
                metadatas_batch = []
                size = 0
                i = j + 1
        if texts_batch:
            batch_res = self._bulk_embed_and_insert_texts(
                texts_batch, metadatas_batch, ids[i : j + 1]
            )
            result_ids.extend(batch_res)
        return result_ids

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self._openai_client is None:
            raise ValueError(
                "Embeddings are unavailable: 'vector_search' is not in allowed_actions."
            )
        return [
            i.embedding
            for i in self._openai_client.embeddings.create(
                input=texts,
                model=self.embedding_model,
                dimensions=self.dimensions,
            ).data
        ]

    def _bulk_embed_and_insert_texts(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]],
        ids: list[str],
    ) -> list[str]:
        """Bulk insert single batch of texts, embeddings, and ids."""
        from bson import ObjectId
        from pymongo.operations import ReplaceOne

        if not texts:
            return []
        # Compute embedding vectors
        embeddings = self._embed_texts(texts)
        docs = [
            {
                "_id": ObjectId(i),
                self.text_key: t,
                self.embedding_key: embedding,
                **m,
            }
            for i, t, m, embedding in zip(
                ids, texts, metadatas, embeddings, strict=False
            )
        ]
        operations = [ReplaceOne({"_id": doc["_id"]}, doc, upsert=True) for doc in docs]
        result = self._coll.bulk_write(operations)
        if result.upserted_ids is None:
            raise ValueError("No documents were inserted.")
        return [str(_id) for _id in result.upserted_ids.values()]

    def _run(
        self,
        query: str | None = None,
        action: MongoDBAction = "vector_search",
        filter: dict[str, Any] | None = None,  # noqa: A002
        projection: dict[str, Any] | None = None,
        sort: dict[str, int] | None = None,
        pipeline: list[dict[str, Any]] | None = None,
        limit: int | None = None,
    ) -> str:
        if action != "vector_search":
            return self._run_query_action(
                action,
                filter=filter,
                projection=projection,
                sort=sort,
                pipeline=pipeline,
                limit=limit,
            )
        if "vector_search" not in self.allowed_actions:
            return "Error: action 'vector_search' is not enabled on this tool."
        if not query:
            return "Error: 'query' is required for the 'vector_search' action."
        return self._vector_search(query)

    def _run_query_action(
        self,
        action: str,
        *,
        filter: dict[str, Any] | None,  # noqa: A002
        projection: dict[str, Any] | None,
        sort: dict[str, int] | None,
        pipeline: list[dict[str, Any]] | None,
        limit: int | None,
    ) -> str:
        """Read-only find/count/aggregate/sample_schema on the configured collection."""
        from bson import json_util

        if action not in self.allowed_actions:
            return f"Error: action '{action}' is not enabled on this tool."
        capped = max(1, min(limit or 20, self.max_limit))
        try:
            if action == "find":
                self._reject_forbidden(filter, projection)
                cursor = self._coll.find(
                    filter or {}, projection, max_time_ms=self.max_time_ms
                )
                if sort:
                    cursor = cursor.sort(list(sort.items()))
                result: Any = list(cursor.limit(capped))
            elif action == "count":
                self._reject_forbidden(filter)
                result = {
                    "count": self._coll.count_documents(
                        filter or {}, maxTimeMS=self.max_time_ms
                    )
                }
            elif action == "aggregate":
                if not pipeline:
                    return "Error: 'pipeline' is required for the 'aggregate' action."
                self._reject_forbidden(pipeline)
                result = list(
                    self._coll.aggregate(
                        [*pipeline, {"$limit": capped}], maxTimeMS=self.max_time_ms
                    )
                )
            else:  # sample_schema
                fields: dict[str, set[str]] = {}
                sampled = 0
                for doc in self._coll.aggregate(
                    [{"$sample": {"size": 20}}], maxTimeMS=self.max_time_ms
                ):
                    sampled += 1
                    self._describe_fields(doc, "", 1, fields)
                result = {
                    "sampled_documents": sampled,
                    "fields": {p: sorted(t) for p, t in sorted(fields.items())},
                }
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            logger.error(f"Error: {e}")
            return f"Error: {type(e).__name__}: {e}"
        return json_util.dumps(result)

    @staticmethod
    def _reject_forbidden(*values: Any) -> None:
        for value in values:
            found = find_forbidden_operator(value)
            if found:
                raise ValueError(
                    f"Operator '{found}' is not allowed: query actions are read-only."
                )

    @classmethod
    def _describe_fields(
        cls, doc: dict[str, Any], prefix: str, depth: int, out: dict[str, set[str]]
    ) -> None:
        for key, value in doc.items():
            path = f"{prefix}{key}"
            out.setdefault(path, set()).add(type(value).__name__)
            if isinstance(value, dict) and depth < 3:
                cls._describe_fields(value, f"{path}.", depth + 1, out)

    def _vector_search(self, query: str) -> str:
        from bson import json_util

        try:
            query_config = self.query_config or MongoDBVectorSearchConfig()
            limit = query_config.limit
            oversampling_factor = query_config.oversampling_factor
            pre_filter = query_config.pre_filter
            include_embeddings = query_config.include_embeddings
            post_filter_pipeline = query_config.post_filter_pipeline

            query_vector = self._embed_texts([query])[0]

            # Atlas Vector Search, potentially with filter
            stage = {
                "index": self.vector_index_name,
                "path": self.embedding_key,
                "queryVector": query_vector,
                "numCandidates": limit * oversampling_factor,  # type: ignore[operator]
                "limit": limit,
            }
            if pre_filter:
                stage["filter"] = pre_filter

            pipeline = [
                {"$vectorSearch": stage},
                {"$set": {"score": {"$meta": "vectorSearchScore"}}},
            ]

            if not include_embeddings:
                pipeline.append({"$project": {self.embedding_key: 0}})

            # Post-processing
            if post_filter_pipeline is not None:
                pipeline.extend(post_filter_pipeline)

            # Execution
            cursor = self._coll.aggregate(pipeline)  # type: ignore[arg-type]
            docs = []

            for doc in cursor:
                docs.append(doc)  # noqa: PERF402
            return json_util.dumps(docs)
        except Exception as e:
            logger.error(f"Error: {e}")
            return ""

    def __del__(self) -> None:
        """Cleanup clients on deletion."""
        try:
            client: Any = getattr(self, "_client", None)
            if client is not None:
                client.close()
        except Exception as e:
            logger.error(f"Error: {e}")

        try:
            openai_client: Any = getattr(self, "_openai_client", None)
            if openai_client is not None:
                openai_client.close()
        except Exception as e:
            logger.error(f"Error: {e}")
