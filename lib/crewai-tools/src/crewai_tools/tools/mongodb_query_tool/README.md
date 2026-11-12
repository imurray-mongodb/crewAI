# MongoDBQueryTool

Read-only, general-purpose queries against any MongoDB deployment. For semantic
search over embeddings use `MongoDBVectorSearchTool` instead.

## Install

```shell
uv add crewai-tools --extra mongodb
```

## Usage

```python
from crewai import Agent
from crewai_tools import MongoDBQueryTool

tool = MongoDBQueryTool(
    connection_string="mongodb+srv://...",  # or set MONGODB_URI
    database_name="sample_mflix",
)
agent = Agent(role="Analyst", goal="Answer questions about the data", backstory="...", tools=[tool])
```

## Actions

| action | needs | returns |
|---|---|---|
| `list_collections` | - | collection names |
| `sample_schema` | `collection` | field paths and observed types |
| `find` | `collection`, optional `filter`/`projection`/`sort`/`limit` | documents |
| `count` | `collection`, optional `filter` | `{"count": n}` |
| `aggregate` | `collection`, `pipeline` | documents |

## Safety

- Read-only: `$out`, `$merge`, `$function`, `$accumulator` and `$where` are
  rejected anywhere in a filter, projection or pipeline.
- Results are capped by `max_limit` (default 100); each operation has a
  server-side `max_time_ms` (default 30s).
- Also connect with a database user that only has the `read` role.
