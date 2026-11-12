"""Configuration helper for MongoDB's official MCP server."""

from __future__ import annotations

import os

from crewai.mcp import MCPServerStdio, create_static_tool_filter


def mongodb_mcp_server(
    connection_string: str | None = None,
    *,
    read_only: bool = True,
    allowed_tool_names: list[str] | None = None,
    disabled_tools: list[str] | None = None,
    version: str = "latest",
    env: dict[str, str] | None = None,
) -> MCPServerStdio:
    """Build an ``MCPServerStdio`` config for ``mongodb-mcp-server``.

    The server is launched with ``npx`` (Node.js required) and is **read-only by
    default**. Pass the result to an agent's ``mcps`` list::

        Agent(..., mcps=[mongodb_mcp_server("mongodb://localhost:27017/mydb")])

    Args:
        connection_string: MongoDB connection string. Defaults to the
            ``MONGODB_URI`` environment variable. When neither is set the server
            starts unconnected and the agent must use its ``connect`` tool. It is
            passed to the server through its environment, never on the command
            line, so it does not appear in process listings.
        read_only: Start the server with ``--readOnly`` so write, create and drop
            tools are disabled. Set to ``False`` explicitly to allow writes.
        allowed_tool_names: If given, only these MCP tools are exposed to the agent.
        disabled_tools: Tool names, operation types or categories the server should
            not register (``MDB_MCP_DISABLED_TOOLS``).
        version: ``mongodb-mcp-server`` npm version. Pin one (e.g. ``"1.2.0"``) for
            reproducible runs; ``"latest"`` follows new releases.
        env: Extra environment variables for the server process (for example
            Atlas API credentials). Overrides the variables set by this helper.

    Returns:
        A ``MCPServerStdio`` configuration.
    """
    args = ["-y", f"mongodb-mcp-server@{version}"]
    if read_only:
        args.append("--readOnly")

    server_env: dict[str, str] = {}
    uri = connection_string or os.environ.get("MONGODB_URI")
    if uri:
        server_env["MDB_MCP_CONNECTION_STRING"] = uri
    if disabled_tools:
        server_env["MDB_MCP_DISABLED_TOOLS"] = ",".join(disabled_tools)
    server_env.update(env or {})

    return MCPServerStdio(
        command="npx",
        args=args,
        env=server_env or None,
        tool_filter=(
            create_static_tool_filter(allowed_tool_names=allowed_tool_names)
            if allowed_tool_names
            else None
        ),
    )
