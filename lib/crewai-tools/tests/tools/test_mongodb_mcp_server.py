from crewai.mcp import MCPServerStdio
from crewai_tools import mongodb_mcp_server
import pytest


@pytest.fixture(autouse=True)
def _no_ambient_uri(monkeypatch):
    monkeypatch.delenv("MONGODB_URI", raising=False)


def test_defaults_are_read_only_and_use_npx():
    server = mongodb_mcp_server("mongodb://localhost:27017")
    assert isinstance(server, MCPServerStdio)
    assert server.command == "npx"
    assert server.args == ["-y", "mongodb-mcp-server@latest", "--readOnly"]


def test_connection_string_goes_in_env_not_args():
    uri = "mongodb://user:secret@localhost:27017"
    server = mongodb_mcp_server(uri)
    assert server.env == {"MDB_MCP_CONNECTION_STRING": uri}
    assert all("secret" not in a for a in server.args)


def test_connection_string_falls_back_to_env_var(monkeypatch):
    monkeypatch.setenv("MONGODB_URI", "mongodb://envhost:27017")
    assert mongodb_mcp_server().env == {
        "MDB_MCP_CONNECTION_STRING": "mongodb://envhost:27017"
    }


def test_no_connection_string_leaves_env_unset():
    assert mongodb_mcp_server().env is None


def test_write_mode_requires_explicit_opt_in():
    server = mongodb_mcp_server("mongodb://localhost", read_only=False)
    assert "--readOnly" not in server.args


def test_version_pin_and_disabled_tools():
    server = mongodb_mcp_server(
        "mongodb://localhost", version="1.2.3", disabled_tools=["atlas", "delete"]
    )
    assert "mongodb-mcp-server@1.2.3" in server.args
    assert server.env["MDB_MCP_DISABLED_TOOLS"] == "atlas,delete"


def test_extra_env_overrides_helper_values():
    server = mongodb_mcp_server(
        "mongodb://a", env={"MDB_MCP_CONNECTION_STRING": "mongodb://b", "X": "1"}
    )
    assert server.env == {"MDB_MCP_CONNECTION_STRING": "mongodb://b", "X": "1"}


def test_allowed_tool_names_builds_filter():
    server = mongodb_mcp_server("mongodb://localhost", allowed_tool_names=["find"])
    assert server.tool_filter is not None
    assert server.tool_filter({"name": "find"}) is True
    assert server.tool_filter({"name": "drop-database"}) is False


def test_no_filter_by_default():
    assert mongodb_mcp_server("mongodb://localhost").tool_filter is None
