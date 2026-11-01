"""MCP resource implementations for Aspen Plus data access."""

import logging
from typing import Any

from mcp.server import Server
from mcp.types import Resource, TextContent

from .aspen_wrapper import AspenPlusWrapper

logger = logging.getLogger(__name__)


def register_resources(server: Server, aspen: AspenPlusWrapper):
    """Register all Aspen Plus resources with the MCP server."""

    @server.list_resources()
    async def list_resources() -> list[Resource]:
        """List available resources."""
        return [
            Resource(
                uri="simulation://status",
                name="Simulation Status",
                mimeType="text/plain",
                description="Current status of the Aspen Plus simulation"
            )
        ]

    @server.read_resource()
    async def read_resource(uri: str) -> str:
        """Read a resource by URI."""
        if uri == "simulation://status":
            if aspen.aspen:
                return "Simulation loaded and ready"
            else:
                return "No simulation loaded"
        else:
            raise ValueError(f"Unknown resource: {uri}")
