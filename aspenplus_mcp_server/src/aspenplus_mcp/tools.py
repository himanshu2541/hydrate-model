"""MCP tool implementations for Aspen Plus operations."""

import logging
from typing import Any

from mcp.server import Server
from mcp.types import Tool, TextContent

from .aspen_wrapper import AspenPlusWrapper

logger = logging.getLogger(__name__)


def register_tools(server: Server, aspen: AspenPlusWrapper):
    """Register all Aspen Plus tools with the MCP server."""

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available tools."""
        return [
            Tool(
                name="open_simulation",
                description="Open an Aspen Plus simulation file (.bkp or .apw)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "Path to the Aspen Plus simulation file"
                        },
                        "use_enhanced": {
                            "type": "boolean",
                            "description": "Use enhanced interface with block/stream manipulation capabilities (default: false)"
                        }
                    },
                    "required": ["filepath"]
                }
            ),
            Tool(
                name="run_simulation",
                description="Run the current Aspen Plus simulation",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            Tool(
                name="get_value",
                description="Get a value from the simulation using a node path",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Node path (e.g., \\Data\\Streams\\S1\\Output\\TEMP_OUT\\MIXED\\MIXED)"
                        }
                    },
                    "required": ["path"]
                }
            ),
            Tool(
                name="set_value",
                description="Set a value in the simulation using a node path",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Node path to the parameter"
                        },
                        "value": {
                            "description": "Value to set (can be number or string)"
                        }
                    },
                    "required": ["path", "value"]
                }
            ),
            Tool(
                name="close_simulation",
                description="Close the current Aspen Plus simulation",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            Tool(
                name="place_block",
                description="Place a new equipment block in the flowsheet (requires enhanced interface)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "block_name": {
                            "type": "string",
                            "description": "Name for the new block"
                        },
                        "equipment_type": {
                            "type": "string",
                            "description": "Type of equipment (Mixer, Heater, Flash2, Radfrac, DSTWU, RPLUG, RCSTR, RYIELD)"
                        }
                    },
                    "required": ["block_name", "equipment_type"]
                }
            ),
            Tool(
                name="delete_block",
                description="Delete an equipment block from the flowsheet (requires enhanced interface)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "block_name": {
                            "type": "string",
                            "description": "Name of the block to delete"
                        }
                    },
                    "required": ["block_name"]
                }
            ),
            Tool(
                name="place_stream",
                description="Place a new stream in the flowsheet (requires enhanced interface)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "stream_name": {
                            "type": "string",
                            "description": "Name for the new stream"
                        },
                        "stream_type": {
                            "type": "string",
                            "description": "Type of stream: MATERIAL, HEAT, or WORK (default: MATERIAL)"
                        }
                    },
                    "required": ["stream_name"]
                }
            ),
            Tool(
                name="connect_stream",
                description="Connect a stream to a block port (requires enhanced interface)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "block_name": {
                            "type": "string",
                            "description": "Name of the block"
                        },
                        "stream_name": {
                            "type": "string",
                            "description": "Name of the stream"
                        },
                        "port_name": {
                            "type": "string",
                            "description": "Name of the port on the block"
                        }
                    },
                    "required": ["block_name", "stream_name", "port_name"]
                }
            ),
            Tool(
                name="save_simulation",
                description="Save the simulation to file (requires enhanced interface)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Optional: new filename to save as"
                        }
                    }
                }
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: Any) -> list[TextContent]:
        """Handle tool calls."""
        try:
            if name == "open_simulation":
                filepath = arguments["filepath"]
                use_enhanced = arguments.get("use_enhanced", False)
                aspen.open_file(filepath, use_enhanced=use_enhanced)
                mode = "enhanced" if use_enhanced else "basic"
                return [TextContent(
                    type="text",
                    text=f"Successfully opened simulation ({mode} mode): {filepath}"
                )]

            elif name == "run_simulation":
                aspen.run_simulation()
                return [TextContent(
                    type="text",
                    text="Simulation completed successfully"
                )]

            elif name == "get_value":
                path = arguments["path"]
                value = aspen.get_value(path)
                return [TextContent(
                    type="text",
                    text=f"Value at {path}: {value}"
                )]

            elif name == "set_value":
                path = arguments["path"]
                value = arguments["value"]
                aspen.set_value(path, value)
                return [TextContent(
                    type="text",
                    text=f"Successfully set {path} to {value}"
                )]

            elif name == "close_simulation":
                aspen.close()
                return [TextContent(
                    type="text",
                    text="Simulation closed"
                )]

            elif name == "place_block":
                block_name = arguments["block_name"]
                equipment_type = arguments["equipment_type"]
                aspen.place_block(block_name, equipment_type)
                return [TextContent(
                    type="text",
                    text=f"Successfully placed {equipment_type} block: {block_name}"
                )]

            elif name == "delete_block":
                block_name = arguments["block_name"]
                aspen.delete_block(block_name)
                return [TextContent(
                    type="text",
                    text=f"Successfully deleted block: {block_name}"
                )]

            elif name == "place_stream":
                stream_name = arguments["stream_name"]
                stream_type = arguments.get("stream_type", "MATERIAL")
                aspen.place_stream(stream_name, stream_type)
                return [TextContent(
                    type="text",
                    text=f"Successfully placed {stream_type} stream: {stream_name}"
                )]

            elif name == "connect_stream":
                block_name = arguments["block_name"]
                stream_name = arguments["stream_name"]
                port_name = arguments["port_name"]
                aspen.connect_stream(block_name, stream_name, port_name)
                return [TextContent(
                    type="text",
                    text=f"Successfully connected {stream_name} to {block_name}.{port_name}"
                )]

            elif name == "save_simulation":
                filename = arguments.get("filename")
                aspen.save_simulation(filename)
                msg = f"Saved as {filename}" if filename else "Saved simulation"
                return [TextContent(
                    type="text",
                    text=msg
                )]

            else:
                raise ValueError(f"Unknown tool: {name}")

        except Exception as e:
            logger.error(f"Tool {name} failed: {e}")
            return [TextContent(
                type="text",
                text=f"Error: {str(e)}"
            )]
