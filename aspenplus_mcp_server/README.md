# AspenPlus MCP Server

MCP (Model Context Protocol) server for Aspen Plus process simulation integration. This server enables AI assistants to interact with Aspen Plus simulations through a standardized interface.

## Features

- **Process Simulation Control**: Open, run, and manage Aspen Plus simulations
- **Data Access**: Read and write stream properties, block parameters, and simulation results
- **Flowsheet Manipulation**: Create, delete, and connect equipment blocks and streams (enhanced mode)
- **Equipment Support**: Mixer, Heater, Flash2, Radfrac, DSTWU, RPLUG, RCSTR, RYIELD
- **AI Integration**: Connect Claude and other AI assistants to Aspen Plus

## Requirements

- Python 3.10 or higher
- Windows OS (for Aspen Plus COM interface)
- Aspen Plus installed
- pywin32 package

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd AspenPlus-MCP-Server
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

## Usage

### Running the Server

```bash
python -m aspenplus_mcp.server
```

### Available Tools

#### Basic Mode Tools
- `open_simulation` - Open an Aspen Plus simulation file (set `use_enhanced: false`)
- `run_simulation` - Run the current simulation
- `get_value` - Get a value from the simulation using a node path
- `set_value` - Set a value in the simulation
- `close_simulation` - Close the current simulation

#### Enhanced Mode Tools (requires `use_enhanced: true`)
- `place_block` - Add a new equipment block to the flowsheet
- `delete_block` - Remove an equipment block from the flowsheet
- `place_stream` - Add a new stream to the flowsheet
- `connect_stream` - Connect a stream to a block port
- `save_simulation` - Save the simulation to file

### Example Node Paths

```
# Stream temperature
\\Data\\Streams\\S1\\Output\\TEMP_OUT\\MIXED\\MIXED

# Block parameter
\\Data\\Blocks\\B1\\Input\\TEMP

# Component flow rate
\\Data\\Streams\\S1\\Output\\MOLEFLOW\\MIXED\\WATER
```

## Development

Install development dependencies:
```bash
pip install -e ".[dev]"
```

Run tests:
```bash
pytest
```

Format code:
```bash
black src/
```

## Configuration with Claude Desktop

Add to your Claude Desktop config file:

**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

**Linux**: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "aspenplus": {
      "command": "C:\\path\\to\\AspenPlus-MCP-Server\\venv\\Scripts\\python.exe",
      "args": [
        "-m",
        "aspenplus_mcp.server"
      ]
    }
  }
}
```

**Important Notes:**
- Use the **full absolute path** to your virtual environment's Python executable
- On Windows, use double backslashes (`\\`) in paths or forward slashes (`/`)
- After editing the config file, restart Claude Desktop completely
- The MCP server will start automatically when Claude Desktop launches

## Usage Examples

### Basic Mode - Reading/Writing Values
```python
# Open simulation in basic mode
open_simulation(filepath="C:/simulations/myprocess.bkp", use_enhanced=False)

# Read stream temperature
get_value(path="\\Data\\Streams\\S1\\Output\\TEMP_OUT\\MIXED\\MIXED")

# Set feed temperature
set_value(path="\\Data\\Streams\\FEED\\Input\\TEMP", value=25.0)

# Run simulation
run_simulation()
```

### Enhanced Mode - Flowsheet Editing
```python
# Open simulation in enhanced mode
open_simulation(filepath="C:/simulations/myprocess.bkp", use_enhanced=True)

# Add a mixer block
place_block(block_name="MIX-01", equipment_type="Mixer")

# Add streams
place_stream(stream_name="STREAM-1", stream_type="MATERIAL")
place_stream(stream_name="STREAM-2", stream_type="MATERIAL")

# Connect streams to mixer
connect_stream(block_name="MIX-01", stream_name="STREAM-1", port_name="IN1")
connect_stream(block_name="MIX-01", stream_name="STREAM-2", port_name="IN2")

# Save changes
save_simulation()
```

## Credits

This project integrates the [AspenPlus-Python-Interface](https://github.com/YouMayCallMeJesus/AspenPlus-Python-Interface) by Richard ten Hagen for advanced flowsheet manipulation capabilities.

## License

MIT
