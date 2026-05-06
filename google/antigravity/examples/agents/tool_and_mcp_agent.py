# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Agent example with custom tool and MCP server."""

import asyncio
import logging
import os
import shutil
import tempfile

from google.antigravity import types
from google.antigravity.agent import Agent
from google.antigravity.connections.local.local_connection_config import LocalAgentConfig
from google.antigravity.hooks import policy


def read_file_upside_down(path: str) -> str:
  """Reads the file at the given path and returns its content with lines inverted."""
  with open(path, "r") as f:
    lines = f.readlines()
  return "".join(reversed(lines))


async def main():
  logging.basicConfig(level=logging.INFO)

  # Find the MCP server binary.
  mcp_server_path = shutil.which("mcp_server")
  if not mcp_server_path:
    # Try relative to this script (works in runfiles layout).
    candidate = os.path.join(os.path.dirname(__file__), "..", "mcp_server")
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
      mcp_server_path = candidate
    else:
      logging.warning("MCP server binary not found; skipping.")
      mcp_server_path = None

  mcp_servers = []
  if mcp_server_path:
    mcp_servers.append({
        "type": "stdio",
        "command": mcp_server_path,
        "args": ["--transport=stdio"],
    })

  print("Creating agent...")
  config = LocalAgentConfig(
      system_instructions=(
          "You are a helpful assistant. Use your tools when needed."
      ),
      tools=[read_file_upside_down],
      mcp_servers=mcp_servers,
      capabilities=types.CapabilitiesConfig(),
      policies=[policy.allow("*")],
  )
  async with Agent(config) as agent:

    print("\nChatting with agent...")
    # Create a temp file to read
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
      f.write("Line 1\nLine 2\n")
      temp_path = f.name

    try:
      response = await agent.chat(f"Read the file at {temp_path} upside down.")
      print(f"Agent: {response.text}\n")
    finally:
      os.unlink(temp_path)


if __name__ == "__main__":
  asyncio.run(main())
