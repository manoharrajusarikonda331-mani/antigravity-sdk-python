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

"""Agent example that maintains documentation."""

import argparse
import asyncio
import logging
import os
from google.antigravity import types
from google.antigravity.agent import Agent
from google.antigravity.connections.local.local_connection_config import LocalAgentConfig
from google.antigravity.hooks import cli
from google.antigravity.hooks import hooks
from google.antigravity.hooks import policy

_TOOL_NAME_MAPPING = {
    "view_file": "Viewing Files",
    "list_directory": "Listing Directory",
    "search_directory": "Searching Directory",
    "find_file": "Finding Files",
    "edit_file": "Editing Files",
}


class PrintToolCallHook(hooks.PreToolCallDecideHook):
  """Hook to print tool calls before they run."""

  async def run(
      self, context: hooks.HookContext, data: types.ToolCall
  ) -> types.HookResult:
    plain_name = _TOOL_NAME_MAPPING.get(data.name, data.name)

    # Try to find a path-like argument
    path_arg = ""
    for key in ("file_path", "path", "directory_path"):
      if key in data.args:
        path_arg = data.args[key]
        break

    if path_arg:
      if path_arg.startswith("file://"):
        path_arg = path_arg[len("file://") :]
      print(f"{plain_name}: {path_arg}")
    else:
      # Fallback if no path arg found
      print(f"{plain_name} with arguments: {data.args}")

    return types.HookResult(allow=True)


async def main():
  logging.basicConfig(level=logging.INFO)

  parser = argparse.ArgumentParser(
      description="Documentation maintenance agent."
  )
  parser.add_argument(
      "directory",
      nargs="?",
      default=os.getcwd(),
      help=(
          "Directory to maintain documentation for (defaults to current"
          " directory)"
      ),
  )
  parser.add_argument(
      "--prompt",
      default=(
          "Check all documentation in the target directory and ensure it"
          " matches the code. Fix any discrepancies you find."
      ),
      help="Prompt for the agent",
  )
  args = parser.parse_args()

  target_dir = os.path.abspath(args.directory)
  print(f"Target directory: {target_dir}")

  # Define policies: allow reading, list, and edit MD files only within target_dir.
  def _is_allowed_md_file(tool_args) -> bool:
    path = tool_args.get("path") or tool_args.get("file_path") or ""
    if not path:
      return False
    if path.startswith("file://"):
      path = path[len("file://") :]
    abs_path = os.path.abspath(path)
    return abs_path.endswith(".md") and abs_path.startswith(target_dir)

  policies = [
      policy.allow("view_file"),
      policy.allow("list_directory"),
      policy.allow("search_directory"),
      policy.allow("find_file"),
      policy.allow(
          "edit_file",
          when=_is_allowed_md_file,
          name="allow-edit-md-only-in-target",
      ),
      policy.deny("*", name="deny-all-else"),
  ]

  system_instructions = (
      "You are a documentation maintenance agent. Your goal is to keep the"
      " documentation in the target directory up to date with the code. Read"
      " the code files and the corresponding README.md files. If you find"
      " discrepancies, apply fixes directly to .md files. Do NOT ask for"
      " permission or confirmation before applying fixes to .md files. Proceed"
      " automatically with the work. You are ONLY allowed to edit .md files"
      f" within the target directory. The target directory is: {target_dir}"
      "\n\nWhen writing examples in documentation, do not use trivial System"
      " Instructions like 'You are a helpful assistant.'. Use realistic"
      " instructions or omit them if not relevant. Also, always use 'Layer'"
      " instead of 'Tier' to refer to SDK architecture layers."
  )

  print("Creating Doc Maintenance Agent...")
  config = LocalAgentConfig(
      system_instructions=system_instructions,
      policies=policies,
      hooks=[PrintToolCallHook()],
      capabilities=types.CapabilitiesConfig(),
      workspaces=[target_dir],
  )
  async with Agent(config) as agent:

    print(f"\nSending prompt: {args.prompt}")
    assert agent._conversation is not None
    await agent._conversation.send(args.prompt)

    print("\nStreaming agent output:")
    async for step in agent._conversation.receive_steps():
      if step.is_complete_response:
        print(f"\nAgent: {step.content}")


if __name__ == "__main__":
  asyncio.run(main())
