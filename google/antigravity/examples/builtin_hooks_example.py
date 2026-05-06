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

r"""Example demonstrating hooks for built-in tool lifecycle events.

Built-in tools (view_file, run_command, edit_file, etc.) are executed inside
the Go harness. The SDK can observe and control them through the following
hook points:

  - PreToolCallDecideHook — approve or deny built-in tool execution
  - PostToolCallHook — observe the result when a built-in tool completes
  - OnToolErrorHook — observe errors when a built-in tool fails

Observing model responses:
  To observe model-generated text, use PostTurnHook (which receives the
  final response after each turn) or inspect conversation.history for the
  full step-by-step trajectory.

To run:
  python3 builtin_hooks_example.py
"""

import asyncio
from collections.abc import Sequence
import os
import tempfile

from absl import app
from absl import logging

from google.antigravity import types
from google.antigravity.connections.local.local_connection import LocalConnectionStrategy
from google.antigravity.connections.local import types as local_types
from google.antigravity.conversation.conversation import Conversation
from google.antigravity.hooks import hook_runner as hooks_runner
from google.antigravity.hooks import hooks

# =============================================================================
# Hook implementations for built-in tool lifecycle
# =============================================================================


class LogBuiltinDecide(hooks.PreToolCallDecideHook):
  """Logs every tool decision and demonstrates selective denial.

  Denies any run_command calls that contain 'rm -rf' in the command line,
  demonstrating policy enforcement for built-in tools.
  """

  async def run(self, context, data) -> types.HookResult:
    print(f"[Decide] Tool: {data.name}, Args keys: {list(data.args.keys())}")

    # Example policy: deny dangerous commands.
    if data.name == types.BuiltinTools.RUN_COMMAND.value:
      cmd = data.args.get("CommandLine", "")
      if "rm -rf" in cmd:
        print(f"[Decide] DENIED dangerous command: {cmd}")
        return types.HookResult(allow=False, message="Dangerous command denied")

    print(f"[Decide] APPROVED: {data.name}")
    return types.HookResult(allow=True)


class LogBuiltinPostTool(hooks.PostToolCallHook):
  """Logs built-in tool results after completion.

  Demonstrates isinstance inspection of structured result types
  (RunCommandResult, ListDirectoryResult, etc.) to extract specific
  fields from each tool's output.
  """

  async def run(self, context, data):
    if not isinstance(data, types.ToolResult):
      print(f"[PostTool] Data: {data}")
      return

    result = data.result
    print(f"[PostTool] Tool: {data.name}")

    if isinstance(result, local_types.RunCommandResult):
      print(f"[PostTool] Command output: {result.output[:200]}")
    elif isinstance(result, local_types.ListDirectoryResult):
      print(f"[PostTool] {len(result.entries)} entries:")
      for entry in result.entries:
        kind = "dir" if entry.is_directory else f"{entry.file_size}b"
        print(f"[PostTool]   {entry.name} ({kind})")
    elif isinstance(result, local_types.SearchDirectoryResult):
      print(f"[PostTool] Search matched {result.num_results} results")
    elif isinstance(result, local_types.FindFileResult):
      print(f"[PostTool] Find output: {result.output[:200]}")
    elif isinstance(result, local_types.EditFileResult):
      print(f"[PostTool] Edit summary: {result.summary[:200]}")
    elif isinstance(result, local_types.GenerateImageResult):
      print(f"[PostTool] Generated image: {result.image_name}")
    else:
      # Fallback for unstructured results (e.g. view_file toolSummary).
      print(f"[PostTool] Result: {str(result)[:200]}")


class LogBuiltinError(hooks.OnToolErrorHook):
  """Logs built-in tool errors for observability."""

  async def run(self, context, data):
    print(f"[ToolError] Error: {data}")
    return None  # No recovery; let the error propagate.


class LogPostTurn(hooks.PostTurnHook):
  """Observes the final model response after each turn.

  This is the recommended way to observe model-generated text.
  """

  async def run(self, context, data):
    preview = str(data)[:150] if data else "(empty)"
    print(f"[PostTurn] Model response: {preview}")


# =============================================================================
# Helper to run a single prompt
# =============================================================================


async def run_prompt(conversation: Conversation, prompt: str) -> None:
  """Sends a prompt and prints the final response."""
  print(f"\n{'='*60}")
  print(f"--- Sending: {prompt!r:.80} ---")
  print(f"{'='*60}")
  await conversation.send(prompt)
  async for step in conversation.receive_steps():
    if step.is_complete_response:
      cascade_id = getattr(step, "cascade_id", "")
      trajectory_id = getattr(step, "trajectory_id", "")
      is_parent = not cascade_id or trajectory_id == cascade_id
      label = "Final response" if is_parent else "Subagent response"
      print(f"\n--- {label} ---\n{step.content[:200]}\n")


# =============================================================================
# Main
# =============================================================================


async def run():
  """Runs the built-in hooks example."""
  # Create sample files in a temp directory so prompts work in any environment.
  with tempfile.TemporaryDirectory() as tmpdir:
    sample_file = os.path.join(tmpdir, "sample.py")
    with open(sample_file, "w") as f:
      f.write("# PostToolCallHook test file\nprint('hello')\n")
    for name in ["alpha.txt", "beta.txt"]:
      with open(os.path.join(tmpdir, name), "w") as f:
        f.write(f"contents of {name}\n")

    hr = hooks_runner.HookRunner(
        pre_tool_call_decide_hooks=[LogBuiltinDecide()],
        post_tool_call_hooks=[LogBuiltinPostTool()],
        on_tool_error_hooks=[LogBuiltinError()],
        post_turn_hooks=[LogPostTurn()],
    )

    strategy = LocalConnectionStrategy(
        hook_runner=hr,
        gemini_config=types.GeminiConfig(),
        capabilities_config=types.CapabilitiesConfig(
            enable_subagents=True,
        ),
    )

    async with Conversation.create(strategy) as conversation:
      # 1. run_command: exercise command execution.
      await run_prompt(
          conversation,
          "Run the command 'echo hello world' and tell me what it printed.",
      )

      # 2. list_dir: exercise directory listing.
      await run_prompt(
          conversation,
          f"List the contents of the directory {tmpdir}.",
      )

      # 3. view_file: exercise file viewing.
      await run_prompt(
          conversation,
          f"View the contents of the file {sample_file}.",
      )

      # 4. grep_search: exercise file content search.
      await run_prompt(
          conversation,
          f"Search for the string 'PostToolCallHook'"
          f" in the file {sample_file}.",
      )

      # 5. PostTurnHook only: no tools needed.
      await run_prompt(
          conversation,
          "What is 2 + 2? Answer directly without using any tools.",
      )

      print("\n--- All prompts complete ---")


def main(argv: Sequence[str]) -> None:
  del argv
  logging.set_verbosity(logging.INFO)
  asyncio.run(run())


if __name__ == "__main__":
  app.run(main)
