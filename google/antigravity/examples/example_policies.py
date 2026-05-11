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

"""Example policies for the Google Antigravity SDK.

These are provided as examples and starting points for developers to iterate
from. They demonstrate the value of strongly typed predicates.
"""

import pydantic
from google.antigravity.hooks import cli
from google.antigravity.hooks import policy


class RunCommandArgs(pydantic.BaseModel):
  """Arguments for run_command tool."""

  command_line: str


class DeleteFileArgs(pydantic.BaseModel):
  """Arguments for delete_file tool."""

  path: str


def _block_rm_predicate(args: RunCommandArgs) -> bool:
  return "rm" in args.command_line


# Example policy: Denies run_command if it contains 'rm'.
# This demonstrates a simple denylist approach.
BLOCK_RM_POLICY = policy.Policy(
    tool="run_command",
    decision=policy.Decision.DENY,
    when=_block_rm_predicate,
    name="block-rm",
)


def _command_allowlist_predicate(args: RunCommandArgs) -> bool:
  allowed_commands = ["ls", "git status", "git diff", "pytest"]
  return not any(args.command_line.startswith(cmd) for cmd in allowed_commands)


# Example policy: Only allow a small set of safe commands.
# This demonstrates a safer allowlist approach using typed arguments.
ONLY_ALLOW_SAFE_COMMANDS_POLICY = policy.Policy(
    tool="run_command",
    decision=policy.Decision.DENY,
    when=_command_allowlist_predicate,
    name="only-allow-safe-commands",
)


def _critical_file_predicate(args: DeleteFileArgs) -> bool:
  return args.path.endswith(".key") or "production" in args.path


# Example policy: Ask user before deleting critical files.
# This demonstrates using ASK_USER decision with a typed predicate.
ASK_FOR_CRITICAL_DELETES = policy.Policy(
    tool="delete_file",
    decision=policy.Decision.ASK_USER,
    when=_critical_file_predicate,
    ask_user=cli.ask_user_handler,
    name="ask-for-critical-deletes",
)
