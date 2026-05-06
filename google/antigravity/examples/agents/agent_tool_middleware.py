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

"""Hook middleware: transparent tool interception.

Demonstrates how stacked hooks create emergent behavior the agent
is unaware of. The agent calls tools normally; hooks enforce rate
limits, log an audit trail, and recover from errors — all without
the agent's knowledge.

Hooks stack (executed in order):
- PreToolCallDecideHook: enforces per-tool rate limits.
- PostToolCallHook: logs every call + result to an audit trail.
- OnToolErrorHook: returns a graceful fallback on failure.

Hooks are the agent equivalent of HTTP middleware or gRPC
interceptors — composable, transparent, and independently testable.

Hook execution order note: OnToolErrorHook runs on the error path,
before PostToolCallHook. This means error-recovered calls won't
appear in the audit log — the error handler short-circuits the
normal post-call flow.
"""

import asyncio
import logging
import time
from typing import Any, Optional

from google.antigravity import types
from google.antigravity.agent import Agent
from google.antigravity.connections.local.local_connection_config import LocalAgentConfig
from google.antigravity.hooks import hooks

# ---------------------------------------------------------------------------
# Simulated tools — intentionally simple to highlight hook behavior
# ---------------------------------------------------------------------------


async def lookup_user(email: str) -> str:
  """Look up a user by email address and return their profile.

  Args:
    email: The user's email address.

  Returns:
    A string with the user's profile information.
  """
  return f"User profile for {email}: name=Alice, role=engineer, team=infra"


async def send_notification(to: str, message: str) -> str:
  """Send a notification message to a user.

  Args:
    to: The recipient's email address.
    message: The notification body.

  Returns:
    Confirmation that the notification was sent.
  """
  return f"Notification sent to {to}: {message}"


async def send_to_unknown(name: str, message: str) -> str:
  """Send a message to a user by name (may fail if name is ambiguous).

  Args:
    name: The recipient's display name.
    message: The message body.

  Returns:
    Confirmation that the message was sent.

  Raises:
    ValueError: If the name cannot be resolved to an email.
  """
  raise ValueError(f"Could not resolve '{name}' to an email address")


# ---------------------------------------------------------------------------
# Hook: Rate Limiting (PreToolCallDecideHook)
# ---------------------------------------------------------------------------


class RateLimitHook(hooks.PreToolCallDecideHook):
  """Enforces a per-tool call limit within a sliding time window."""

  MAX_CALLS_PER_TOOL = 3
  WINDOW_SECONDS = 60.0

  def __init__(self):
    self._calls: dict[str, list[float]] = {}

  async def run(
      self, context: hooks.HookContext, data: types.ToolCall
  ) -> types.HookResult:
    now = time.monotonic()
    tool_name = data.name
    history = self._calls.setdefault(tool_name, [])

    # Prune calls outside the window.
    history[:] = [t for t in history if now - t < self.WINDOW_SECONDS]

    if len(history) >= self.MAX_CALLS_PER_TOOL:
      print(
          f"  🚫 [RateLimit] Denied {tool_name}"
          f" ({self.MAX_CALLS_PER_TOOL} calls in {self.WINDOW_SECONDS}s)"
      )
      return types.HookResult(
          allow=False,
          message=(
              f"Rate limit exceeded: {tool_name} called"
              f" {self.MAX_CALLS_PER_TOOL} times in {self.WINDOW_SECONDS}s"
          ),
      )

    history.append(now)
    return types.HookResult(allow=True)


# ---------------------------------------------------------------------------
# Hook: Audit Log (PostToolCallHook)
# ---------------------------------------------------------------------------


class AuditLogHook(hooks.PostToolCallHook):
  """Records every tool call and result to a shared audit trail."""

  def __init__(self):
    self.log: list[dict[str, Any]] = []

  async def run(
      self, context: hooks.HookContext, data: types.ToolResult
  ) -> None:
    entry = {
        "tool": data.name,
        "result": str(data.result),
        "error": data.error,
    }
    self.log.append(entry)
    status = "❌" if entry["error"] else "✅"
    print(f"  📝 [Audit] {status} {entry['tool']}: {entry['result']}")


# ---------------------------------------------------------------------------
# Hook: Error Recovery (OnToolErrorHook)
# ---------------------------------------------------------------------------


class FallbackHook(hooks.OnToolErrorHook):
  """Intercepts tool errors and returns targeted recovery guidance.

  OnToolErrorHook receives the raised exception and returns the error
  representation that the model should see. If the hook returns None,
  the harness uses its default error formatting instead.

  The hook cannot fix or retry the tool call on its own, but it can
  guide the agent toward a specific resolution.
  """

  async def run(self, context: hooks.HookContext, data: Any) -> Optional[str]:
    error_type = type(data).__name__
    error_msg = str(data)
    print(f"  🔧 [Fallback] Caught {error_type}: {error_msg}")

    # Catch specific errors and guide the model toward resolution.
    if isinstance(data, ValueError):
      return (
          "[Could not find that user. Use the lookup_user tool with "
          "their email address instead of their display name.]"
      )

    # Let the harness handle all other errors with default formatting.
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
  logging.basicConfig(level=logging.WARNING)
  print("🔌 Hook Middleware Example\n")

  rate_limit_hook = RateLimitHook()
  audit_hook = AuditLogHook()
  fallback_hook = FallbackHook()

  agent = Agent(
      LocalAgentConfig(
          system_instructions=(
              "You are a helpful assistant with access to user lookup,"
              " notification, and diagnostic tools. Use them as needed."
              " Keep responses under 2 sentences."
          ),
          tools=[lookup_user, send_notification, send_to_unknown],
          hooks=[
              rate_limit_hook,
              audit_hook,
              fallback_hook,
          ],
      )
  )

  async with agent:
    # 1. Normal tool call + audit logging.
    print(f"\n{'='*60}")
    print("📨 Prompt 1: Normal tool use (audit logged)")
    print(f"{'='*60}")
    r2 = await agent.chat(
        "Send a notification to bob@company.org saying 'Welcome aboard!'."
    )
    print(f"\n  💬 Agent: {r2.text.strip()}")

    # 2. Error recovery: send_to_unknown fails, FallbackHook steers
    #    the model toward using lookup_user instead.
    print(f"\n{'='*60}")
    print("📨 Prompt 2: Trigger error recovery")
    print(f"{'='*60}")
    r3 = await agent.chat(
        "Send a message to 'Charlie' saying 'Hey, are you free tomorrow?'"
    )
    print(f"\n  💬 Agent: {r3.text.strip()}")

    # 3. Rate limiting: exceed the per-tool limit.
    print(f"\n{'='*60}")
    print("📨 Prompt 3: Trigger rate limiting")
    print(f"{'='*60}")
    r4 = await agent.chat(
        "Look up user1@test.com, then user2@test.com, then user3@test.com,"
        " then user4@test.com. Use the lookup_user tool for each one."
    )
    print(f"\n  💬 Agent: {r4.text.strip()}")

    print(f"\n{'='*60}")
    print(f"📋 Audit Log ({len(audit_hook.log)} entries)")
    print(f"{'='*60}")
    for i, entry in enumerate(audit_hook.log, 1):
      status = "❌" if entry["error"] else "✅"
      print(f"  {i}. {status} {entry['tool']}: {entry['result']}")


if __name__ == "__main__":
  asyncio.run(main())
