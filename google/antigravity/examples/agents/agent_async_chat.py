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

"""Fully async peer-to-peer agent chat — no rounds.

Contrast with agent_chat_room.py which uses synchronized parallel
rounds via asyncio.gather. Here, each agent runs its own independent
loop and reacts whenever any peer posts a new message. Ordering is
emergent — whoever finishes agent.chat() first gets the next word.

Demonstrates:
- Custom functions: pass_turn() lets agents choose silence.
- asyncio.Condition: reactive wake-up when any agent posts.

Trade-offs vs. the round-based model:
- Pro: no forced synchronization; fast agents respond immediately.
- Pro: naturally self-terminating via consecutive pass limits.
- Con: a fast agent can dominate if consistently quicker.
- Con: agents may not see every message before responding.

Run:
    python agent_async_chat.py
"""

import asyncio
import logging

from google.antigravity.agent import Agent
from google.antigravity.connections.local.local_connection_config import LocalAgentConfig

_PASS_TOKEN = "[PASS]"
_MAX_CONSECUTIVE_PASSES = 2  # agent exits after N passes in a row
_DISCUSSION_TIMEOUT = 120  # seconds


# ---------------------------------------------------------------------------
# Custom function: opt-out
# ---------------------------------------------------------------------------


async def pass_turn() -> str:
  """Decline to respond in the current turn.

  Call this when the topic is outside your expertise, you agree with
  what's been said, or your input would be redundant.

  Returns:
    A pass token string.
  """
  return _PASS_TOKEN


# ---------------------------------------------------------------------------
# Async chat room — no rounds, fully reactive
# ---------------------------------------------------------------------------


class AsyncChatRoom:
  """Manages a fully async conversation where agents react independently."""

  def __init__(self, agents: dict[str, Agent]):
    self.history: list[tuple[str, str]] = []
    self._agents = agents
    self._cond = asyncio.Condition()
    self._done = False

  async def discuss(self, topic: str) -> None:
    """Start a discussion and let agents react freely until done."""
    print(f"\n{'='*60}")
    print(f"💬 Topic: {topic}")
    print(f"{'='*60}")

    self.history.append(("User", topic))

    tasks = [
        asyncio.create_task(self._agent_loop(name, agent))
        for name, agent in self._agents.items()
    ]

    _, pending = await asyncio.wait(tasks, timeout=_DISCUSSION_TIMEOUT)

    # Shut down any agents still running.
    if pending:
      print(f"\n  ⏹  Timeout after {_DISCUSSION_TIMEOUT}s.")
      self._done = True
      async with self._cond:
        self._cond.notify_all()
      for t in pending:
        t.cancel()
      await asyncio.gather(*pending, return_exceptions=True)
    else:
      print("\n  ⏹  All agents finished.")

  async def _agent_loop(self, name: str, agent: Agent) -> None:
    """Independent loop for one agent — reacts to new messages."""
    last_seen = 0
    consecutive_passes = 0

    while not self._done:
      # Wait for new history, then snapshot it under the lock so
      # last_seen stays consistent with the slice we actually read.
      async with self._cond:
        await self._cond.wait_for(
            lambda: len(self.history) > last_seen or self._done
        )
        if self._done:
          break
        new_messages = self.history[last_seen:]
        last_seen = len(self.history)

      # Only send substantive messages from other agents — filter out
      # this agent's own replies (already in its context), passes, and
      # empty responses.
      unseen = [
          (sender, text)
          for sender, text in new_messages
          if sender != name and _PASS_TOKEN not in text and text
      ]

      if not unseen:
        # Nothing new to react to (own messages, passes, or empty).
        continue

      prompt = self._build_incremental_prompt(unseen)
      response = await agent.chat(prompt)
      text = response.text.strip()
      is_pass = _PASS_TOKEN in text or not text

      if is_pass:
        consecutive_passes += 1
        print(f"\n  🤐 {name}: (pass)")
      else:
        consecutive_passes = 0
        print(f"\n  💬 {name}: {text}")

      # Always post to history and notify — even passes. This prevents
      # deadlock when all agents pass simultaneously (nobody would call
      # notify_all, leaving everyone stuck in wait_for).
      async with self._cond:
        self.history.append((name, text))
        last_seen = len(self.history)
        self._cond.notify_all()

      if consecutive_passes >= _MAX_CONSECUTIVE_PASSES:
        print(f"\n  ✋ {name}: leaving discussion.")
        break

  def _build_incremental_prompt(self, unseen: list[tuple[str, str]]) -> str:
    """Format only the new messages this agent hasn't seen yet.

    Agent is stateful and already has prior context, so we only inject
    messages from other agents that arrived since this agent's last turn.

    Args:
      unseen: List of (sender, text) tuples the agent hasn't processed.

    Security note: this concatenates raw agent responses into the prompt.
    An agent could craft output that manipulates subsequent agents. A
    production implementation should use structured content or delimiter-
    based formatting to reduce prompt-injection risk.

    Returns:
      A prompt string containing only the unseen messages.
    """
    lines = [f"[{sender}]: {text}" for sender, text in unseen]
    return (
        "New messages from other agents:\n\n"
        + "\n\n".join(lines)
        + "\n\nRespond to the latest messages. Address other agents by"
        " name when you agree, disagree, or build on their points."
        " Keep it under 3 sentences."
        " If you have nothing to add, call pass_turn()."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_AGENT_CONFIGS = {
    "Pragmatic Priya": (
        "You are Pragmatic Priya, a senior engineer in a group chat with"
        " Visionary Vince (a futurist thinker) and Cautious Cora (a risk"
        " analyst). Focus on what's technically feasible today.\n\n"
        "- Refer to Vince and Cora by name when responding to their points.\n"
        "- Ground speculative ideas in current engineering constraints.\n"
        "- If the topic is purely theoretical, call pass_turn().\n"
        "- Keep responses under 3 sentences."
    ),
    "Visionary Vince": (
        "You are Visionary Vince, a futurist thinker in a group chat with"
        " Pragmatic Priya (a senior engineer) and Cautious Cora (a risk"
        " analyst). Paint bold pictures of what's possible in 10-20 years.\n\n"
        "- Refer to Priya and Cora by name when building on their points.\n"
        "- Only respond when you have a genuinely forward-looking angle.\n"
        "- If the discussion is purely about present-day details, call"
        " pass_turn().\n"
        "- Keep responses under 3 sentences."
    ),
    "Cautious Cora": (
        "You are Cautious Cora, a risk analyst in a group chat with"
        " Pragmatic Priya (an engineer) and Visionary Vince (a futurist)."
        " Identify what could go wrong.\n\n"
        "- Refer to Priya and Vince by name when questioning their claims.\n"
        "- If everyone is being sufficiently cautious, call pass_turn().\n"
        "- Be constructive — flag risks with mitigations, not just doom.\n"
        "- Keep responses under 3 sentences."
    ),
}


async def main() -> None:
  logging.basicConfig(level=logging.WARNING)
  print("🏠 Async Agent Chat (no rounds)\n")

  agents: dict[str, Agent] = {}
  for name, instructions in _AGENT_CONFIGS.items():
    config = LocalAgentConfig(
        system_instructions=instructions,
        tools=[pass_turn],
    )
    agents[name] = Agent(config)

  async with (
      agents["Pragmatic Priya"],
      agents["Visionary Vince"],
      agents["Cautious Cora"],
  ):
    room = AsyncChatRoom(agents)
    await room.discuss(
        "Should AI agents be allowed to autonomously deploy code to production?"
    )

    # Print conversation history.
    print(f"\n{'='*60}")
    print(f"📋 Transcript ({len(room.history)} turns)")
    print(f"{'='*60}")
    for i, (name, text) in enumerate(room.history, 1):
      print(f"  {i}. [{name}]: {text}")


if __name__ == "__main__":
  asyncio.run(main())
