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

"""Synchronized parallel agent chat room with opt-out.

Three agents (Rational Rita, Creative Cal, Skeptical Sam) discuss topics
as equals. All agents process in parallel each round. Each can call
pass_turn() to stay silent. Conversation continues until all agents pass
or the max depth is reached.

Demonstrates:
- Custom functions: pass_turn() lets agents choose silence.
- Triggers: a 60s timer nudges all agents to wrap up.
- Async parallelism: all agents think simultaneously via asyncio.gather.

Design note: we use agent.chat() in parallel via asyncio.gather rather
than a purely trigger-driven model. TriggerContext.send() delivers
messages and returns the response, but it bypasses Agent.chat() and
operates on the Connection directly. The agent.chat() approach gives us
explicit round management and prompt injection. See agent_async_chat.py
for a fully reactive alternative using asyncio.Condition.

Run:
    python agent_chat_room.py
"""

import asyncio
import logging

from google.antigravity.agent import Agent
from google.antigravity.connections.local.local_connection_config import LocalAgentConfig
from google.antigravity.triggers import helpers
from google.antigravity.triggers import triggers as triggers_module

_PASS_TOKEN = "[PASS]"
_MAX_ROUNDS = 4


# ---------------------------------------------------------------------------
# Custom function: opt-out
# ---------------------------------------------------------------------------


async def pass_turn() -> str:
  """Decline to respond in the current round.

  Call this when the topic is outside your expertise, you agree with
  what's been said, or your input would be redundant.

  Returns:
    A pass token string.
  """
  return _PASS_TOKEN


# ---------------------------------------------------------------------------
# Trigger: moderator nudge after a delay
# ---------------------------------------------------------------------------


async def _moderator_nudge(ctx: triggers_module.TriggerContext) -> None:
  """Nudges the agent to wrap up after a delay."""
  await ctx.send(
      "The discussion is wrapping up. Make your final point concisely.",
  )


# ---------------------------------------------------------------------------
# Chat room
# ---------------------------------------------------------------------------


class ChatRoom:
  """Manages a synchronized round-based conversation between agents."""

  def __init__(self, agents: dict[str, Agent]):
    self.history: list[tuple[str, str]] = []
    self._agents = agents
    self._last_seen: dict[str, int] = {name: 0 for name in agents}

  async def discuss(self, topic: str) -> None:
    """Run a multi-round discussion with synchronized parallel turns."""
    print(f"\n{'='*60}")
    print(f"💬 Topic: {topic}")
    print(f"{'='*60}")

    self.history.append(("User", topic))

    for _ in range(_MAX_ROUNDS):
      responses = await self._parallel_round()

      if not responses:
        print("\n  ⏹  All agents passed — discussion complete.")
        break

      for name, text in responses:
        self.history.append((name, text))
    else:
      print(f"\n  ⏹  Max rounds reached ({_MAX_ROUNDS}).")

  async def _parallel_round(self) -> list[tuple[str, str]]:
    """Ask all agents simultaneously, return non-pass responses."""

    async def _ask(name: str, ag: Agent) -> tuple[str, str]:
      # Build a per-agent prompt with only messages it hasn't seen,
      # excluding its own (already in the stateful Agent's context).
      unseen = [
          (sender, text)
          for sender, text in self.history[self._last_seen[name] :]
          if sender != name
      ]
      # Safe: each concurrent _ask task only accesses its own key.
      self._last_seen[name] = len(self.history)
      if not unseen:
        return (name, "")
      prompt = self._build_incremental_prompt(unseen)
      response = await ag.chat(prompt)
      return (name, response.text.strip())

    tasks = [_ask(n, a) for n, a in self._agents.items()]
    results = await asyncio.gather(*tasks)

    responses = []
    for name, text in results:
      if _PASS_TOKEN in text or not text:
        print(f"\n  🤐 {name}: (pass)")
      else:
        print(f"\n  💬 {name}: {text}")
        responses.append((name, text))

    return responses

  def _build_incremental_prompt(self, unseen: list[tuple[str, str]]) -> str:
    """Format only the new messages this agent hasn't seen yet.

    Agent is stateful and already has prior context, so we only inject
    messages from other agents that arrived since its last turn.

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
    "Rational Rita": (
        "You are Rational Rita, a research specialist in a group chat with"
        " Creative Cal (an imaginative thinker) and Skeptical Sam (a devil's"
        " advocate). Give concise, factual answers grounded in evidence.\n\n"
        "- Refer to Cal and Sam by name when responding to their points.\n"
        "- Correct inaccuracies from other agents.\n"
        "- If the topic is purely creative/opinion, call pass_turn().\n"
        "- Keep responses under 3 sentences."
    ),
    "Creative Cal": (
        "You are Creative Cal, a creative thinker in a group chat with"
        " Rational Rita (a fact-driven researcher) and Skeptical Sam (a"
        " devil's advocate). Offer imaginative perspectives and metaphors.\n\n"
        "- Refer to Rita and Sam by name when building on their points.\n"
        "- Only respond when you have a genuinely fresh angle.\n"
        "- If the discussion is purely factual, call pass_turn().\n"
        "- Keep responses under 3 sentences."
    ),
    "Skeptical Sam": (
        "You are Skeptical Sam, a devil's advocate in a group chat with"
        " Rational Rita (a researcher) and Creative Cal (a creative"
        " thinker). Challenge assumptions and poke holes.\n\n"
        "- Refer to Rita and Cal by name when questioning their claims.\n"
        "- If everyone is being balanced, call pass_turn().\n"
        "- Be constructive, not contrarian for its own sake.\n"
        "- Keep responses under 3 sentences."
    ),
}


async def main() -> None:
  logging.basicConfig(level=logging.WARNING)
  print("🏠 Agent Chat Room\n")

  agents: dict[str, Agent] = {}
  for name, instructions in _AGENT_CONFIGS.items():
    config = LocalAgentConfig(
        system_instructions=instructions,
        tools=[pass_turn],
        triggers=[helpers.every(60, _moderator_nudge)],
    )
    agents[name] = Agent(config)

  async with (
      agents["Rational Rita"],
      agents["Creative Cal"],
      agents["Skeptical Sam"],
  ):
    room = ChatRoom(agents)

    topics = [
        "Should we colonize Mars, or focus on fixing Earth first?",
        "What's the most overrated programming language?",
    ]

    for topic in topics:
      await room.discuss(topic)

    # Print conversation history.
    print(f"\n{'='*60}")
    print(f"📋 Transcript ({len(room.history)} turns)")
    print(f"{'='*60}")
    for i, (name, text) in enumerate(room.history, 1):
      print(f"  {i}. [{name}]: {text}")


if __name__ == "__main__":
  asyncio.run(main())
