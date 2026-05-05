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

"""Stateful conversation session for the Antigravity SDK.

Conversation is the Layer 2 session API. It wraps a Connection with:
- Step history accumulation (with compaction index tracking)
- A chat() convenience method (send + collect in one call)
- State introspection (idle, turn count, last response)

Layer 1 (Agent) delegates to Conversation; power users can use
Conversation directly with any ConnectionStrategy.
"""

import contextlib
from typing import Any, AsyncIterator

from google.antigravity import types
from google.antigravity.connections import connection


# Default maximum number of steps to retain in history.
_DEFAULT_MAX_HISTORY_SIZE = 10_000


class Conversation:
  """Stateful session wrapping a single conversation with the agent.

  Accumulates step history, tracks turn start indices and compaction indices,
  and provides convenience methods for common send/receive patterns.
  """

  def __init__(
      self,
      conn: connection.Connection,
      *,
      max_history_size: int = _DEFAULT_MAX_HISTORY_SIZE,
  ):
    """Initializes the conversation with a connection and empty history.

    Args:
      conn: The established connection to the agent backend.
      max_history_size: Maximum number of steps to retain in history.
        When exceeded, the oldest steps are discarded. Set to 0 to
        disable the limit.
    """
    self._connection = conn
    self._steps: list[types.Step] = []
    self._turn_start_indices: list[int] = []
    self._compaction_indices: list[int] = []
    self._max_history_size = max_history_size

  @classmethod
  @contextlib.asynccontextmanager
  async def create(
      cls,
      strategy: connection.ConnectionStrategy,
  ) -> AsyncIterator["Conversation"]:
    """Creates a new conversation.

    Args:
      strategy: The connection strategy to use to interact with an agent.
    """
    async with strategy:
      yield cls(strategy.connect())

  # ---------------------------------------------------------------------------
  # Core send / receive
  # ---------------------------------------------------------------------------

  async def send(
      self,
      prompt: types.Content | None,
      **kwargs: Any,
  ) -> None:
    """Sends a message to the agent.

    Records a turn boundary in the history before sending.

    Args:
      prompt: The user message to send.
      **kwargs: Strategy-specific options.
    """
    await self.wait_for_idle()
    self._turn_start_indices.append(len(self._steps))
    await self._connection.send(prompt, **kwargs)

  async def receive_steps(self) -> AsyncIterator[types.Step]:
    """Receives steps as they complete, blocks until execution is idle.

    Steps are recorded in history as they arrive. The iterator exits once
    the execution turn is complete.

    Yields:
      Steps as they complete.
    """
    async for step in self._connection.receive_steps():
      self._steps.append(step)
      if step.type == types.StepType.COMPACTION:
        self._compaction_indices.append(len(self._steps) - 1)
      self._enforce_max_history()
      yield step

  async def chat(self, prompt: types.Content) -> types.ChatResponse:
    """Sends a prompt and returns the complete response.

    This is a convenience wrapper around send() + receive_steps() that
    collects all steps and extracts the final model response.

    Args:
      prompt: The user message to send.

    Returns:
      A ChatResponse with the final text and all steps.
    """
    await self.send(prompt)
    steps = []
    final_response = ""
    structured_output = None
    async for step in self.receive_steps():
      steps.append(step)
      if step.is_complete_response:
        final_response = step.content
      if step.type == types.StepType.FINISH:
        structured_output = step.structured_output
    return types.ChatResponse(
        text=final_response, steps=steps, structured_output=structured_output
    )

  # ---------------------------------------------------------------------------
  # History and state
  # ---------------------------------------------------------------------------

  @property
  def history(self) -> list[types.Step]:
    """Returns all steps received across all turns.

    This is the full, uncompacted transcript. Use compaction_indices to
    identify where the model's context window was compacted.
    """
    return list(self._steps)

  @property
  def last_response(self) -> str:
    """Returns the content of the most recent final model response."""
    for step in reversed(self._steps):
      if step.is_complete_response:
        return step.content
    return ""

  @property
  def turn_count(self) -> int:
    """Returns the number of send() calls made on this conversation."""
    return len(self._turn_start_indices)

  @property
  def compaction_indices(self) -> list[int]:
    """Step indices where the model's context was compacted.

    Each index corresponds to the position of a compaction step in
    the history list. Steps before these indices may no longer be in
    the model's context window, but remain in the full history for
    transcript and debugging.
    """
    return list(self._compaction_indices)

  def clear_history(self) -> None:
    """Clears the accumulated step history, turn indices, and compaction indices.

    Use this to free memory during long-running sessions. The conversation
    remains active — only the recorded history is discarded.
    """
    self._steps.clear()
    self._turn_start_indices.clear()
    self._compaction_indices.clear()

  def _enforce_max_history(self) -> None:
    """Trims history to max_history_size if a limit is set."""
    if self._max_history_size and len(self._steps) > self._max_history_size:
      overflow = len(self._steps) - self._max_history_size
      self._steps = self._steps[overflow:]
      # Adjust indices to account for removed steps.
      self._turn_start_indices = [
          i - overflow
          for i in self._turn_start_indices
          if i >= overflow
      ]
      self._compaction_indices = [
          i - overflow
          for i in self._compaction_indices
          if i >= overflow
      ]

  @property
  def is_idle(self) -> bool:
    """Returns True if the conversation is idle and ready for input."""
    return self._connection.is_idle

  @property
  def conversation_id(self) -> str:
    """Returns the conversation identifier, if one exists."""
    return self._connection.conversation_id

  # ---------------------------------------------------------------------------
  # Lifecycle
  # ---------------------------------------------------------------------------

  async def cancel(self) -> None:
    """Cancels the current turn."""
    await self._connection.cancel()

  async def delete(self) -> None:
    """Deletes this conversation from the backend."""
    await self._connection.delete()

  async def signal_idle(self) -> None:
    """Signals that the conversation is ready to receive input."""
    await self._connection.signal_idle()

  async def wait_for_idle(self) -> None:
    """Blocks until the conversation is idle."""
    await self._connection.wait_for_idle()

  async def wait_for_wakeup(self, timeout: float = 300.0) -> bool:
    """Blocks until the conversation wakes up.

    Args:
      timeout: Maximum seconds to wait.

    Returns:
      True if the conversation woke up, False on timeout.
    """
    return await self._connection.wait_for_wakeup(timeout)

  async def disconnect(self) -> None:
    """Disconnect the conversation's background stream."""
    await self._connection.disconnect()
