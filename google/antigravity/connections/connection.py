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

"""Base interfaces for connections in the Antigravity SDK.

A Connection is the SDK's public interface for interacting with an agent
backend, regardless of where the agent runs. Layer 2 APIs (Conversation,
AgentConfig) depend ONLY on this interface — never on transport details.

A ConnectionStrategy knows how to establish a Connection for a specific
backend type and how to tear it down.
"""


import abc
from typing import Any, AsyncIterator
from google.antigravity import types


class Connection(abc.ABC):
  """A live session with an agent backend.

  This is the common contract that all connection types implement.
  Layer 2 APIs depend only on this interface.
  """

  @property
  def is_idle(self) -> bool:
    """Returns True if the connection is idle and ready for input."""
    return True

  @property
  def conversation_id(self) -> str:
    """Returns the conversation identifier, if one exists."""
    return ""

  @abc.abstractmethod
  async def send(self, prompt: types.Content | None, **kwargs: Any) -> None:
    """Sends a prompt to the agent.

    Args:
      prompt: The user message to send.
      **kwargs: Strategy-specific options (model overrides, media, etc.).
    """
    ...

  @abc.abstractmethod
  def receive_steps(self) -> AsyncIterator[types.Step]:
    """Receives steps as they complete from the agent.

    Yields Step objects representing agent actions. The exact fields populated
    depend on the backend, but all steps conform to the Step model.

    Yields:
      Step objects.
    """
    ...

  @abc.abstractmethod
  async def disconnect(self) -> None:
    """Disconnects the session and releases resources."""
    ...

  async def cancel(self) -> None:
    """Cancels the current turn."""
    pass

  async def delete(self) -> None:
    """Deletes this connection from the backend."""
    pass

  async def signal_idle(self) -> None:
    """Signals that the connection is idle and ready to receive input."""
    pass

  async def wait_for_idle(self) -> None:
    """Blocks until the connection becomes idle."""
    pass

  async def wait_for_wakeup(self, timeout: float = 300.0) -> bool:  # pylint: disable=unused-argument
    """Blocks until the connection wakes up."""
    return False

  async def send_tool_results(self, results: list[types.ToolResult]) -> None:
    """Sends tool execution results back to the agent.

    Each connection strategy serializes the results into the backend
    wire format.

    Args:
      results: A list of ToolResult objects.
    """
    pass

  @abc.abstractmethod
  async def send_trigger_notification(self, content: str) -> None:
    """Sends a trigger message to the agent.

    Args:
      content: The trigger message content.
    """
    ...


class ConnectionStrategy(abc.ABC):
  """Strategy for establishing a Connection to an agent backend.

  Each backend type (local, Interactions API, cloud agent) provides its own
  ConnectionStrategy implementation that handles process management,
  transport setup, authentication, and health checking.
  """

  @abc.abstractmethod
  def connect(self) -> Connection:
    """Returns the established Connection.

    Raises:
      RuntimeError: If the connection has not been established.
    """
    # TODO(kibergus): This method is meant to return a new independent
    # connection, but at the moment most of the implementations return the same
    # connection. This will be rectified in a separate CL.
    ...

  @abc.abstractmethod
  async def __aenter__(self) -> None:
    """Starts the backend."""
    ...

  @abc.abstractmethod
  async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
    """Tears down the backend and releases all resources."""
    ...
