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

"""Base interfaces for connections in the Google Antigravity SDK.

A Connection is the SDK's public interface for interacting with an agent
backend, regardless of where the agent runs. Layer 2 APIs (Conversation,
AgentConfig) depend ONLY on this interface — never on transport details.

A ConnectionStrategy knows how to establish a Connection for a specific
backend type and how to tear it down.
"""

import abc
import json
import logging
from typing import Any, AsyncIterator, Callable
import pydantic
from google.antigravity import types


class AgentConfig(abc.ABC, pydantic.BaseModel):
  """Abstract base class for agent configuration.

  Each ConnectionStrategy defines a concrete subclass with the
  config fields it needs. Agent introspects the config type to
  auto-dispatch to the correct strategy factory.
  """

  model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

  system_instructions: str | types.SystemInstructions | None = None
  capabilities: types.CapabilitiesConfig = pydantic.Field(
      default_factory=lambda: types.CapabilitiesConfig(
          enabled_tools=types.BuiltinTools.read_only()
      )
  )
  tools: list[Callable[..., Any]] = pydantic.Field(default_factory=list)
  policies: list[Any] = pydantic.Field(default_factory=list)
  hooks: list[Any] = pydantic.Field(default_factory=list)
  triggers: list[Any] = pydantic.Field(default_factory=list)
  mcp_servers: list[types.McpServerConfig] = pydantic.Field(
      default_factory=list
  )
  workspaces: list[str] = pydantic.Field(default_factory=list)
  conversation_id: str | None = None
  save_dir: str | None = None
  response_schema: dict[str, Any] | type[pydantic.BaseModel] | str | None = None
  skills_paths: list[str] = pydantic.Field(default_factory=list)

  @pydantic.field_validator("response_schema")
  def _validate_schema(cls, v):  # pylint: disable=no-self-argument
    if v is None:
      return None
    if isinstance(v, str):
      try:
        json.loads(v)
        return v
      except json.JSONDecodeError:
        logging.warning(
            "Provided response_schema string is not a valid JSON. Schema"
            " ignored."
        )
        return None
    if isinstance(v, dict):
      return json.dumps(v)
    if isinstance(v, type) and issubclass(v, pydantic.BaseModel):
      return json.dumps(v.model_json_schema())
    logging.warning(
        "Unsupported response_schema format: %s. Schema ignored.", type(v)
    )
    return None

  @abc.abstractmethod
  def create_strategy(
      self,
      *,
      tool_runner: Any,
      hook_runner: Any,
  ) -> "ConnectionStrategy":
    """Creates the ConnectionStrategy for this config.

    The Agent calls this after setting up ToolRunner, HookRunner,
    and policies. The strategy receives the fully-wired runners.

    Args:
      tool_runner: The fully-wired ToolRunner.
      hook_runner: The fully-wired HookRunner.

    Returns:
      A ConnectionStrategy instance configured with the specified runners.
    """
    ...


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

    Returns:
      An async iterator of Step objects.

    Yields:
      Step objects as they occur.
    """
    ...

  async def disconnect(self) -> None:
    """Disconnects the session and releases resources."""
    ...

  async def cancel(self) -> None:
    """Cancels the current turn in progress."""
    pass

  async def delete(self) -> None:
    """Deletes this connection and all associated state from the backend."""
    pass

  async def signal_idle(self) -> None:
    """Signals that the connection is idle and ready to receive input."""
    pass

  async def wait_for_idle(self) -> None:
    """Blocks until the connection becomes idle."""
    pass

  async def wait_for_wakeup(self, timeout: float = 300.0) -> bool:  # pylint: disable=unused-argument
    """Blocks until the connection wakes up or the timeout is reached.

    Args:
      timeout: Maximum seconds to wait.

    Returns:
      True if the connection woke up, False on timeout.
    """
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

    Returns:
      The active Connection object.

    Raises:
      RuntimeError: If the connection has not been established.
    """
    # TODO(kibergus): This method is meant to return a new independent
    # connection, but at the moment most of the implementations return the same
    # connection. This will be rectified in a separate CL.
    ...

  @abc.abstractmethod
  async def __aenter__(self) -> None:
    """Starts the backend and prepares for connections."""
    ...

  @abc.abstractmethod
  async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
    """Tears down the backend and releases all resources.

    Args:
      exc_type: The exception type, if any.
      exc_val: The exception value, if any.
      exc_tb: The traceback, if any.
    """
    ...
