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

"""Layer 1 API for Antigravity SDK."""

import asyncio
import json
import logging
import tempfile
from typing import Any, Callable

import pydantic

from google.antigravity import types
from google.antigravity.connections import connection as connection_module
from google.antigravity.connections import local_connection
from google.antigravity.conversation import conversation
from google.antigravity.hooks import cli
from google.antigravity.hooks import hook_runner
from google.antigravity.hooks import hooks
from google.antigravity.hooks import policy
from google.antigravity.mcp import bridge
from google.antigravity.tools import tool_runner
from google.antigravity.triggers import trigger_runner
from google.antigravity.triggers import triggers as triggers_lib

_Hook = hooks.Hook


class AgentConfig(pydantic.BaseModel):
  """Declarative configuration for an Agent.

  This is a pure data object — no runtime state. It can be reused
  across multiple Agent instances, serialized, or tested in isolation.

  Top-level ``model`` and ``api_key`` are convenience sugar that flow
  into ``gemini_config``.  Do not set both the sugar and the structured
  path — a ``ValueError`` is raised on conflict.

  Attributes:
    system_instructions: Agent instructions. Strings are auto-wrapped in
      TemplatedSystemInstructions during session start.
    gemini_config: Model backend configuration.
    capabilities: Builtin tool enablement. Defaults to read-only.
    tools: Custom Python tools to register.
    policies: Custom policies to enforce.
    hooks: Custom hooks to register.
    triggers: Custom triggers to register.
    mcp_servers: MCP server configurations.
    workspaces: Directory paths to restrict the agent to.
    session_config: Session persistence and resumption configuration.
    response_schema: Optional Pydantic model or JSON schema dict for structured
      output.
    skills_paths: List of paths to skills to load.
    model: Sugar — sets gemini_config.models.default.
    api_key: Sugar — sets gemini_config.api_key.
  """

  model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

  system_instructions: str | types.SystemInstructions
  gemini_config: types.GeminiConfig = pydantic.Field(
      default_factory=types.GeminiConfig
  )
  capabilities: types.CapabilitiesConfig = pydantic.Field(
      default_factory=lambda: types.CapabilitiesConfig(
          enabled_tools=types.BuiltinTools.read_only()
      )
  )
  tools: list[Callable[..., Any]] = pydantic.Field(default_factory=list)
  policies: list[policy.Policy] = pydantic.Field(default_factory=list)
  hooks: list[_Hook] = pydantic.Field(default_factory=list)
  triggers: list[triggers_lib.Trigger] = pydantic.Field(default_factory=list)
  mcp_servers: list[dict[str, Any]] = pydantic.Field(default_factory=list)
  workspaces: list[str] = pydantic.Field(default_factory=list)
  session_config: types.SessionConfig = pydantic.Field(
      default_factory=types.SessionConfig
  )
  response_schema: dict[str, Any] | type[pydantic.BaseModel] | str | None = None
  skills_paths: list[str] = pydantic.Field(default_factory=list)

  # Top-level sugar — flows into gemini_config.
  model: str | None = None
  api_key: str | None = None

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

  @pydantic.model_validator(mode="after")
  def _apply_sugar(self) -> "AgentConfig":
    # Defensive copy: prevent mutation of shared GeminiConfig instances.
    self.gemini_config = self.gemini_config.model_copy(deep=True)

    if self.model is not None:
      if "default" in self.gemini_config.models.model_fields_set:
        raise ValueError(
            "Cannot set both 'model' sugar and "
            "'gemini_config.models.default'. Use one or the other."
        )
      self.gemini_config.models.default = types.ModelEntry(name=self.model)
    if self.api_key is not None:
      if self.gemini_config.api_key is not None:
        raise ValueError(
            "Cannot set both 'api_key' sugar and "
            "'gemini_config.api_key'. Use one or the other."
        )
      self.gemini_config.api_key = self.api_key
    return self


class Agent:
  """High-level Agent API for simplified interaction."""

  def __init__(self, config: AgentConfig):
    """Initializes the Agent.

    Args:
        config: Declarative agent configuration.
    """
    self._config = config
    if config.response_schema:
      self._config.capabilities.finish_tool_schema_json = config.response_schema
    self._strategy = None
    self._conversation = None
    self._conversation_cm = None
    self._tool_runner = None
    self._hook_runner = None
    self._trigger_runner = None
    self._mcp_bridge = None
    self._pending_hooks = list(config.hooks)
    self._pending_triggers = list(config.triggers)

  def register_hook(self, hook: hooks.Hook):
    """Registers a hook by inferring its type."""
    if not self._hook_runner:
      self._pending_hooks.append(hook)
      return
    self._hook_runner.register_hook(hook)

  def register_trigger(self, trigger: triggers_lib.Trigger):
    """Registers a trigger.

    Cannot be called after the agent has started.

    Args:
      trigger: The trigger function to register.

    Raises:
      RuntimeError: If the agent has already started.
    """
    if self._trigger_runner:
      raise RuntimeError(
          "Cannot register triggers after the agent has started."
      )
    self._pending_triggers.append(trigger)

  async def __aenter__(self) -> "Agent":
    """Starts the agent session."""
    logging.info("Starting Agent session")
    try:
      self._tool_runner = tool_runner.ToolRunner(tools=self._config.tools)

      self._hook_runner = hook_runner.HookRunner()

      # Register pending hooks
      for hook in self._pending_hooks:
        self._hook_runner.register_hook(hook)
      self._pending_hooks.clear()

      # Apply policies
      active_policies = list(self._config.policies)
      cfg = self._config.capabilities
      read_only_tools = set(types.BuiltinTools.read_only())
      # enabled_tools and disabled_tools are mutually exclusive
      # (enforced by CapabilitiesConfig validation).
      if cfg.enabled_tools is not None:
        active_tools = set(cfg.enabled_tools)
      elif cfg.disabled_tools is not None:
        active_tools = set(types.BuiltinTools) - set(cfg.disabled_tools)
      else:
        active_tools = set(types.BuiltinTools)
      has_write_tools = bool(active_tools - read_only_tools)
      if has_write_tools and not active_policies:
        raise ValueError(
            "Policies must be provided when non-read-only builtin tools are "
            "enabled to prevent interactive handlers from hanging in "
            "non-interactive contexts."
        )

      if active_policies:
        self._hook_runner.pre_tool_call_decide_hooks.append(
            policy.enforce(active_policies)
        )

      # Connect MCP servers
      if self._config.mcp_servers:
        logging.info("Connecting to MCP servers...")
        self._mcp_bridge = bridge.McpBridge(self._tool_runner)
        for server_cfg in self._config.mcp_servers:
          srv_type = server_cfg.get("type")
          if srv_type == "stdio":
            await self._mcp_bridge.connect_stdio(
                server_cfg["command"], server_cfg.get("args", [])
            )
          elif srv_type == "sse":
            await self._mcp_bridge.connect_sse(
                server_cfg["url"], server_cfg.get("headers")
            )
          else:
            raise ValueError(f"Unknown MCP server type: {srv_type}")

      if isinstance(self._config.system_instructions, str):
        si = types.TemplatedSystemInstructions(
            sections=[
                types.SystemInstructionSection(
                    content=self._config.system_instructions
                )
            ]
        )
      else:
        si = self._config.system_instructions

      session = self._config.session_config
      # Merge top-level workspaces into session_config.
      if self._config.workspaces:
        merged = list(session.workspaces or []) + self._config.workspaces
        session = session.model_copy(update={"workspaces": merged})
      if session.save_dir is None:
        default_dir = tempfile.mkdtemp(prefix="antigravity_")
        logging.info("No save_dir specified; using %s", default_dir)
        session = session.model_copy(update={"save_dir": default_dir})

      self._strategy = local_connection.LocalConnectionStrategy(
          tool_runner=self._tool_runner,
          hook_runner=self._hook_runner,
          gemini_config=self._config.gemini_config,
          system_instructions=si,
          capabilities_config=self._config.capabilities,
          session_config=session,
          skills_paths=self._config.skills_paths,
      )

      logging.info("Starting connection and creating conversation...")
      self._conversation_cm = conversation.Conversation.create(self._strategy)
      self._conversation = await self._conversation_cm.__aenter__()

      # Start triggers via TriggerRunner.
      if self._pending_triggers:
        logging.info("Starting triggers...")
        self._trigger_runner = trigger_runner.TriggerRunner(
            triggers=list(self._pending_triggers),
            connection=self._conversation._connection,
        )
        await self._trigger_runner.start()
        self._pending_triggers.clear()

      return self
    except Exception:
      logging.exception("Failed to start Agent session, cleaning up...")
      await self.__aexit__(None, None, None)
      raise

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    """Stops the agent session."""
    logging.info("Stopping Agent session")
    if self._trigger_runner:
      await self._trigger_runner.stop()
      self._trigger_runner = None
    if self._mcp_bridge:
      await self._mcp_bridge.stop()
    if self._conversation_cm:
      await self._conversation_cm.__aexit__(exc_type, exc_val, exc_tb)

  async def chat(self, prompt: types.Content) -> types.ChatResponse:
    """Sends a prompt and returns the final response."""
    if not self._conversation:
      raise RuntimeError(
          "Agent session not started. Use 'async with Agent(...)'."
      )
    return await self._conversation.chat(prompt)

  async def run_interactive_loop(self):
    """Runs an interactive CLI loop."""
    if not self._conversation:
      raise RuntimeError(
          "Agent session not started. Use 'async with Agent(...)'."
      )

    assert self._hook_runner is not None
    self._hook_runner.on_interaction_hooks.append(cli.AskQuestionHook())
    print("Starting interactive loop. Type 'exit' or 'quit' to end.")
    while True:
      try:
        user_input = await asyncio.to_thread(input, "User: ")
        user_input = user_input.strip()
        if not user_input:
          continue
        if user_input.lower() in ("exit", "quit"):
          print("Goodbye!")
          break

        await self._conversation.send(user_input)

        async for step in self._conversation.receive_steps():
          if step.is_complete_response:
            print(f"Agent: {step.content}")

      except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")
        break
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception("Error in interactive loop: %s", e)
        print(f"Error: {e}")

  @property
  def connection(self) -> connection_module.Connection:
    """Returns the underlying Connection.

    Intended for advanced use cases that need direct transport access.
    Prefer Agent methods for normal interaction — bypassing the Agent
    and Conversation layers skips history tracking and hook dispatch.

    Raises:
      RuntimeError: If the agent session has not been started.
    """
    if not self._conversation:
      raise RuntimeError(
          "Agent session not started. Use 'async with Agent(...)'."
      )
    return self._conversation._connection

  @property
  def conversation_id(self) -> str | None:
    """Returns the conversation identifier assigned by the runtime.

    Available after the session has started and at least one message has
    been exchanged.  Pass this value back via SessionConfig.conversation_id
    to resume from a saved session.  Returns None before the session starts.
    """
    if not self._conversation:
      return None
    return self._conversation.conversation_id or None
