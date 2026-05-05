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

"""Tests for Agent API."""

import os
import unittest
from unittest import mock

from google.antigravity import agent
from google.antigravity import types
from google.antigravity.connections import local_connection
from google.antigravity.conversation import conversation
from google.antigravity.hooks import hooks
from google.antigravity.hooks import policy


class AgentTest(unittest.IsolatedAsyncioTestCase):

  @mock.patch(
      "google.antigravity.agent."
      "local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_lifecycle(self, mock_conv_create, mock_strategy_class):

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_class.return_value = mock_strategy_instance

    mock_conversation = mock.MagicMock(spec=conversation.Conversation)
    mock_cm = mock.AsyncMock()
    mock_cm.__aenter__.return_value = mock_conversation
    mock_conv_create.return_value = mock_cm

    config = agent.AgentConfig(system_instructions="test")
    async with agent.Agent(config) as ag:
      self.assertEqual(ag._conversation, mock_conversation)

  @mock.patch(
      "google.antigravity.agent."
      "local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_chat(self, mock_conv_create, mock_strategy_class):

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    mock_conversation = mock.MagicMock(spec=conversation.Conversation)
    mock_cm = mock.AsyncMock()
    mock_cm.__aenter__.return_value = mock_conversation
    mock_conv_create.return_value = mock_cm

    mock_conversation.chat = mock.AsyncMock(
        return_value=types.ChatResponse(
            text="Hello back",
            steps=[types.Step(is_complete_response=True, content="Hello back")],
        )
    )

    config = agent.AgentConfig(system_instructions="test")
    async with agent.Agent(config) as ag:
      response = await ag.chat("Hello")
      self.assertEqual(response.text, "Hello back")
      self.assertEqual(len(response.steps), 1)
      mock_conversation.chat.assert_called_once_with("Hello")

  @mock.patch.object(local_connection, "LocalConnectionStrategy", autospec=True)
  @mock.patch.object(conversation.Conversation, "create", autospec=True)
  async def test_agent_chat_multimodal_input(
      self, mock_conv_create, mock_strategy_class
  ):
    """Verifies that the Agent public API method accepts multimodal Content payloads."""
    mock_strategy_instance = mock_strategy_class.return_value
    mock_strategy_instance.stop = mock.AsyncMock()

    mock_conversation = mock.MagicMock(spec=conversation.Conversation)
    mock_cm = mock.AsyncMock()
    mock_cm.__aenter__.return_value = mock_conversation
    mock_conv_create.return_value = mock_cm

    mock_conversation.chat = mock.AsyncMock(
        return_value=types.ChatResponse(
            text="Analyzed image content",
            steps=[
                types.Step(
                    is_final_response=True, content="Analyzed image content"
                )
            ],
        )
    )

    config = agent.AgentConfig(system_instructions="test")
    async with agent.Agent(config) as ag:
      multimodal_prompt = [
          "Look at this:",
          types.Part(
              inline_data=types.Blob(mime_type="image/png", data=b"png_bytes")
          ),
      ]
      response = await ag.chat(multimodal_prompt)
      self.assertEqual(response.text, "Analyzed image content")
      mock_conversation.chat.assert_called_once_with(multimodal_prompt)

  @mock.patch(
      "google.antigravity.agent."
      "local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_read_only_default(
      self, mock_conv_create, mock_strategy_class
  ):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    config = agent.AgentConfig(system_instructions="test")
    async with agent.Agent(config):
      _, kwargs = mock_strategy_class.call_args
      capabilities_config = kwargs.get("capabilities_config")
      self.assertIsNotNone(capabilities_config)
      self.assertEqual(
          capabilities_config.enabled_tools, types.BuiltinTools.read_only()
      )

  @mock.patch(
      "google.antigravity.agent."
      "local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_requires_policies_in_write_mode(
      self, mock_conv_create, mock_strategy_class
  ):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    config = agent.AgentConfig(
        system_instructions="test",
        capabilities=types.CapabilitiesConfig(),
    )
    with self.assertRaises(ValueError):
      async with agent.Agent(config):
        pass

  @mock.patch(
      "google.antigravity.agent."
      "local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_policy_guard_explicit_write_tool(
      self, mock_conv_create, mock_strategy_class
  ):
    """Guard fires when enabled_tools includes a non-read-only tool."""
    del mock_conv_create
    mock_strategy_class.return_value = mock.MagicMock(stop=mock.AsyncMock())
    config = agent.AgentConfig(
        system_instructions="test",
        capabilities=types.CapabilitiesConfig(
            enabled_tools=[types.BuiltinTools.RUN_COMMAND],
        ),
    )
    with self.assertRaises(ValueError):
      async with agent.Agent(config):
        pass

  @mock.patch(
      "google.antigravity.agent."
      "local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_policy_guard_explicit_all_tools(
      self, mock_conv_create, mock_strategy_class
  ):
    """Guard fires when all tools are listed explicitly."""
    del mock_conv_create
    mock_strategy_class.return_value = mock.MagicMock(stop=mock.AsyncMock())
    config = agent.AgentConfig(
        system_instructions="test",
        capabilities=types.CapabilitiesConfig(
            enabled_tools=list(types.BuiltinTools),
        ),
    )
    with self.assertRaises(ValueError):
      async with agent.Agent(config):
        pass

  @mock.patch(
      "google.antigravity.agent."
      "local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_policy_guard_empty_disabled_tools(
      self, mock_conv_create, mock_strategy_class
  ):
    """Guard fires when disabled_tools=[] (= all tools enabled)."""
    del mock_conv_create
    mock_strategy_class.return_value = mock.MagicMock(stop=mock.AsyncMock())
    config = agent.AgentConfig(
        system_instructions="test",
        capabilities=types.CapabilitiesConfig(disabled_tools=[]),
    )
    with self.assertRaises(ValueError):
      async with agent.Agent(config):
        pass

  @mock.patch(
      "google.antigravity.agent."
      "local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_policy_guard_read_only_explicit_passes(
      self, mock_conv_create, mock_strategy_class
  ):
    """No guard when only read-only tools are explicitly enabled."""
    del mock_conv_create
    mock_strategy_class.return_value = mock.MagicMock(stop=mock.AsyncMock())
    config = agent.AgentConfig(
        system_instructions="test",
        capabilities=types.CapabilitiesConfig(
            enabled_tools=types.BuiltinTools.read_only(),
        ),
    )
    async with agent.Agent(config):
      pass  # Should not raise.

  @mock.patch(
      "google.antigravity.agent."
      "local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_policy_guard_write_tools_with_policy_passes(
      self, mock_conv_create, mock_strategy_class
  ):
    """No guard when write tools are enabled AND policies are provided."""
    del mock_conv_create
    mock_strategy_class.return_value = mock.MagicMock(stop=mock.AsyncMock())
    config = agent.AgentConfig(
        system_instructions="test",
        capabilities=types.CapabilitiesConfig(),
        policies=[policy.deny("*")],
    )
    async with agent.Agent(config):
      pass  # Should not raise.

  @mock.patch(
      "google.antigravity.agent."
      "local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_register_hook(
      self, mock_conv_create, mock_strategy_class
  ):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    class MyPreTurnHook(hooks.PreTurnHook):

      async def run(self, context, data):
        return types.HookResult(allow=True)

    my_hook = MyPreTurnHook()

    # Test constructor registration
    config = agent.AgentConfig(system_instructions="test", hooks=[my_hook])
    async with agent.Agent(config) as ag:
      self.assertIn(my_hook, ag._hook_runner.pre_turn_hooks)

    # Test dynamic registration
    config = agent.AgentConfig(system_instructions="test")
    async with agent.Agent(config) as ag:
      ag.register_hook(my_hook)
      self.assertIn(my_hook, ag._hook_runner.pre_turn_hooks)

  @mock.patch(
      "google.antigravity.agent."
      "local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  @mock.patch(
      "google.antigravity.agent."
      "trigger_runner.TriggerRunner"
  )
  async def test_agent_register_trigger(
      self,
      mock_trigger_runner_class,
      mock_conv_create,
      mock_strategy_class,
  ):

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    mock_conversation = mock.MagicMock(spec=conversation.Conversation)
    mock_conversation._connection = mock.MagicMock()

    mock_cm = mock.AsyncMock()
    mock_cm.__aenter__.return_value = mock_conversation
    mock_conv_create.return_value = mock_cm

    mock_runner_instance = mock.AsyncMock()
    mock_trigger_runner_class.return_value = mock_runner_instance

    async def my_trigger(ctx):
      del ctx  # Unused.
      pass

    # Test constructor registration: TriggerRunner started with trigger.
    config = agent.AgentConfig(
        system_instructions="test", triggers=[my_trigger]
    )
    async with agent.Agent(config):
      mock_trigger_runner_class.assert_called_once()
      call_kwargs = mock_trigger_runner_class.call_args[1]
      self.assertEqual(call_kwargs["triggers"], [my_trigger])
      mock_runner_instance.start.assert_called_once()

    # TriggerRunner.stop() called during __aexit__.
    mock_runner_instance.stop.assert_called_once()

    mock_trigger_runner_class.reset_mock()
    mock_runner_instance.reset_mock()

    # Test dynamic registration before start.
    config = agent.AgentConfig(system_instructions="test")
    ag = agent.Agent(config)
    ag.register_trigger(my_trigger)
    async with ag:
      mock_trigger_runner_class.assert_called_once()
      call_kwargs = mock_trigger_runner_class.call_args[1]
      self.assertEqual(call_kwargs["triggers"], [my_trigger])
      mock_runner_instance.start.assert_called_once()

  @mock.patch(
      "google.antigravity.agent."
      "local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_register_hook_before_start(
      self, mock_conv_create, mock_strategy_class
  ):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    class MyPreTurnHook(hooks.PreTurnHook):

      async def run(self, context, data):
        return types.HookResult(allow=True)

    my_hook = MyPreTurnHook()

    config = agent.AgentConfig(system_instructions="test")
    ag = agent.Agent(config)
    ag.register_hook(my_hook)
    self.assertIn(my_hook, ag._pending_hooks)

    async with ag:
      self.assertIn(my_hook, ag._hook_runner.pre_turn_hooks)
      self.assertEqual(len(ag._pending_hooks), 0)

  @mock.patch(
      "google.antigravity.agent."
      "local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_register_trigger_after_start(
      self, mock_conv_create, mock_strategy_class
  ):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    async def my_trigger(_):
      pass

    config = agent.AgentConfig(
        system_instructions="test", triggers=[my_trigger]
    )
    async with agent.Agent(config) as ag:
      with self.assertRaises(RuntimeError):
        ag.register_trigger(my_trigger)

  @mock.patch(
      "google.antigravity.agent."
      "local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_with_policies(
      self, mock_conv_create, mock_strategy_class
  ):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    my_policy = policy.allow("some_tool")

    config = agent.AgentConfig(
        system_instructions="test",
        capabilities=types.CapabilitiesConfig(),
        policies=[my_policy],
    )
    async with agent.Agent(config) as ag:
      self.assertEqual(len(ag._hook_runner.pre_tool_call_decide_hooks), 1)

  @mock.patch(
      "google.antigravity.agent."
      "local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_write_mode_with_policies(
      self, mock_conv_create, mock_strategy_class
  ):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    my_policy = policy.allow("some_tool")

    config = agent.AgentConfig(
        system_instructions="test",
        capabilities=types.CapabilitiesConfig(),
        policies=[my_policy],
    )
    async with agent.Agent(config):
      _, kwargs = mock_strategy_class.call_args
      capabilities_config = kwargs.get("capabilities_config")
      self.assertIsNotNone(capabilities_config)
      self.assertNotEqual(
          capabilities_config.enabled_tools, types.BuiltinTools.read_only()
      )

  @mock.patch(
      "google.antigravity.agent."
      "local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_mcp_server_unknown_type(
      self, mock_conv_create, mock_strategy_class
  ):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    mcp_servers = [{"type": "unknown_type"}]

    with self.assertRaises(ValueError):
      config = agent.AgentConfig(
          system_instructions="test", mcp_servers=mcp_servers
      )
      async with agent.Agent(config):
        pass

  async def test_agent_chat_before_start(self):
    ag = agent.Agent(agent.AgentConfig(system_instructions="test"))
    with self.assertRaises(RuntimeError):
      await ag.chat("hello")

  async def test_agent_connection_before_start(self):
    ag = agent.Agent(agent.AgentConfig(system_instructions="test"))
    with self.assertRaises(RuntimeError):
      _ = ag.connection

  async def test_agent_run_interactive_loop_before_start(self):
    ag = agent.Agent(agent.AgentConfig(system_instructions="test"))
    with self.assertRaises(RuntimeError):
      await ag.run_interactive_loop()

  @mock.patch(
      "google.antigravity.agent."
      "local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_api_key_env(self, mock_conv_create, mock_strategy_class):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    with mock.patch.dict("os.environ", {}, clear=True):
      config = agent.AgentConfig(system_instructions="test", api_key="test_key")
      async with agent.Agent(config):
        self.assertIsNone(os.environ.get("GEMINI_API_KEY"))
        _, kwargs = mock_strategy_class.call_args
        gemini_config = kwargs.get("gemini_config")
        self.assertIsNotNone(gemini_config)
        self.assertEqual(gemini_config.api_key, "test_key")

  @mock.patch(
      "google.antigravity.agent."
      "local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_model_sugar_flows_to_strategy(
      self, mock_conv_create, mock_strategy_class
  ):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    config = agent.AgentConfig(
        system_instructions="test", model="gemini-2.5-pro"
    )
    async with agent.Agent(config):
      _, kwargs = mock_strategy_class.call_args
      gemini_config = kwargs.get("gemini_config")
      self.assertIsNotNone(gemini_config)
      self.assertEqual(gemini_config.models.default.name, "gemini-2.5-pro")

  @mock.patch(
      "google.antigravity.agent.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_with_system_instructions_object(
      self, mock_conv_create, mock_strategy_class
  ):
    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    si_obj = types.CustomSystemInstructions(text="custom si")
    config = agent.AgentConfig(system_instructions=si_obj)
    async with agent.Agent(config):
      _, kwargs = mock_strategy_class.call_args
      si = kwargs.get("system_instructions")
      self.assertEqual(si, si_obj)

  @mock.patch(
      "google.antigravity.agent."
      "local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_with_session_config(
      self, mock_conv_create, mock_strategy_class
  ):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    sc = types.SessionConfig(
        conversation_id="resume-id",
        save_dir="/state",
        workspaces=["/path/1", "/path/2"],
    )
    config = agent.AgentConfig(system_instructions="test", session_config=sc)
    async with agent.Agent(config) as _:
      _, kwargs = mock_strategy_class.call_args
      session_config = kwargs.get("session_config")
      self.assertEqual(session_config, sc)
      self.assertEqual(session_config.conversation_id, "resume-id")
      self.assertEqual(session_config.save_dir, "/state")
      self.assertEqual(session_config.workspaces, ["/path/1", "/path/2"])

  @mock.patch(
      "google.antigravity.agent."
      "local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_with_skills_paths(
      self, mock_conv_create, mock_strategy_class
  ):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    skills_paths = ["/path/1", "/path/2"]
    config = agent.AgentConfig(
        system_instructions="test", skills_paths=skills_paths
    )
    async with agent.Agent(config) as _:
      _, kwargs = mock_strategy_class.call_args
      sp = kwargs.get("skills_paths")
      self.assertEqual(sp, skills_paths)

  @mock.patch(
      "google.antigravity.agent."
      "local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  @mock.patch("google.antigravity.agent.bridge.McpBridge")
  async def test_agent_mcp_servers(
      self,
      mock_mcp_bridge,
      mock_conv_create,
      mock_strategy_class,
  ):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    mock_bridge_instance = mock.MagicMock()
    mock_bridge_instance.connect_stdio = mock.AsyncMock()
    mock_bridge_instance.connect_sse = mock.AsyncMock()
    mock_bridge_instance.stop = mock.AsyncMock()
    mock_mcp_bridge.return_value = mock_bridge_instance

    mcp_servers = [
        {"type": "stdio", "command": "python3", "args": ["server.py"]},
        {"type": "sse", "url": "http://localhost:8000/sse"},
    ]

    config = agent.AgentConfig(
        system_instructions="test", mcp_servers=mcp_servers
    )
    async with agent.Agent(config) as ag:
      mock_mcp_bridge.assert_called_once_with(ag._tool_runner)
      mock_bridge_instance.connect_stdio.assert_called_once_with(
          "python3", ["server.py"]
      )
      mock_bridge_instance.connect_sse.assert_called_once_with(
          "http://localhost:8000/sse", None
      )

    mock_bridge_instance.stop.assert_called_once()

  @mock.patch(
      "google.antigravity.agent."
      "local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  @mock.patch("asyncio.to_thread")
  async def test_agent_run_interactive_loop(
      self, mock_to_thread, mock_conv_create, mock_strategy_class
  ):
    mock_strategy_instance = mock.MagicMock()

    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    mock_conversation = mock.MagicMock(spec=conversation.Conversation)
    mock_conversation.send = mock.AsyncMock()

    async def mock_receive_steps():
      yield types.Step(is_complete_response=True, content="Agent response")

    mock_conversation.receive_steps = mock_receive_steps

    mock_cm = mock.AsyncMock()
    mock_cm.__aenter__.return_value = mock_conversation
    mock_conv_create.return_value = mock_cm

    # Mock input to return '', 'hello' then 'exit'
    mock_to_thread.side_effect = ["", "hello", "exit"]

    config = agent.AgentConfig(system_instructions="test")
    async with agent.Agent(config) as ag:
      with mock.patch("builtins.print") as mock_print:
        await ag.run_interactive_loop()

    mock_conversation.send.assert_called_once_with("hello")
    mock_print.assert_any_call("Agent: Agent response")

  @mock.patch(
      "google.antigravity.agent."
      "local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  @mock.patch("asyncio.to_thread")
  async def test_agent_run_interactive_loop_interrupt(
      self, mock_to_thread, mock_conv_create, mock_strategy_class
  ):
    del mock_conv_create  # Unused.
    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    mock_to_thread.side_effect = KeyboardInterrupt()

    config = agent.AgentConfig(system_instructions="test")
    async with agent.Agent(config) as ag:
      with mock.patch("builtins.print") as mock_print:
        await ag.run_interactive_loop()

    mock_print.assert_any_call("\nGoodbye!")

  @mock.patch(
      "google.antigravity.agent."
      "local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  @mock.patch("asyncio.to_thread")
  async def test_agent_run_interactive_loop_exception(
      self, mock_to_thread, mock_conv_create, mock_strategy_class
  ):
    del mock_conv_create  # Unused.
    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    mock_to_thread.side_effect = [ValueError("Fail"), "exit"]

    config = agent.AgentConfig(system_instructions="test")
    async with agent.Agent(config) as ag:
      with mock.patch("builtins.print") as mock_print:
        await ag.run_interactive_loop()

    mock_print.assert_any_call("Error: Fail")

  @mock.patch(
      "google.antigravity.agent."
      "local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_connection_after_start(
      self, mock_conv_create, mock_strategy_class
  ):
    mock_strategy_instance = mock.MagicMock()

    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    mock_conversation = mock.MagicMock(spec=conversation.Conversation)
    mock_conversation._connection = mock.MagicMock()
    mock_cm = mock.AsyncMock()
    mock_cm.__aenter__.return_value = mock_conversation
    mock_conv_create.return_value = mock_cm

    config = agent.AgentConfig(system_instructions="test")
    async with agent.Agent(config) as ag:
      conn = ag.connection
      self.assertEqual(conn, mock_conversation._connection)


class AgentConfigTest(unittest.TestCase):
  """Tests for AgentConfig sugar, conflict guards, and defensive copy."""

  def test_sugar_model_flows_to_gemini_config(self):
    """Verifies model sugar flows to gemini_config.models.default.name."""
    config = agent.AgentConfig(
        system_instructions="test", model="gemini-2.5-pro"
    )
    self.assertEqual(config.gemini_config.models.default.name, "gemini-2.5-pro")

  def test_sugar_api_key_flows_to_gemini_config(self):
    """Verifies api_key sugar flows to gemini_config.api_key."""
    config = agent.AgentConfig(system_instructions="test", api_key="my-key")
    self.assertEqual(config.gemini_config.api_key, "my-key")

  def test_conflict_model_raises(self):
    """Verifies ValueError when both model sugar and structured config are set."""
    with self.assertRaises(ValueError):
      agent.AgentConfig(
          system_instructions="test",
          model="gemini-2.5-pro",
          gemini_config=types.GeminiConfig(
              models=types.ModelConfig(
                  default=types.ModelEntry(name="different-model"),
              ),
          ),
      )

  def test_conflict_api_key_raises(self):
    """Verifies ValueError when both api_key sugar and gemini_config.api_key are set."""
    with self.assertRaises(ValueError):
      agent.AgentConfig(
          system_instructions="test",
          api_key="sugar-key",
          gemini_config=types.GeminiConfig(api_key="config-key"),
      )

  def test_defensive_copy(self):
    """Verifies shared GeminiConfig is not cross-contaminated."""
    shared = types.GeminiConfig()
    config1 = agent.AgentConfig(
        system_instructions="test",
        gemini_config=shared,
        model="model-a",
    )
    config2 = agent.AgentConfig(
        system_instructions="test",
        gemini_config=shared,
        model="model-b",
    )
    self.assertEqual(config1.gemini_config.models.default.name, "model-a")
    self.assertEqual(config2.gemini_config.models.default.name, "model-b")
    self.assertEqual(shared.models.default.name, types.DEFAULT_MODEL)

  def test_defaults(self):
    """Verifies AgentConfig defaults: read-only capabilities, default model."""
    config = agent.AgentConfig(system_instructions="test")
    self.assertEqual(
        config.capabilities.enabled_tools,
        types.BuiltinTools.read_only(),
    )
    self.assertEqual(
        config.gemini_config.models.default.name, types.DEFAULT_MODEL
    )
    self.assertIsNone(config.gemini_config.api_key)

  def test_model_sugar_does_not_clobber_image_generation(self):
    """Verifies model sugar only sets default slot, not image_generation."""
    config = agent.AgentConfig(
        system_instructions="test", model="custom-chat-model"
    )
    self.assertEqual(
        config.gemini_config.models.default.name, "custom-chat-model"
    )
    self.assertEqual(
        config.gemini_config.models.image_generation.name,
        types.DEFAULT_IMAGE_GENERATION_MODEL,
    )

  def test_conflict_model_with_gemini_config_no_model(self):
    """Verifies no conflict when gemini_config has no explicit default."""
    config = agent.AgentConfig(
        system_instructions="test",
        model="custom-model",
        gemini_config=types.GeminiConfig(api_key="key-only"),
    )
    self.assertEqual(config.gemini_config.models.default.name, "custom-model")
    self.assertEqual(config.gemini_config.api_key, "key-only")

  @mock.patch.object(
      agent.local_connection, "LocalConnectionStrategy", autospec=True
  )
  @mock.patch.object(conversation.Conversation, "create", autospec=True)
  async def test_agent_with_response_schema(
      self, mock_conv_create, mock_strategy_class
  ):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock_strategy_class.return_value
    mock_strategy_instance.stop = mock.AsyncMock()

    schema_dict = {"properties": {"field": {"type": "string"}}}
    config = agent.AgentConfig(
        system_instructions="test", response_schema=schema_dict
    )
    async with agent.Agent(config) as _:
      _, kwargs = mock_strategy_class.call_args
      capabilities_config = kwargs.get("capabilities_config")
      self.assertIsNotNone(capabilities_config)
      self.assertEqual(
          capabilities_config.finish_tool_schema_json,
          '{"properties": {"field": {"type": "string"}}}',
      )

  def test_conversation_id_returns_none_before_session(self):
    """Verifies conversation_id is None before the session starts."""
    a = agent.Agent(agent.AgentConfig(system_instructions="test"))
    self.assertIsNone(a.conversation_id)

  def test_conversation_id_returns_value_after_session(self):
    """Verifies conversation_id returns the runtime-assigned ID."""
    a = agent.Agent(agent.AgentConfig(system_instructions="test"))
    mock_conv = mock.MagicMock()
    mock_conv.conversation_id = "test-conv-123"
    a._conversation = mock_conv
    self.assertEqual(a.conversation_id, "test-conv-123")


if __name__ == "__main__":
  unittest.main()
