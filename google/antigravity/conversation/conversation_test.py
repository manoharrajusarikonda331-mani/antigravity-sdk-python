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

"""Tests for the Conversation stateful session layer.

Validates history accumulation, compaction tracking, chat() convenience,
state introspection, and clean delegation to the Connection ABC.
"""

import unittest
from unittest import mock

from google.antigravity import types
from google.antigravity.connections import connection
from google.antigravity.conversation import conversation


def _make_step(
    content: str = "",
    *,
    step_index: int = 0,
    step_type: types.StepType = types.StepType.MODEL_RESPONSE,
    source: types.StepSource = types.StepSource.MODEL,
    status: types.StepStatus = types.StepStatus.DONE,
    is_final: bool = False,
) -> types.Step:
  """Creates a Step with sensible defaults for testing."""
  return types.Step(
      id=str(step_index),
      step_index=step_index,
      type=step_type,
      source=source,
      status=status,
      content=content,
      is_complete_response=is_final,
  )


class ConversationCreateTest(unittest.IsolatedAsyncioTestCase):
  """Validates the Conversation.create factory."""

  async def test_create_delegates_to_strategy(self):
    """Verifies that create enters the strategy context and calls connect."""
    mock_connection = mock.MagicMock(spec=connection.Connection)
    mock_strategy = mock.AsyncMock(spec=connection.ConnectionStrategy)
    mock_strategy.connect.return_value = mock_connection

    async with conversation.Conversation.create(mock_strategy) as conv:
      self.assertIsInstance(conv, conversation.Conversation)

    mock_strategy.connect.assert_called_once()


class ConversationSendTest(unittest.IsolatedAsyncioTestCase):
  """Validates send behavior including idle-wait and turn tracking."""

  async def test_send_waits_for_idle_then_delegates(self):
    """Verifies send calls wait_for_idle before delegating to connection."""
    mock_connection = mock.AsyncMock(spec=connection.Connection)
    conv = conversation.Conversation(mock_connection)

    await conv.send("hello")

    mock_connection.wait_for_idle.assert_called_once()
    mock_connection.send.assert_called_once_with("hello")

  async def test_send_multimodal_input(self):
    """Verifies that send accepts multimodal Content payloads and delegates to connection."""
    mock_connection = mock.AsyncMock(spec=connection.Connection)
    conv = conversation.Conversation(mock_connection)

    multimodal_prompt = [
        "Context string",
        types.Document(mime_type="application/pdf", data=b"pdf"),
    ]
    await conv.send(multimodal_prompt)

    mock_connection.wait_for_idle.assert_called_once()
    mock_connection.send.assert_called_once_with(multimodal_prompt)

  async def test_send_records_turn_boundary(self):
    """Verifies each send records a turn boundary index in the history."""
    mock_connection = mock.AsyncMock(spec=connection.Connection)
    conv = conversation.Conversation(mock_connection)

    await conv.send("first")
    self.assertEqual(conv.turn_count, 1)

    await conv.send("second")
    self.assertEqual(conv.turn_count, 2)


class ConversationReceiveStepsTest(unittest.IsolatedAsyncioTestCase):
  """Validates receive_steps delegation and history accumulation."""

  async def test_receive_steps_yields_from_connection(self):
    """Verifies steps are yielded from the underlying connection."""
    step1 = _make_step("step1", step_index=1)
    step2 = _make_step("step2", step_index=2)

    mock_connection = mock.MagicMock(spec=connection.Connection)

    async def mock_generator():
      yield step1
      yield step2

    mock_connection.receive_steps.return_value = mock_generator()
    conv = conversation.Conversation(mock_connection)

    steps = []
    async for step in conv.receive_steps():
      steps.append(step)

    self.assertEqual(len(steps), 2)
    self.assertEqual(steps[0].content, "step1")
    self.assertEqual(steps[1].content, "step2")

  async def test_receive_steps_accumulates_history(self):
    """Verifies steps are recorded in history as they are received."""
    step1 = _make_step("a", step_index=1)
    step2 = _make_step("b", step_index=2)

    mock_connection = mock.MagicMock(spec=connection.Connection)

    async def mock_generator():
      yield step1
      yield step2

    mock_connection.receive_steps.return_value = mock_generator()
    conv = conversation.Conversation(mock_connection)

    # History starts empty.
    self.assertEqual(conv.history, [])

    async for _ in conv.receive_steps():
      pass

    self.assertEqual(len(conv.history), 2)
    self.assertEqual(conv.history[0].content, "a")
    self.assertEqual(conv.history[1].content, "b")

  async def test_history_returns_copy(self):
    """Verifies history returns a copy, not a reference to internal state."""
    step = _make_step("x", step_index=1)
    mock_connection = mock.MagicMock(spec=connection.Connection)

    async def mock_generator():
      yield step

    mock_connection.receive_steps.return_value = mock_generator()
    conv = conversation.Conversation(mock_connection)

    async for _ in conv.receive_steps():
      pass

    history = conv.history
    history.clear()
    self.assertEqual(len(conv.history), 1)

  async def test_compaction_step_tracked(self):
    """Verifies compaction steps are recorded in compaction_indices."""
    regular = _make_step("text", step_index=1)
    compaction = _make_step(
        "compacted",
        step_index=2,
        step_type=types.StepType.COMPACTION,
    )
    after = _make_step("more", step_index=3)

    mock_connection = mock.MagicMock(spec=connection.Connection)

    async def mock_generator():
      yield regular
      yield compaction
      yield after

    mock_connection.receive_steps.return_value = mock_generator()
    conv = conversation.Conversation(mock_connection)

    async for _ in conv.receive_steps():
      pass

    # Compaction step is at index 1 in the history list.
    self.assertEqual(conv.compaction_indices, [1])
    # Full history is preserved.
    self.assertEqual(len(conv.history), 3)

  async def test_compaction_indices_returns_copy(self):
    """Verifies compaction_indices returns a copy."""
    compaction = _make_step(
        "",
        step_index=1,
        step_type=types.StepType.COMPACTION,
    )
    mock_connection = mock.MagicMock(spec=connection.Connection)

    async def mock_generator():
      yield compaction

    mock_connection.receive_steps.return_value = mock_generator()
    conv = conversation.Conversation(mock_connection)

    async for _ in conv.receive_steps():
      pass

    indices = conv.compaction_indices
    indices.clear()
    self.assertEqual(len(conv.compaction_indices), 1)


class ConversationHistoryTest(unittest.IsolatedAsyncioTestCase):
  """Validates history accessors across multiple turns."""

  async def test_last_response_returns_most_recent_final(self):
    """Verifies last_response returns the content of the most recent final step."""
    mock_connection = mock.MagicMock(spec=connection.Connection)

    async def gen1():
      yield _make_step("first answer", step_index=1, is_final=True)

    async def gen2():
      yield _make_step("second answer", step_index=2, is_final=True)

    mock_connection.receive_steps.return_value = gen1()
    conv = conversation.Conversation(mock_connection)

    mock_connection.wait_for_idle = mock.AsyncMock()
    mock_connection.send = mock.AsyncMock()
    await conv.send("q1")
    async for _ in conv.receive_steps():
      pass

    self.assertEqual(conv.last_response, "first answer")

    mock_connection.receive_steps.return_value = gen2()
    await conv.send("q2")
    async for _ in conv.receive_steps():
      pass

    self.assertEqual(conv.last_response, "second answer")

  async def test_last_response_empty_when_no_final(self):
    """Verifies last_response returns empty string when no final step exists."""
    conv = conversation.Conversation(mock.MagicMock(spec=connection.Connection))
    self.assertEqual(conv.last_response, "")

  async def test_multi_turn_history_accumulates(self):
    """Verifies history accumulates across multiple send/receive cycles."""
    mock_connection = mock.MagicMock(spec=connection.Connection)

    async def gen1():
      yield _make_step("a", step_index=1)

    async def gen2():
      yield _make_step("b", step_index=2)
      yield _make_step("c", step_index=3)

    mock_connection.wait_for_idle = mock.AsyncMock()
    mock_connection.send = mock.AsyncMock()
    conv = conversation.Conversation(mock_connection)

    mock_connection.receive_steps.return_value = gen1()
    await conv.send("turn1")
    async for _ in conv.receive_steps():
      pass

    mock_connection.receive_steps.return_value = gen2()
    await conv.send("turn2")
    async for _ in conv.receive_steps():
      pass

    self.assertEqual(len(conv.history), 3)
    self.assertEqual(conv.turn_count, 2)


class ConversationChatTest(unittest.IsolatedAsyncioTestCase):
  """Validates the chat() convenience method."""

  async def test_chat_returns_response_with_text_and_steps(self):
    """Verifies chat() collects steps and returns the final response text."""
    tool_step = _make_step("", step_index=1, step_type=types.StepType.TOOL_CALL)
    final_step = _make_step("the answer", step_index=2, is_final=True)

    mock_connection = mock.MagicMock(spec=connection.Connection)
    mock_connection.wait_for_idle = mock.AsyncMock()
    mock_connection.send = mock.AsyncMock()

    async def mock_generator():
      yield tool_step
      yield final_step

    mock_connection.receive_steps.return_value = mock_generator()
    conv = conversation.Conversation(mock_connection)

    result = await conv.chat("question")

    self.assertIsInstance(result, types.ChatResponse)
    self.assertEqual(result.text, "the answer")
    self.assertEqual(len(result.steps), 2)

  async def test_chat_multimodal_input(self):
    """Verifies that the chat convenience wrapper accepts and forwards multimodal Content prompts."""
    final_step = _make_step("image analysis done", step_index=1, is_final=True)
    mock_connection = mock.MagicMock(spec=connection.Connection)
    mock_connection.wait_for_idle = mock.AsyncMock()
    mock_connection.send = mock.AsyncMock()

    async def mock_generator():
      yield final_step

    mock_connection.receive_steps.return_value = mock_generator()
    conv = conversation.Conversation(mock_connection)

    multimodal_prompt = [
        "Analyze this blueprint:",
        types.Image(mime_type="image/png", data=b"blueprint_bytes"),
    ]
    result = await conv.chat(multimodal_prompt)

    self.assertEqual(result.text, "image analysis done")
    mock_connection.send.assert_called_once_with(multimodal_prompt)

  async def test_chat_records_in_history(self):
    """Verifies chat() steps are accumulated in conversation history."""
    step = _make_step("done", step_index=1, is_final=True)
    mock_connection = mock.MagicMock(spec=connection.Connection)
    mock_connection.wait_for_idle = mock.AsyncMock()
    mock_connection.send = mock.AsyncMock()

    async def mock_generator():
      yield step

    mock_connection.receive_steps.return_value = mock_generator()
    conv = conversation.Conversation(mock_connection)

    await conv.chat("q")

    self.assertEqual(len(conv.history), 1)
    self.assertEqual(conv.turn_count, 1)

  async def test_chat_empty_response_when_no_final(self):
    """Verifies chat() returns empty text when no final response step exists."""
    step = _make_step("interim", step_index=1, is_final=False)
    mock_connection = mock.MagicMock(spec=connection.Connection)
    mock_connection.wait_for_idle = mock.AsyncMock()
    mock_connection.send = mock.AsyncMock()

    async def mock_generator():
      yield step

    mock_connection.receive_steps.return_value = mock_generator()
    conv = conversation.Conversation(mock_connection)

    result = await conv.chat("q")

    self.assertEqual(result.text, "")
    self.assertEqual(len(result.steps), 1)

  async def test_chat_returns_structured_output_when_final_step_has_it(self):
    """Verifies chat() collects and returns structured_output from the final step."""
    final_step = _make_step(
        "done", step_index=1, step_type=types.StepType.FINISH, is_final=True
    )
    final_step.structured_output = {"total_revenue": 386.0}

    mock_connection = mock.MagicMock(spec=connection.Connection)
    mock_connection.wait_for_idle = mock.AsyncMock()
    mock_connection.send = mock.AsyncMock()

    async def mock_generator():
      yield final_step

    mock_connection.receive_steps.return_value = mock_generator()
    conv = conversation.Conversation(mock_connection)

    result = await conv.chat("question")

    self.assertEqual(result.structured_output, {"total_revenue": 386.0})


class ConversationStateTest(unittest.IsolatedAsyncioTestCase):
  """Validates state introspection properties."""

  async def test_is_idle_delegates_to_connection(self):
    """Verifies is_idle reads from connection.is_idle property."""
    mock_connection = mock.MagicMock(spec=connection.Connection)
    type(mock_connection).is_idle = mock.PropertyMock(return_value=False)
    conv = conversation.Conversation(mock_connection)

    self.assertFalse(conv.is_idle)

    type(mock_connection).is_idle = mock.PropertyMock(return_value=True)
    self.assertTrue(conv.is_idle)

  async def test_conversation_id_delegates_to_connection(self):
    """Verifies conversation_id reads from connection.conversation_id."""
    mock_connection = mock.MagicMock(spec=connection.Connection)
    type(mock_connection).conversation_id = mock.PropertyMock(
        return_value="conv-123"
    )
    conv = conversation.Conversation(mock_connection)

    self.assertEqual(conv.conversation_id, "conv-123")

  async def test_conversation_id_empty_by_default(self):
    """Verifies conversation_id returns empty string from default ABC impl."""
    mock_connection = mock.MagicMock(spec=connection.Connection)
    type(mock_connection).conversation_id = mock.PropertyMock(return_value="")
    conv = conversation.Conversation(mock_connection)

    self.assertEqual(conv.conversation_id, "")


class ConversationLifecycleTest(unittest.IsolatedAsyncioTestCase):
  """Validates direct delegation of lifecycle methods without hasattr guards."""

  async def test_cancel_delegates(self):
    """Verifies cancel delegates directly to connection."""
    mock_connection = mock.AsyncMock(spec=connection.Connection)
    conv = conversation.Conversation(mock_connection)
    await conv.cancel()
    mock_connection.cancel.assert_called_once()

  async def test_delete_delegates(self):
    """Verifies delete delegates directly to connection."""
    mock_connection = mock.AsyncMock(spec=connection.Connection)
    conv = conversation.Conversation(mock_connection)
    await conv.delete()
    mock_connection.delete.assert_called_once()

  async def test_signal_idle_delegates(self):
    """Verifies signal_idle delegates directly to connection."""
    mock_connection = mock.AsyncMock(spec=connection.Connection)
    conv = conversation.Conversation(mock_connection)
    await conv.signal_idle()
    mock_connection.signal_idle.assert_called_once()

  async def test_wait_for_idle_delegates(self):
    """Verifies wait_for_idle delegates directly to connection."""
    mock_connection = mock.AsyncMock(spec=connection.Connection)
    conv = conversation.Conversation(mock_connection)
    await conv.wait_for_idle()
    mock_connection.wait_for_idle.assert_called_once()

  async def test_wait_for_wakeup_delegates(self):
    """Verifies wait_for_wakeup delegates to connection with timeout."""
    mock_connection = mock.AsyncMock(spec=connection.Connection)
    mock_connection.wait_for_wakeup.return_value = True
    conv = conversation.Conversation(mock_connection)
    result = await conv.wait_for_wakeup(timeout=60.0)
    self.assertTrue(result)
    mock_connection.wait_for_wakeup.assert_called_once_with(60.0)

  async def test_disconnect_delegates(self):
    """Verifies disconnect delegates directly to connection."""
    mock_connection = mock.AsyncMock(spec=connection.Connection)
    conv = conversation.Conversation(mock_connection)
    await conv.disconnect()
    mock_connection.disconnect.assert_called_once()


class ConversationClearHistoryTest(unittest.IsolatedAsyncioTestCase):
  """Validates clear_history and max_history_size behavior."""

  async def test_clear_history_resets_all_state(self):
    """Verifies clear_history empties steps, turns, and compaction indices."""
    mock_connection = mock.MagicMock(spec=connection.Connection)
    mock_connection.wait_for_idle = mock.AsyncMock()
    mock_connection.send = mock.AsyncMock()

    compaction = _make_step(
        "", step_index=1, step_type=types.StepType.COMPACTION,
    )
    final = _make_step("answer", step_index=2, is_final=True)

    async def mock_generator():
      yield compaction
      yield final

    mock_connection.receive_steps.return_value = mock_generator()
    conv = conversation.Conversation(mock_connection)

    await conv.send("q")
    async for _ in conv.receive_steps():
      pass

    # Verify state is populated.
    self.assertEqual(len(conv.history), 2)
    self.assertEqual(conv.turn_count, 1)
    self.assertEqual(len(conv.compaction_indices), 1)

    conv.clear_history()

    self.assertEqual(conv.history, [])
    self.assertEqual(conv.turn_count, 0)
    self.assertEqual(conv.compaction_indices, [])
    self.assertEqual(conv.last_response, "")

  async def test_max_history_trims_oldest_steps(self):
    """Verifies max_history_size trims oldest steps when exceeded."""
    mock_connection = mock.MagicMock(spec=connection.Connection)

    async def mock_generator():
      for i in range(10):
        yield _make_step(f"step-{i}", step_index=i)

    mock_connection.receive_steps.return_value = mock_generator()
    conv = conversation.Conversation(mock_connection, max_history_size=5)

    async for _ in conv.receive_steps():
      pass

    self.assertEqual(len(conv.history), 5)
    # Oldest steps are trimmed; newest remain.
    self.assertEqual(conv.history[0].content, "step-5")
    self.assertEqual(conv.history[-1].content, "step-9")

  async def test_max_history_adjusts_compaction_indices(self):
    """Verifies compaction indices are adjusted when history is trimmed."""
    mock_connection = mock.MagicMock(spec=connection.Connection)

    async def mock_generator():
      yield _make_step("a", step_index=0)
      yield _make_step(
          "", step_index=1, step_type=types.StepType.COMPACTION,
      )
      yield _make_step("b", step_index=2)
      yield _make_step("c", step_index=3)
      yield _make_step("d", step_index=4)

    mock_connection.receive_steps.return_value = mock_generator()
    conv = conversation.Conversation(mock_connection, max_history_size=3)

    async for _ in conv.receive_steps():
      pass

    # History trimmed to last 3 steps: [b, c, d].
    self.assertEqual(len(conv.history), 3)
    # Compaction at original index 1 was before the trim window, so removed.
    self.assertEqual(conv.compaction_indices, [])

  async def test_max_history_zero_disables_limit(self):
    """Verifies max_history_size=0 disables the limit."""
    mock_connection = mock.MagicMock(spec=connection.Connection)

    async def mock_generator():
      for i in range(100):
        yield _make_step(f"step-{i}", step_index=i)

    mock_connection.receive_steps.return_value = mock_generator()
    conv = conversation.Conversation(mock_connection, max_history_size=0)

    async for _ in conv.receive_steps():
      pass

    self.assertEqual(len(conv.history), 100)


class ConversationUsageMetadataTest(unittest.IsolatedAsyncioTestCase):

  def _make_step_with_usage(
      self,
      step_index: int = 0,
      prompt: int | None = None,
      candidates: int | None = None,
      total: int | None = None,
      thoughts: int | None = None,
      cached: int | None = None,
      is_final: bool = False,
  ) -> types.Step:
    """Creates a Step with optional usage_metadata for testing."""
    usage = types.UsageMetadata(
        prompt_token_count=prompt,
        cached_content_token_count=cached,
        candidates_token_count=candidates,
        thoughts_token_count=thoughts,
        total_token_count=total,
    )
    return types.Step(
        id=str(step_index),
        step_index=step_index,
        type=types.StepType.MODEL_RESPONSE,
        source=types.StepSource.MODEL,
        status=types.StepStatus.DONE,
        content="",
        is_complete_response=is_final,
        usage_metadata=usage,
    )

  def _make_conv_with_mock(self):
    mock_connection = mock.MagicMock(spec=connection.Connection)
    mock_connection.wait_for_idle = mock.AsyncMock()
    mock_connection.send = mock.AsyncMock()
    return conversation.Conversation(mock_connection), mock_connection

  async def test_total_usage_starts_at_zero(self):
    """Verifies cumulative usage is initialized to zero, not None."""
    conv, _ = self._make_conv_with_mock()
    usage = conv.total_usage
    self.assertEqual(
        usage,
        types.UsageMetadata(
            prompt_token_count=0,
            cached_content_token_count=0,
            candidates_token_count=0,
            thoughts_token_count=0,
            total_token_count=0,
        ),
    )

  async def test_total_usage_accumulates_across_steps(self):
    """Verifies usage is summed from every step that reports it."""
    conv, mock_connection = self._make_conv_with_mock()

    async def gen():
      yield self._make_step_with_usage(0, prompt=100, candidates=50, total=150)
      yield self._make_step_with_usage(1, prompt=200, candidates=30, total=230)

    mock_connection.receive_steps.return_value = gen()
    async for _ in conv.receive_steps():
      pass

    usage = conv.total_usage
    self.assertEqual(usage.prompt_token_count, 300)
    self.assertEqual(usage.candidates_token_count, 80)
    self.assertEqual(usage.total_token_count, 380)

  async def test_total_usage_ignores_none_fields(self):
    """Verifies None usage fields don't affect the cumulative total."""
    conv, mock_connection = self._make_conv_with_mock()

    step_with = self._make_step_with_usage(0, prompt=100, thoughts=10)
    step_without = _make_step("no usage", step_index=1)  # usage_metadata=None

    async def gen():
      yield step_with
      yield step_without

    mock_connection.receive_steps.return_value = gen()
    async for _ in conv.receive_steps():
      pass

    usage = conv.total_usage
    self.assertEqual(usage.prompt_token_count, 100)
    self.assertEqual(usage.thoughts_token_count, 10)

  async def test_total_usage_accumulates_across_turns(self):
    """Verifies cumulative usage spans multiple send/receive cycles."""
    conv, mock_connection = self._make_conv_with_mock()

    async def gen1():
      yield self._make_step_with_usage(0, prompt=100, total=120)

    async def gen2():
      yield self._make_step_with_usage(1, prompt=150, total=180)

    mock_connection.receive_steps.return_value = gen1()
    await conv.send("turn1")
    async for _ in conv.receive_steps():
      pass

    mock_connection.receive_steps.return_value = gen2()
    await conv.send("turn2")
    async for _ in conv.receive_steps():
      pass

    usage = conv.total_usage
    self.assertEqual(usage.prompt_token_count, 250)
    self.assertEqual(usage.total_token_count, 300)

  async def test_total_usage_returns_copy(self):
    """Verifies total_usage returns a copy, not a reference to internal state."""
    conv, _ = self._make_conv_with_mock()
    usage = conv.total_usage
    usage.prompt_token_count = 999
    self.assertEqual(conv.total_usage.prompt_token_count, 0)

  async def test_clear_history_resets_usage(self):
    """Verifies clear_history resets cumulative usage to zero."""
    conv, mock_connection = self._make_conv_with_mock()

    async def gen():
      yield self._make_step_with_usage(0, prompt=500, total=600)

    mock_connection.receive_steps.return_value = gen()
    await conv.send("q")
    async for _ in conv.receive_steps():
      pass

    self.assertEqual(conv.total_usage.prompt_token_count, 500)

    conv.clear_history()

    self.assertEqual(conv.total_usage.prompt_token_count, 0)
    self.assertEqual(conv.total_usage.total_token_count, 0)

  async def test_chat_returns_accumulated_usage_metadata(self):
    """Verifies chat() accumulates usage across all steps in the turn."""
    conv, mock_connection = self._make_conv_with_mock()

    step1 = self._make_step_with_usage(0, prompt=100, candidates=50, total=150)
    step2 = self._make_step_with_usage(
        1, prompt=200, candidates=30, total=230, is_final=True
    )
    step2.content = "the answer"
    step2.is_complete_response = True

    async def gen():
      yield step1
      yield step2

    mock_connection.receive_steps.return_value = gen()
    result = await conv.chat("question")

    self.assertIsNotNone(result.usage_metadata)
    self.assertEqual(result.usage_metadata.prompt_token_count, 300)
    self.assertEqual(result.usage_metadata.candidates_token_count, 80)
    self.assertEqual(result.usage_metadata.total_token_count, 380)

  async def test_chat_returns_none_usage_when_absent(self):
    """Verifies chat() returns None usage_metadata when no step has it."""
    conv, mock_connection = self._make_conv_with_mock()

    async def gen():
      yield _make_step("answer", step_index=0, is_final=True)

    mock_connection.receive_steps.return_value = gen()
    result = await conv.chat("question")

    self.assertIsNone(result.usage_metadata)


if __name__ == "__main__":
  unittest.main()
