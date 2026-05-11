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

"""Tests for Google Antigravity SDK Pydantic type definitions.

Validates model construction, validation, immutability, forward compatibility,
and the AntigravityValidationError wrapper.
"""

import pathlib
import tempfile
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import pydantic

from google.antigravity import types
from google.antigravity.conversation import conversation


class ToolCallTest(unittest.TestCase):
  """Validates the ToolCall Pydantic model."""

  def test_basic_construction(self):
    """Verifies that a ToolCall can be constructed with name and args.

    What: Checks basic field assignment.
    Why: Validates the happy path for the most commonly used SDK type.
    How: Constructs a ToolCall and asserts field values.
    """
    tc = types.ToolCall(name="read_file", args={"path": "/tmp/foo"})
    self.assertEqual(tc.name, "read_file")
    self.assertEqual(tc.args, {"path": "/tmp/foo"})

  def test_default_args(self):
    """Verifies that args defaults to empty dict when omitted.

    What: Checks default factory for args field.
    Why: Many tool calls have no arguments.
    How: Constructs a ToolCall without args and asserts empty dict.
    """
    tc = types.ToolCall(name="no_args_tool")
    self.assertEqual(tc.args, {})

  def test_id_defaults_to_none(self):
    """Verifies that id defaults to None when omitted."""
    tc = types.ToolCall(name="tool")
    self.assertIsNone(tc.id)

  def test_id_explicitly_set(self):
    """Verifies that id can be explicitly set."""
    tc = types.ToolCall(id="call_123", name="tool")
    self.assertEqual(tc.id, "call_123")

  def test_extra_fields_ignored(self):
    """Verifies that unknown fields are silently dropped.

    What: Checks extra='ignore' behavior.
    Why: Forward compatibility — newer backends may add fields.
    How: Constructs a ToolCall with an unknown field and asserts it's absent.
    """
    tc = types.ToolCall(name="tool", unknown_field="value")
    self.assertFalse(hasattr(tc, "unknown_field"))

  def test_missing_name_raises(self):
    """Verifies that omitting required field 'name' raises.

    What: Checks required field validation.
    Why: Every tool call must have a name.
    How: Attempts construction without name and asserts ValidationError.
    """
    with self.assertRaises(pydantic.ValidationError):
      types.ToolCall()


class ToolResultTest(unittest.TestCase):
  """Validates the ToolResult Pydantic model."""

  def test_success_result(self):
    """Verifies construction of a successful ToolResult.

    What: Checks that result and error fields are set correctly.
    Why: Validates the common success case.
    How: Constructs a ToolResult with a result and asserts fields.
    """
    tr = types.ToolResult(name="sum_tool", result=42)
    self.assertEqual(tr.name, "sum_tool")
    self.assertEqual(tr.result, 42)
    self.assertIsNone(tr.error)

  def test_error_result(self):
    """Verifies construction of an error ToolResult.

    What: Checks that error field is populated.
    Why: Validates the error path for failed tool executions.
    How: Constructs a ToolResult with an error string and asserts.
    """
    tr = types.ToolResult(name="bad_tool", error="kaboom")
    self.assertEqual(tr.error, "kaboom")
    self.assertIsNone(tr.result)

  def test_mutable(self):
    """Verifies that ToolResult is mutable (not frozen).

    What: Checks that fields can be updated after construction.
    Why: ToolResult is built up during execution and may need mutation.
    How: Sets a field after construction and asserts the new value.
    """
    tr = types.ToolResult(name="tool")
    tr.result = "updated"
    self.assertEqual(tr.result, "updated")

  def test_id_defaults_to_none(self):
    """Verifies that id defaults to None when omitted."""
    tr = types.ToolResult(name="tool")
    self.assertIsNone(tr.id)

  def test_id_explicitly_set(self):
    """Verifies that id can be explicitly set for call correlation."""
    tr = types.ToolResult(id="call_123", name="tool", result="ok")
    self.assertEqual(tr.id, "call_123")

  def test_id_mutable(self):
    """Verifies that id can be set after construction."""
    tr = types.ToolResult(name="tool")
    tr.id = "call_456"
    self.assertEqual(tr.id, "call_456")

  def test_extra_fields_ignored(self):
    """Verifies extra='ignore' on ToolResult.

    What: Checks forward compatibility.
    Why: Consistent extra field handling across all models.
    How: Passes an unknown field and asserts it's not present.
    """
    tr = types.ToolResult(name="tool", unknown="value")
    self.assertFalse(hasattr(tr, "unknown"))


class StepTest(unittest.TestCase):
  """Validates the Step Pydantic model."""

  def test_basic_construction(self):
    """Verifies that a Step can be constructed with all fields."""
    tc = types.ToolCall(name="run_command", args={"cmd": "ls"})
    step = types.Step(
        id="1",
        step_index=1,
        type=types.StepType.TOOL_CALL,
        status=types.StepStatus.DONE,
        source=types.StepSource.MODEL,
        content="output",
        thinking="reasoning",
        tool_calls=[tc],
        error="",
    )
    self.assertEqual(step.id, "1")
    self.assertEqual(step.step_index, 1)
    self.assertEqual(step.type, types.StepType.TOOL_CALL)
    self.assertEqual(step.tool_calls[0].name, "run_command")

  def test_defaults(self):
    """Verifies that all Step fields have sensible defaults."""
    step = types.Step()
    self.assertEqual(step.id, "")
    self.assertEqual(step.step_index, 0)
    self.assertEqual(step.type, types.StepType.UNKNOWN)
    self.assertEqual(step.status, types.StepStatus.UNKNOWN)
    self.assertEqual(step.source, types.StepSource.UNKNOWN)
    self.assertEqual(step.content, "")
    self.assertEqual(step.thinking, "")
    self.assertEqual(step.tool_calls, [])
    self.assertEqual(step.error, "")
    self.assertIsNone(step.is_complete_response)
    self.assertIsNone(step.structured_output)

  def test_mutable(self):
    """Verifies that Step is mutable as per Karmel's model."""
    step = types.Step(id="1", content="hello")
    step.content = "goodbye"
    self.assertEqual(step.content, "goodbye")

  def test_extra_fields_allowed(self):
    """Verifies extra='allow' on Step as per Karmel's model."""
    step = types.Step(id="1", future_field="value")
    self.assertTrue(hasattr(step, "future_field"))
    self.assertEqual(getattr(step, "future_field"), "value")

  def test_nested_tool_call(self):
    """Verifies that a Step can contain a nested ToolCall."""
    step = types.Step(
        id="5",
        type=types.StepType.TOOL_CALL,
        tool_calls=[{"name": "my_tool", "args": {"x": 1}}],
    )
    self.assertEqual(len(step.tool_calls), 1)
    self.assertEqual(step.tool_calls[0].name, "my_tool")
    self.assertEqual(step.tool_calls[0].args, {"x": 1})


class HookResultTest(unittest.TestCase):
  """Validates the HookResult Pydantic model."""

  def test_defaults(self):
    """Verifies that HookResult defaults to allow=True.

    What: Checks default values.
    Why: The default behavior should be permissive.
    How: Constructs a HookResult with no arguments and checks allow.
    """
    hr = types.HookResult()
    self.assertTrue(hr.allow)
    self.assertEqual(hr.message, "")

  def test_deny(self):
    """Verifies construction of a deny HookResult.

    What: Checks explicit deny behavior.
    Why: Validates the policy enforcement path.
    How: Constructs with allow=False and a message.
    """
    hr = types.HookResult(allow=False, message="blocked by policy")
    self.assertFalse(hr.allow)
    self.assertEqual(hr.message, "blocked by policy")

  def test_mutable(self):
    """Verifies that HookResult is mutable.

    What: Checks that allow can be changed after construction.
    Why: Hook runners may need to update results during dispatch.
    How: Modifies the allow field after construction.
    """
    hr = types.HookResult(allow=True)
    hr.allow = False
    self.assertFalse(hr.allow)


class QuestionResponseTest(unittest.TestCase):
  """Validates the QuestionResponse Pydantic model."""

  def test_defaults(self):
    """Verifies QuestionResponse defaults.

    What: Checks that all fields have sensible defaults.
    Why: Most responses only populate one field.
    How: Constructs with no arguments and checks defaults.
    """
    qr = types.QuestionResponse()
    self.assertIsNone(qr.selected_option_ids)
    self.assertEqual(qr.freeform_response, "")
    self.assertFalse(qr.skipped)

  def test_skipped(self):
    """Verifies construction of a skipped response.

    What: Checks skipped flag.
    Why: Users can skip questions.
    How: Constructs with skipped=True and asserts.
    """
    qr = types.QuestionResponse(skipped=True)
    self.assertTrue(qr.skipped)

  def test_selected_options(self):
    """Verifies construction with selected option IDs.

    What: Checks option selection.
    Why: Most common response type is selecting from options.
    How: Constructs with selected_option_ids and asserts.
    """
    qr = types.QuestionResponse(selected_option_ids=["opt1", "opt2"])
    self.assertEqual(qr.selected_option_ids, ["opt1", "opt2"])

  def test_write_in(self):
    """Verifies construction with a write-in response.

    What: Checks write-in text.
    Why: Freeform text is an alternative to option selection.
    How: Constructs with freeform_response and asserts.
    """
    qr = types.QuestionResponse(freeform_response="custom answer")
    self.assertEqual(qr.freeform_response, "custom answer")


class QuestionHookResultTest(unittest.TestCase):
  """Validates the QuestionHookResult Pydantic model."""

  def test_basic_construction(self):
    """Verifies construction with a list of responses.

    What: Checks required field 'responses'.
    Why: Every interaction must have at least one response.
    How: Constructs with a list of QuestionResponse objects.
    """
    qhr = types.QuestionHookResult(
        responses=[types.QuestionResponse(skipped=True)]
    )
    self.assertEqual(len(qhr.responses), 1)
    self.assertTrue(qhr.responses[0].skipped)
    self.assertFalse(qhr.cancelled)

  def test_cancelled(self):
    """Verifies cancelled interaction.

    What: Checks cancelled flag.
    Why: User may cancel an interaction (e.g. EOF).
    How: Constructs with cancelled=True.
    """
    qhr = types.QuestionHookResult(responses=[], cancelled=True)
    self.assertTrue(qhr.cancelled)

  def test_missing_responses_raises(self):
    """Verifies that omitting required 'responses' raises.

    What: Checks required field validation.
    Why: responses is a required field.
    How: Attempts construction without responses.
    """
    with self.assertRaises(pydantic.ValidationError):
      types.QuestionHookResult()


class AntigravityValidationErrorTest(unittest.TestCase):
  """Validates the AntigravityValidationError wrapper."""

  def test_basic_construction(self):
    """Verifies direct construction with a message.

    What: Checks that the exception stores message and errors.
    Why: SDK consumers catch this instead of pydantic.ValidationError.
    How: Constructs the exception and checks attributes.
    """
    err = types.AntigravityValidationError("bad input")
    self.assertEqual(str(err), "bad input")
    self.assertEqual(err.message, "bad input")
    self.assertEqual(err.errors, [])

  def test_from_pydantic(self):
    """Verifies construction from a real Pydantic ValidationError.

    What: Checks the from_pydantic factory method.
    Why: This is the primary construction path at SDK boundaries.
    How: Triggers a ValidationError and wraps it.
    """
    err = None
    try:
      types.ToolCall()  # Missing required 'name' field.
    except pydantic.ValidationError as e:
      err = e

    self.assertIsNotNone(err, "Expected ValidationError was not raised.")
    wrapped = types.AntigravityValidationError.from_pydantic(err)
    self.assertIn("name", wrapped.message)
    self.assertGreater(len(wrapped.errors), 0)

  def test_is_exception(self):
    """Verifies that AntigravityValidationError is a proper Exception.

    What: Checks isinstance relationship.
    Why: Must be catchable as a standard Python exception.
    How: Asserts isinstance against Exception.
    """
    err = types.AntigravityValidationError("test")
    self.assertIsInstance(err, Exception)

  def test_with_errors_list(self):
    """Verifies construction with an explicit errors list.

    What: Checks that the errors list is preserved.
    Why: Structured errors allow programmatic handling.
    How: Passes an explicit errors list and asserts.
    """
    errors = [{"type": "missing", "loc": ("name",), "msg": "Field required"}]
    err = types.AntigravityValidationError("validation failed", errors=errors)
    self.assertEqual(len(err.errors), 1)
    self.assertEqual(err.errors[0]["type"], "missing")

  def test_step_pydantic(self):
    """Tests that Step can be instantiated as a Pydantic model."""
    step = types.Step(id="1", content="test content")
    self.assertEqual(step.id, "1")
    self.assertEqual(step.content, "test content")
    self.assertEqual(step.type, types.StepType.UNKNOWN)
    self.assertEqual(step.tool_calls, [])

  def test_step_with_tool_calls(self):
    """Tests that Step can hold multiple tool calls."""
    tc1 = types.ToolCall(name="tool1")
    tc2 = types.ToolCall(name="tool2")
    step = types.Step(tool_calls=[tc1, tc2])
    self.assertEqual(len(step.tool_calls), 2)
    self.assertEqual(step.tool_calls[0].name, "tool1")
    self.assertEqual(step.tool_calls[1].name, "tool2")

  def test_tool_call_pydantic(self):
    """Tests that ToolCall can be instantiated as a Pydantic model."""
    tc = types.ToolCall(name="my_tool", args={"a": 1})
    self.assertEqual(tc.name, "my_tool")
    self.assertEqual(tc.args, {"a": 1})


class AskQuestionModelsTest(unittest.TestCase):
  """Tests for AskQuestion related models."""

  def test_ask_question_option(self):
    opt = types.AskQuestionOption(id="A", text="Option A")
    self.assertEqual(opt.id, "A")
    self.assertEqual(opt.text, "Option A")

  def test_ask_question_entry(self):
    opt = types.AskQuestionOption(id="A", text="Option A")
    entry = types.AskQuestionEntry(question="Q?", options=[opt])
    self.assertEqual(entry.question, "Q?")
    self.assertEqual(len(entry.options), 1)
    self.assertFalse(entry.is_multi_select)

  def test_ask_question_interaction_spec(self):
    opt = types.AskQuestionOption(id="A", text="Option A")
    entry = types.AskQuestionEntry(question="Q?", options=[opt])
    spec = types.AskQuestionInteractionSpec(questions=[entry])
    self.assertEqual(len(spec.questions), 1)


class ThinkingLevelTest(unittest.TestCase):
  """Tests for the ThinkingLevel enum."""

  def test_enum_values(self):
    """Verifies each enum member has the expected string value."""
    self.assertEqual(types.ThinkingLevel.MINIMAL, "minimal")
    self.assertEqual(types.ThinkingLevel.LOW, "low")
    self.assertEqual(types.ThinkingLevel.MEDIUM, "medium")
    self.assertEqual(types.ThinkingLevel.HIGH, "high")

  def test_string_comparison(self):
    """Verifies ThinkingLevel members compare equal to their string values."""
    self.assertEqual(types.ThinkingLevel.LOW, "low")
    self.assertNotEqual(types.ThinkingLevel.LOW, "high")


class GeminiConfigTest(unittest.TestCase):
  """Tests for the GeminiConfig Pydantic model."""

  def test_default_construction(self):
    """Verifies that GeminiConfig can be constructed with all defaults."""
    config = types.GeminiConfig()
    self.assertIsNone(config.api_key)
    self.assertEqual(config.models.default.name, "gemini-3-flash-preview")
    self.assertIsNone(config.models.default.generation.thinking_level)

  def test_explicit_field_assignment(self):
    """Verifies that all fields can be explicitly set."""
    config = types.GeminiConfig(
        api_key="test-key",
        models=types.ModelConfig(
            default=types.ModelEntry(
                name="gemini-2.5-pro",
                generation=types.GenerationConfig(
                    thinking_level=types.ThinkingLevel.LOW,
                ),
            ),
        ),
    )
    self.assertEqual(config.api_key, "test-key")
    self.assertEqual(config.models.default.name, "gemini-2.5-pro")
    self.assertEqual(
        config.models.default.generation.thinking_level,
        types.ThinkingLevel.LOW,
    )

  def test_string_coercion_in_model_config(self):
    """Verifies that ModelConfig coerces strings to ModelEntry."""
    config = types.ModelConfig(default="gemini-2.5-pro")
    self.assertIsInstance(config.default, types.ModelEntry)
    self.assertEqual(config.default.name, "gemini-2.5-pro")

  def test_thinking_level_from_string(self):
    """Verifies that thinking_level accepts raw string values."""
    gen = types.GenerationConfig(thinking_level="high")
    self.assertEqual(gen.thinking_level, types.ThinkingLevel.HIGH)

  def test_thinking_level_invalid_string(self):
    """Verifies that invalid thinking_level strings raise ValidationError."""
    with self.assertRaises(pydantic.ValidationError):
      types.GenerationConfig(thinking_level="turbo")

  def test_per_model_api_key(self):
    """Verifies per-model API key overrides."""
    entry = types.ModelEntry(name="model-x", api_key="per-model-key")
    config = types.GeminiConfig(
        api_key="shared-key",
        models=types.ModelConfig(default=entry),
    )
    self.assertEqual(config.api_key, "shared-key")
    self.assertEqual(config.models.default.api_key, "per-model-key")

  def test_image_generation_model_default(self):
    """Verifies the default image generation model."""
    config = types.ModelConfig()
    self.assertEqual(
        config.image_generation.name, "gemini-3.1-flash-image-preview"
    )

  def test_string_coercion_image_generation_slot(self):
    """Verifies that BeforeValidator coerces string to ModelEntry for image_generation."""
    config = types.ModelConfig(image_generation="custom-image-model")
    self.assertIsInstance(config.image_generation, types.ModelEntry)
    self.assertEqual(config.image_generation.name, "custom-image-model")

  def test_gemini_config_mutable_for_sugar(self):
    """Verifies GeminiConfig fields can be mutated (needed by AgentConfig sugar)."""
    config = types.GeminiConfig()
    config.api_key = "new-key"
    self.assertEqual(config.api_key, "new-key")


class SystemInstructionsTest(unittest.TestCase):
  """Tests for the SystemInstructions Pydantic model union."""

  def test_custom_construction(self):
    """Verifies construction of CustomSystemInstructions."""
    si = types.CustomSystemInstructions(text="Override all defaults.")
    self.assertEqual(si.text, "Override all defaults.")

  def test_templated_construction(self):
    """Verifies construction of TemplatedSystemInstructions."""
    section = types.SystemInstructionSection(
        title="extra", content="More instructions"
    )
    si = types.TemplatedSystemInstructions(
        identity="New Identity", sections=[section]
    )
    self.assertEqual(si.identity, "New Identity")
    self.assertEqual(len(si.sections), 1)
    self.assertEqual(si.sections[0].title, "extra")

  def test_union_parsing_custom(self):
    """Verifies that Pydantic parses CustomSystemInstructions from dict."""
    data = {"text": "Be helpful."}
    adapter = pydantic.TypeAdapter(types.SystemInstructions)
    si = adapter.validate_python(data)
    self.assertIsInstance(si, types.CustomSystemInstructions)
    self.assertEqual(si.text, "Be helpful.")

  def test_union_parsing_templated(self):
    """Verifies that Pydantic parses TemplatedSystemInstructions from dict."""
    data = {
        "identity": "I am robot",
        "sections": [{"title": "rules", "content": "Do no harm"}],
    }
    adapter = pydantic.TypeAdapter(types.SystemInstructions)
    si = adapter.validate_python(data)
    self.assertIsInstance(si, types.TemplatedSystemInstructions)
    self.assertEqual(si.identity, "I am robot")
    self.assertEqual(len(si.sections), 1)
    self.assertEqual(si.sections[0].title, "rules")

  def test_custom_text_is_required(self):
    """Verifies that CustomSystemInstructions raises when text is missing."""
    with self.assertRaises(pydantic.ValidationError):
      types.CustomSystemInstructions()  # type: ignore

  def test_templated_empty_construction(self):
    """Verifies that TemplatedSystemInstructions can be constructed empty."""
    si = types.TemplatedSystemInstructions()
    self.assertIsNone(si.identity)
    self.assertEqual(si.sections, [])

  def test_union_parsing_empty_dict(self):
    """Verifies that Pydantic parses empty dict as TemplatedSystemInstructions."""
    data = {}
    adapter = pydantic.TypeAdapter(types.SystemInstructions)
    si = adapter.validate_python(data)
    self.assertIsInstance(si, types.TemplatedSystemInstructions)
    self.assertIsNone(si.identity)
    self.assertEqual(si.sections, [])


class BuiltinToolsTest(unittest.TestCase):
  """Tests for the BuiltinTools enum."""

  def test_enum_values(self):
    """Verifies each enum member has the expected string value."""
    self.assertEqual(types.BuiltinTools.LIST_DIR, "list_directory")
    self.assertEqual(types.BuiltinTools.SEARCH_DIR, "search_directory")
    self.assertEqual(types.BuiltinTools.VIEW_FILE, "view_file")
    self.assertEqual(types.BuiltinTools.CREATE_FILE, "create_file")
    self.assertEqual(types.BuiltinTools.EDIT_FILE, "edit_file")
    self.assertEqual(types.BuiltinTools.RUN_COMMAND, "run_command")
    self.assertEqual(types.BuiltinTools.ASK_QUESTION, "ask_question")

  def test_read_only_covers_all_tools(self):
    """Verifies read_only + write tools = full enum.

    If a new BuiltinTools member is added without updating either read_only()
    or this test's write_tools set, the test will fail, forcing the developer
    to categorize the new tool.
    """
    read_only = set(types.BuiltinTools.read_only())
    write_tools = {
        types.BuiltinTools.CREATE_FILE,
        types.BuiltinTools.EDIT_FILE,
        types.BuiltinTools.RUN_COMMAND,
        types.BuiltinTools.ASK_QUESTION,
        types.BuiltinTools.START_SUBAGENT,
        types.BuiltinTools.GENERATE_IMAGE,
    }
    self.assertEqual(
        read_only | write_tools,
        set(types.BuiltinTools),
        "A new BuiltinTools member was added but not categorized in"
        " read_only() or this test's write_tools set.",
    )
    self.assertFalse(
        read_only & write_tools,
        "read_only and write_tools must not overlap.",
    )

  def test_nondestructive_covers_all_tools(self):
    """Verifies nondestructive + destructive tools = full enum.

    If a new BuiltinTools member is added without updating either
    nondestructive() or this test's destructive_tools set, the test will fail,
    forcing the developer to categorize the new tool.
    """
    nondestructive = set(types.BuiltinTools.nondestructive())
    destructive_tools = {
        types.BuiltinTools.RUN_COMMAND,
    }
    self.assertEqual(
        nondestructive | destructive_tools,
        set(types.BuiltinTools),
        "A new BuiltinTools member was added but not categorized in"
        " nondestructive() or this test's destructive_tools set.",
    )
    self.assertFalse(
        nondestructive & destructive_tools,
        "nondestructive and destructive_tools must not overlap.",
    )

  def test_all_returns_every_member(self):
    """Verifies that all() returns every enum member."""
    self.assertEqual(set(types.BuiltinTools.all()), set(types.BuiltinTools))
    self.assertEqual(len(types.BuiltinTools.all()), len(types.BuiltinTools))

  def test_none_returns_empty_list(self):
    """Verifies that none() returns an empty list."""
    self.assertEqual(types.BuiltinTools.none(), [])


class CapabilitiesConfigTest(unittest.TestCase):
  """Tests for the CapabilitiesConfig Pydantic model."""

  def test_default_construction(self):
    """Verifies defaults: subagents enabled, no tool lists, no threshold."""
    config = types.CapabilitiesConfig()
    self.assertTrue(config.enable_subagents)
    self.assertIsNone(config.enabled_tools)
    self.assertIsNone(config.disabled_tools)
    self.assertIsNone(config.compaction_threshold)
    self.assertIsNone(config.finish_tool_schema_json)

  def test_enabled_tools(self):
    """Verifies that enabled_tools accepts a list of BuiltinTools."""
    config = types.CapabilitiesConfig(
        enabled_tools=[types.BuiltinTools.VIEW_FILE]
    )
    self.assertEqual(config.enabled_tools, [types.BuiltinTools.VIEW_FILE])
    self.assertIsNone(config.disabled_tools)

  def test_disabled_tools(self):
    """Verifies that disabled_tools accepts a list of BuiltinTools."""
    config = types.CapabilitiesConfig(
        disabled_tools=[
            types.BuiltinTools.RUN_COMMAND,
        ]
    )
    self.assertIsNone(config.enabled_tools)
    self.assertEqual(len(config.disabled_tools), 1)

  def test_mutually_exclusive_raises(self):
    """Verifies that setting both enabled_tools and disabled_tools raises."""
    with self.assertRaises(pydantic.ValidationError):
      types.CapabilitiesConfig(
          enabled_tools=[types.BuiltinTools.VIEW_FILE],
          disabled_tools=[types.BuiltinTools.RUN_COMMAND],
      )

  def test_compaction_threshold_explicit(self):
    """Verifies that compaction_threshold accepts an explicit integer."""
    config = types.CapabilitiesConfig(compaction_threshold=50000)
    self.assertEqual(config.compaction_threshold, 50000)


class AntigravityConnectionErrorTest(unittest.TestCase):
  """Validates the AntigravityConnectionError hierarchy."""

  def test_is_exception(self):
    """Verifies that AntigravityConnectionError is a proper Exception."""
    err = types.AntigravityConnectionError("connection failed")
    self.assertIsInstance(err, Exception)

  def test_message(self):
    """Verifies that the message is stored and retrievable."""
    err = types.AntigravityConnectionError("timeout")
    self.assertEqual(str(err), "timeout")


class ImageTest(unittest.TestCase):
  """Tests for the Image content attachment primitive and its validators."""

  def test_basic_construction(self):
    """Verifies that an Image can be successfully constructed with valid arguments."""
    img = types.Image(
        data=b"png_data", mime_type="image/png", description="diagram"
    )
    self.assertEqual(img.data, b"png_data")
    self.assertEqual(img.mime_type, "image/png")
    self.assertEqual(img.description, "diagram")

  def test_unsupported_mime_type_raises(self):
    """Verifies that an unsupported Image MIME type triggers ValidationError."""
    with self.assertRaises(pydantic.ValidationError):
      types.Image(data=b"gif_bytes", mime_type="image/gif")

  def test_from_file_success(self):
    """Verifies that from_file loader loads bytes and guesses MIME correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
      tmp_file = pathlib.Path(tmpdir) / "photo.png"
      fake_bytes = b"png_file_content"
      tmp_file.write_bytes(fake_bytes)

      img = types.Image.from_file(tmp_file, description="profile photo")
      self.assertEqual(img.data, fake_bytes)
      self.assertEqual(img.mime_type, "image/png")
      self.assertEqual(img.description, "profile photo")


class AudioTest(unittest.TestCase):
  """Validates the Audio content attachment primitive and its validators."""

  def test_basic_construction(self):
    """Verifies that an Audio can be successfully constructed with valid arguments."""
    audio = types.Audio(data=b"mp3_data", mime_type="audio/mp3")
    self.assertEqual(audio.data, b"mp3_data")
    self.assertEqual(audio.mime_type, "audio/mp3")

  def test_unsupported_mime_type_raises(self):
    """Verifies that an unsupported Audio MIME type triggers ValidationError."""
    with self.assertRaises(pydantic.ValidationError):
      types.Audio(data=b"wav_bytes", mime_type="audio/unsupported-wav")


class VideoTest(unittest.TestCase):
  """Validates the Video content attachment primitive and its validators."""

  def test_basic_construction(self):
    """Verifies that a Video can be successfully constructed with valid arguments."""
    video = types.Video(data=b"mp4_data", mime_type="video/mp4")
    self.assertEqual(video.data, b"mp4_data")
    self.assertEqual(video.mime_type, "video/mp4")

  def test_unsupported_mime_type_raises(self):
    """Verifies that an unsupported Video MIME type triggers ValidationError."""
    with self.assertRaises(pydantic.ValidationError):
      types.Video(data=b"mov_bytes", mime_type="video/unsupported-mov")


class ContentFromFileResolverTest(parameterized.TestCase):
  """Validates the global from_file content resolver helper function."""

  @parameterized.named_parameters(
      ("image", "diagram.png", types.Image, "image/png"),
      ("document", "report.pdf", types.Document, "application/pdf"),
      ("audio", "clip.mp3", types.Audio, "audio/mpeg"),
      ("video", "movie.mp4", types.Video, "video/mp4"),
  )
  def test_resolves_from_file(self, filename, expected_class, expected_mime):
    """Verifies that local files are resolved to the correct Content primitives."""
    with tempfile.TemporaryDirectory() as tmpdir:
      tmp_file = pathlib.Path(tmpdir) / filename
      tmp_file.write_bytes(b"fake_bytes")

      res = types.from_file(tmp_file, description="test asset")
      self.assertIsInstance(res, expected_class)
      self.assertEqual(res.mime_type, expected_mime)
      self.assertEqual(res.description, "test asset")

  def test_non_existent_path_raises_error(self):
    """Verifies that passing a non-existent path triggers FileNotFoundError."""
    with self.assertRaises(FileNotFoundError):
      types.from_file("non_existent_file_xyz.png")

  def test_directory_path_raises_error(self):
    """Verifies that passing a path targeting a directory triggers IsADirectoryError."""
    with tempfile.TemporaryDirectory() as tmpdir:
      with self.assertRaises(IsADirectoryError):
        types.from_file(tmpdir)

  def test_inference_failure_raises_clear_value_error(self):
    """Verifies that an extensionless path failure raises a descriptive ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
      tmp_file = pathlib.Path(tmpdir) / "extensionless_data_blob"
      tmp_file.write_bytes(b"some_anonymous_bytes")

      with self.assertRaisesRegex(
          ValueError, "Could not infer a valid MIME type"
      ):
        types.from_file(tmp_file)

  def test_permission_error_wrapping(self):
    """Verifies that permission errors are caught and wrapped appropriately."""
    with tempfile.TemporaryDirectory() as tmpdir:
      tmp_file = pathlib.Path(tmpdir) / "locked.png"
      tmp_file.write_bytes(b"data")

      # Mock read_bytes to raise PermissionError
      with mock.patch.object(
          pathlib.Path,
          "read_bytes",
          autospec=True,
          side_effect=PermissionError("access denied"),
      ):
        with self.assertRaisesRegex(PermissionError, "Permission denied"):
          types.from_file(tmp_file)

  def test_os_error_wrapping(self):
    """Verifies that OS-level read errors are caught and wrapped appropriately."""
    with tempfile.TemporaryDirectory() as tmpdir:
      tmp_file = pathlib.Path(tmpdir) / "faulty.png"
      tmp_file.write_bytes(b"data")

      # Mock read_bytes to raise OSError
      with mock.patch.object(
          pathlib.Path,
          "read_bytes",
          autospec=True,
          side_effect=OSError("disk read failure"),
      ):
        with self.assertRaisesRegex(OSError, "Failed to read file"):
          types.from_file(tmp_file)


class ChatResponseStreamTest(unittest.IsolatedAsyncioTestCase):
  """Tests for ChatResponse async stream and caching properties."""

  async def test_text_concatenation(self):
    """Verifies that text() aggregates and concatenates all Text chunks."""
    t1 = types.Text(step_index=1, text="Hello ")
    t2 = types.Thought(step_index=1, text="internal reasoning...")
    t3 = types.Text(step_index=2, text="world!")

    async def mock_stream():
      yield t1
      yield t2
      yield t3

    response = types.ChatResponse(
        mock_stream(),
        conversation=mock.MagicMock(spec=conversation.Conversation),
    )
    self.assertEqual(await response.text(), "Hello world!")

  async def test_thoughts_sugared_stream(self):
    """Verifies thoughts property yields only Thought delta strings."""
    t1 = types.Thought(step_index=1, text="Let me think...")
    t2 = types.Text(step_index=2, text="Standard response text.")

    async def mock_stream():
      yield t1
      yield t2

    response = types.ChatResponse(
        mock_stream(),
        conversation=mock.MagicMock(spec=conversation.Conversation),
    )
    thoughts = [token async for token in response.thoughts]
    self.assertEqual(thoughts, ["Let me think..."])

  async def test_tool_calls_sugared_stream(self):
    """Verifies tool_calls property yields only ToolCall objects."""
    t1 = types.Text(step_index=1, text="Invoking tool...")
    t2 = types.ToolCall(id="call_1", name="get_weather", args={})

    async def mock_stream():
      yield t1
      yield t2

    response = types.ChatResponse(
        mock_stream(),
        conversation=mock.MagicMock(spec=conversation.Conversation),
    )
    calls = [call async for call in response.tool_calls]
    self.assertEqual(len(calls), 1)
    self.assertEqual(calls[0].name, "get_weather")

  async def test_lazy_caching_and_re_iteration(self):
    """Verifies that stream tokens are cached in memory and replayed for re-iteration."""
    t1 = types.Text(step_index=1, text="A")
    t2 = types.Text(step_index=2, text="B")

    pull_count = 0

    async def mock_stream():
      nonlocal pull_count
      pull_count += 1
      yield t1
      yield t2

    response = types.ChatResponse(
        mock_stream(),
        conversation=mock.MagicMock(spec=conversation.Conversation),
    )

    # Iteration Round 1 (pulls from live stream and caches)
    with self.subTest("round_1_live_caching"):
      round_1 = [token async for token in response]
      self.assertEqual(round_1, ["A", "B"])
      self.assertEqual(pull_count, 1)

    # Iteration Round 2 (replays from buffered cache without pulling again)
    with self.subTest("round_2_cache_replay"):
      round_2 = [token async for token in response]
      self.assertEqual(round_2, ["A", "B"])
      self.assertEqual(pull_count, 1)

  async def test_structured_output_lazy_accessor(self):
    """Verifies structured_output resolves the stream and fetches parsed payload from conversation."""
    t_text = types.Text(step_index=1, text="finished")

    async def mock_stream():
      yield t_text

    mock_conv = mock.MagicMock(spec=conversation.Conversation)
    mock_conv.get_last_structured_output.return_value = {"result": "data"}

    response = types.ChatResponse(mock_stream(), conversation=mock_conv)

    data = await response.structured_output()
    self.assertEqual(data, {"result": "data"})
    mock_conv.get_last_structured_output.assert_called_once()

  async def test_usage_metadata_lazy_accessor(self):
    """Verifies usage_metadata resolves the stream and fetches payload from conversation."""
    t_text = types.Text(step_index=1, text="finished")

    async def mock_stream():
      yield t_text

    mock_conv = mock.MagicMock(spec=conversation.Conversation)
    mock_conv.last_turn_usage = types.UsageMetadata(
        prompt_token_count=10,
        candidates_token_count=20,
        total_token_count=30,
    )

    response = types.ChatResponse(mock_stream(), conversation=mock_conv)

    await response.resolve()
    usage = response.usage_metadata
    self.assertIsNotNone(usage)
    self.assertEqual(usage.prompt_token_count, 10)
    self.assertEqual(usage.candidates_token_count, 20)
    self.assertEqual(usage.total_token_count, 30)

  def test_thought_chunk_validation(self):
    """Verifies that the Thought subclass validates Pydantic schemas correctly."""
    thought = types.Thought(step_index=1, text="reasoning", signature=b"sig")
    self.assertEqual(thought.step_index, 1)
    self.assertEqual(thought.text, "reasoning")
    self.assertEqual(thought.signature, b"sig")

  def test_text_chunk_validation(self):
    """Verifies that the Text subclass validates Pydantic schemas correctly."""
    text = types.Text(step_index=2, text="conversational answer")
    self.assertEqual(text.step_index, 2)
    self.assertEqual(text.text, "conversational answer")

  async def test_empty_stream_text(self):
    """Verifies that an empty stream resolves text to an empty string."""

    async def mock_empty_stream():
      return
      yield

    response = types.ChatResponse(
        mock_empty_stream(),
        conversation=mock.MagicMock(spec=conversation.Conversation),
    )
    self.assertEqual(await response.text(), "")


if __name__ == "__main__":
  absltest.main()
