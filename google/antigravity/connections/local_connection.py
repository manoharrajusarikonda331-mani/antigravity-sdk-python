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

"""Local connection for the Antigravity SDK."""

import asyncio
import collections
import dataclasses
import importlib.resources
import json
import logging
import os
import shutil
import struct
import subprocess
import tempfile
import threading
import typing
from typing import Any, AsyncIterator, Callable

from google.genai import types as genai_types
from google.protobuf import json_format
import pydantic
import websockets

from google.antigravity import types
from google.antigravity.connections import connection
from google.antigravity.connections import localharness_pb2
from google.antigravity.hooks import hook_runner as h_runner
from google.antigravity.hooks import hooks
from google.antigravity.tools import tool_runner as t_runner


class LocalAgentConfig(connection.AgentConfig):
  """Configuration for the local harness backend.

  This is the default config for the Agent class. It uses the
  Go-based localharness binary.
  """

  gemini_config: types.GeminiConfig = pydantic.Field(
      default_factory=types.GeminiConfig
  )

  # Top-level shorthand fields — flow into gemini_config.
  model: str | None = None
  api_key: str | None = None

  @pydantic.model_validator(mode="after")
  def _apply_shorthand_configs(self) -> "LocalAgentConfig":
    """Applies top-level shorthand fields (model, api_key) to gemini_config."""
    # Defensive copy: prevent mutation of shared GeminiConfig instances.
    self.gemini_config = self.gemini_config.model_copy(deep=True)

    if self.model is not None:
      if "default" in self.gemini_config.models.model_fields_set:
        raise ValueError(
            "Cannot set both 'model' shorthand and "
            "'gemini_config.models.default'. Use one or the other."
        )
      self.gemini_config.models.default = types.ModelEntry(name=self.model)
    if self.api_key is not None:
      if self.gemini_config.api_key is not None:
        raise ValueError(
            "Cannot set both 'api_key' shorthand and "
            "'gemini_config.api_key'. Use one or the other."
        )
      self.gemini_config.api_key = self.api_key
    return self

  def create_strategy(
      self,
      *,
      tool_runner: Any,
      hook_runner: Any,
  ) -> "LocalConnectionStrategy":
    if isinstance(self.system_instructions, str):
      si = types.TemplatedSystemInstructions(
          sections=[
              types.SystemInstructionSection(content=self.system_instructions)
          ]
      )
    else:
      si = self.system_instructions

    save_dir = self.save_dir
    if save_dir is None:
      save_dir = tempfile.mkdtemp(prefix="antigravity_")
      logging.info("No save_dir specified; using %s", save_dir)

    return LocalConnectionStrategy(
        tool_runner=tool_runner,
        hook_runner=hook_runner,
        gemini_config=self.gemini_config,
        system_instructions=si,
        capabilities_config=self.capabilities,
        conversation_id=self.conversation_id,
        save_dir=save_dir,
        workspaces=self.workspaces,
        skills_paths=self.skills_paths,
    )


resources = None


@dataclasses.dataclass
class _StepTracker:
  """Tracks state and handled requests for a trajectory step to prevent non-linearity bugs."""

  state: int = localharness_pb2.StepUpdate.State.STATE_UNSPECIFIED
  handled_requests: set[str] = dataclasses.field(default_factory=set)

  def update_state(self, new_state: int) -> None:
    """Updates state and clears handled requests if transitioning out of waiting."""
    if (
        self.state == localharness_pb2.StepUpdate.State.STATE_WAITING_FOR_USER
        and new_state
        != localharness_pb2.StepUpdate.State.STATE_WAITING_FOR_USER
    ):
      self.handled_requests.clear()
    self.state = new_state

  def mark_handled(self, request_type: str) -> bool:
    """Marks a request as handled to prevent duplicate processing.

    Args:
        request_type: The string identifier of the request (e.g.
          "questions_request").

    Returns:
        bool: True if the request was newly marked as handled. False
        if it was already handled previously in this wait state.
    """
    if request_type in self.handled_requests:
      return False
    self.handled_requests.add(request_type)
    return True


_SOURCE_MAP = {
    "SOURCE_SYSTEM": types.StepSource.SYSTEM,
    "SOURCE_USER": types.StepSource.USER,
    "SOURCE_MODEL": types.StepSource.MODEL,
}

_STATUS_MAP = {
    "STATE_ACTIVE": types.StepStatus.ACTIVE,
    "STATE_DONE": types.StepStatus.DONE,
    "STATE_WAITING_FOR_USER": types.StepStatus.WAITING_FOR_USER,
    "STATE_ERROR": types.StepStatus.ERROR,
}

# Map from BuiltinTools enum to the proto field name on StepUpdate.
# Used for (a) determining step type and (b) extracting tool-confirmation args.
# Kept as an explicit map because enum values and proto field names may diverge.
_BUILTIN_TOOL_PROTO_FIELDS: dict[types.BuiltinTools, str] = {
    types.BuiltinTools.CREATE_FILE: "create_file",
    types.BuiltinTools.EDIT_FILE: "edit_file",
    types.BuiltinTools.FIND_FILE: "find_file",
    types.BuiltinTools.LIST_DIR: "list_directory",
    types.BuiltinTools.RUN_COMMAND: "run_command",
    types.BuiltinTools.SEARCH_DIR: "search_directory",
    types.BuiltinTools.VIEW_FILE: "view_file",
    types.BuiltinTools.START_SUBAGENT: "invoke_subagent",
    types.BuiltinTools.GENERATE_IMAGE: "generate_image",
    types.BuiltinTools.FINISH: "finish",
}

# Fallback action name used when a tool confirmation request does not match any
# known BuiltinTools proto field. This represents a pre-request notification
# from the Connection for a host-side tool whose specific call will follow.
DEFAULT_HOST_TOOL_NAME = "pre_request_host_tool_request"


class LocalConnectionStep(types.Step):
  """Connection-specific step for LocalConnection."""

  cascade_id: str = ""
  trajectory_id: str = ""
  target: str = ""

  @classmethod
  def from_dict(cls, step_dict: dict[str, Any]) -> "LocalConnectionStep":
    """Creates a LocalConnectionStep from a dictionary representation of StepUpdate."""
    traj_id = step_dict.get("trajectory_id", "")
    step_idx = step_dict.get("step_index", 0)

    id_str = f"{traj_id}-{step_idx}" if traj_id else str(step_idx)

    tc_dict = step_dict.get("tool_call")
    tool_calls = []
    if tc_dict:
      tool_calls.append(
          types.ToolCall(
              name=tc_dict["name"],
              args=tc_dict.get("args", {}),
              id=tc_dict.get("id"),
          )
      )

    # Determine high-level type
    step_type = types.StepType.UNKNOWN
    if step_dict.get("compaction") is not None:
      step_type = types.StepType.COMPACTION
    elif step_dict.get("finish") is not None:
      step_type = types.StepType.FINISH
    elif tc_dict or any(
        step_dict.get(k) is not None
        for k in _BUILTIN_TOOL_PROTO_FIELDS.values()
    ):
      step_type = types.StepType.TOOL_CALL
    elif step_dict.get("text"):
      step_type = types.StepType.MODEL_RESPONSE

    source_str = step_dict.get("source")
    source = _SOURCE_MAP.get(source_str, types.StepSource.UNKNOWN)

    status_str = step_dict.get("state")
    status = _STATUS_MAP.get(status_str, types.StepStatus.UNKNOWN)

    is_from_model = source == types.StepSource.MODEL
    is_done = status == types.StepStatus.DONE
    has_text = bool(step_dict.get("text"))
    is_target_user = step_dict.get("target") == "TARGET_USER"
    # The idle signal (trajectory_state_update / STATE_IDLE) arrives as a
    # separate event type with no text content, so we cannot retroactively
    # mark a step at idle time. Instead, we flag each step that is a
    # completed model response directed at the user. Multiple steps per
    # turn may carry this flag; consumers that want the *last* response
    # should iterate fully (Conversation.chat() does this).
    is_complete_response = (
        is_from_model and is_done and has_text and is_target_user
    )

    structured_output = None
    if step_type == types.StepType.FINISH:
      finish_dict = step_dict.get("finish", {})
      output_string = finish_dict.get("output_string")
      if output_string:
        try:
          structured_output = json.loads(output_string)
        except json.JSONDecodeError:
          logging.warning(
              "Failed to parse structured output JSON.", exc_info=True
          )

    return cls(
        id=id_str,
        step_index=step_idx,
        cascade_id=step_dict.get("cascade_id", ""),
        trajectory_id=traj_id,
        type=step_type,
        source=source,
        status=status,
        content=step_dict.get("text", ""),
        content_delta=step_dict.get("text_delta", ""),
        thinking=step_dict.get("thinking", ""),
        thinking_delta=step_dict.get("thinking_delta", ""),
        tool_calls=tool_calls,
        error=step_dict.get("error_message", ""),
        is_complete_response=is_complete_response,
        target=step_dict.get("target", ""),
        structured_output=structured_output,
    )


class _PendingCallKey(typing.NamedTuple):
  """Key for tracking approved built-in tool calls."""

  trajectory_id: str
  step_index: int


class _PendingCallValue(typing.NamedTuple):
  """Value for tracking approved built-in tool calls."""

  tool_call: types.ToolCall
  operation_context: hooks.OperationContext


# TODO(karmel): Split these result types into a dedicated types file
# in a local/ directory once the LocalConnection package is refactored.


class RunCommandResult(pydantic.BaseModel):
  """Structured result from a run_command tool execution."""

  output: str = ""

  def __str__(self) -> str:
    return self.output


class ListDirectoryEntry(pydantic.BaseModel):
  """Single entry in a directory listing."""

  name: str
  is_directory: bool = False
  file_size: int = 0


class ListDirectoryResult(pydantic.BaseModel):
  """Structured result from a list_directory tool execution."""

  entries: list[ListDirectoryEntry] = pydantic.Field(default_factory=list)

  def __str__(self) -> str:
    parts = []
    for e in self.entries:
      if e.is_directory:
        parts.append(f"{e.name}/ (dir)")
      else:
        parts.append(f"{e.name} ({e.file_size} bytes)")
    return os.linesep.join(parts)


class SearchDirectoryResult(pydantic.BaseModel):
  """Structured result from a search_directory tool execution."""

  num_results: int = 0

  def __str__(self) -> str:
    return f"{self.num_results} results"


class FindFileResult(pydantic.BaseModel):
  """Structured result from a find_file tool execution."""

  output: str = ""

  def __str__(self) -> str:
    return self.output


class EditFileResult(pydantic.BaseModel):
  """Structured result from an edit_file tool execution."""

  summary: str = ""

  def __str__(self) -> str:
    return self.summary


class GenerateImageResult(pydantic.BaseModel):
  """Structured result from a generate_image tool execution."""

  image_name: str = ""

  def __str__(self) -> str:
    return self.image_name


class TextResult(pydantic.BaseModel):
  """Generic fallback for tools without structured output (e.g. view_file)."""

  text: str = ""

  def __str__(self) -> str:
    return self.text


# Union of all structured tool result types.
ToolOutput = (
    RunCommandResult
    | ListDirectoryResult
    | SearchDirectoryResult
    | FindFileResult
    | EditFileResult
    | GenerateImageResult
    | TextResult
)


def _extract_tool_result(
    step_update: localharness_pb2.StepUpdate,
) -> ToolOutput | None:
  """Extracts a structured tool result from per-action fields.

  The Go harness populates result data on the action sub-messages of
  StepUpdate (e.g. ActionRunCommand.combined_output,
  ActionListDirectory.results) when the step transitions to STATE_DONE.
  This function inspects each action field and returns a typed result
  object.

  Each StepUpdate corresponds to a single tool execution, so at most one
  action field will be set. The elif chain returns the first match.

  Args:
    step_update: The StepUpdate proto from the harness.

  Returns:
    A typed result object, or None if no structured result is present.
  """
  _run_command = _BUILTIN_TOOL_PROTO_FIELDS[types.BuiltinTools.RUN_COMMAND]
  _list_dir = _BUILTIN_TOOL_PROTO_FIELDS[types.BuiltinTools.LIST_DIR]
  _find_file = _BUILTIN_TOOL_PROTO_FIELDS[types.BuiltinTools.FIND_FILE]
  _search_dir = _BUILTIN_TOOL_PROTO_FIELDS[types.BuiltinTools.SEARCH_DIR]
  _edit_file = _BUILTIN_TOOL_PROTO_FIELDS[types.BuiltinTools.EDIT_FILE]
  _gen_image = _BUILTIN_TOOL_PROTO_FIELDS[types.BuiltinTools.GENERATE_IMAGE]

  # run_command -> raw stdout/stderr, e.g. "hello world\n"
  if step_update.HasField(_run_command):
    rc = step_update.run_command
    if rc.combined_output:
      return RunCommandResult(output=rc.combined_output)
  # list_directory -> structured entry list
  elif step_update.HasField(_list_dir):
    ld = step_update.list_directory
    if ld.results:
      entries = [
          ListDirectoryEntry(
              name=r.name,
              is_directory=r.is_directory,
              file_size=r.file_size,
          )
          for r in ld.results
      ]
      return ListDirectoryResult(entries=entries)
  # find_file -> raw find output, e.g. "/path/to/foo.py\n/path/to/bar.py"
  elif step_update.HasField(_find_file):
    ff = step_update.find_file
    if ff.output:
      return FindFileResult(output=ff.output)
  # search_directory -> result count, e.g. "3 results"
  elif step_update.HasField(_search_dir):
    sd = step_update.search_directory
    if sd.num_results:
      return SearchDirectoryResult(num_results=sd.num_results)
  # edit_file -> diff summary from step text
  elif step_update.HasField(_edit_file):
    ef = step_update.edit_file
    if ef.diff_block:
      return EditFileResult(summary=step_update.text)
  # generate_image -> image filename, e.g. "sunset_photo"
  elif step_update.HasField(_gen_image):
    gi = step_update.generate_image
    if gi.image_name:
      return GenerateImageResult(image_name=gi.image_name)
  return None


def callable_to_tool_proto(
    fn: Callable[..., Any],
    tool_runner: t_runner.ToolRunner | None = None,
) -> localharness_pb2.Tool:
  """Converts a Python callable to a localharness Tool proto.

  Uses google.genai.types.FunctionDeclaration for schema extraction.
  If a ``tool_runner`` is provided, the runner's ``get_public_callable``
  is used to strip injectable parameters (e.g. ``ToolContext``) from
  the schema so the model never sees them.

  Args:
      fn: The Python callable to convert.
      tool_runner: Optional ToolRunner that owns schema-hiding logic.

  Returns:
      A localharness_pb2.Tool proto.
  """
  if isinstance(fn, t_runner.ToolWithSchema):
    return localharness_pb2.Tool(
        name=fn.__name__,
        description=fn.__doc__ or "",
        parameters_json_schema=json.dumps(fn.input_schema),
    )

  # Use the ToolRunner's public callable to strip injectable params.
  target_fn = fn
  if tool_runner is not None:
    tool_name = fn.__name__
    if tool_name in tool_runner.tools:
      target_fn = tool_runner.get_public_callable(tool_name)

  decl = genai_types.FunctionDeclaration.from_callable_with_api_option(
      callable=target_fn,
      api_option="GEMINI_API",
  )
  if decl.parameters:
    parameters = decl.parameters.model_dump(exclude_none=True)
  elif decl.parameters_json_schema:
    parameters = decl.parameters_json_schema
  else:
    parameters = {"type": "OBJECT"}
  return localharness_pb2.Tool(
      name=decl.name,
      description=decl.description or "",
      parameters_json_schema=json.dumps(parameters),
  )


class LocalConnection(connection.Connection):
  """Connection to the Go-based local harness."""

  def __init__(
      self,
      process: subprocess.Popen[bytes],
      ws: Any,
      tool_runner: t_runner.ToolRunner | None = None,
      hook_runner: h_runner.HookRunner | None = None,
  ):
    self._hook_runner = hook_runner
    self._process = process
    self._ws = ws
    self._tool_runner = tool_runner
    self._step_trackers: dict[tuple[str, int], _StepTracker] = {}
    self._step_queue = asyncio.Queue()
    self._background_tasks = set()
    self._reader_task = asyncio.create_task(self._ws_reader_loop())
    self._current_turn_context = None
    self._cancelled = False
    self._cancelled_message = ""
    self._is_idle = asyncio.Event()
    self._is_idle.set()
    # Set of trajectory IDs for currently-running subagents. The connection
    # is only considered idle when the parent trajectory is idle AND this
    # set is empty, ensuring post-tool-call hooks for subagent completions
    # fire before receive_steps() returns.
    self._active_subagent_ids: set[str] = set()
    # Maps subagent trajectory_id -> final response content. Populated
    # when the reader loop sees an is_complete_response step from a subagent
    # trajectory, and consumed when that trajectory goes idle.
    self._subagent_responses: dict[str, str] = {}
    self._parent_idle = True
    # The cascade_id from step updates identifies the parent trajectory.
    # A step belongs to the parent trajectory when cascade_id ==
    # trajectory_id; otherwise it belongs to a subagent trajectory.
    # We store the cascade_id so TrajectoryStateUpdate (which lacks the
    # field) can distinguish parent vs. subagent trajectories.
    self._cascade_id: str | None = None

    # Flag set early in disconnect() so the reader loop can distinguish
    # expected closures from harness crashes.
    self._disconnecting = False

    # Stderr lines from the Go harness, captured by a background thread.
    # Retained in a bounded deque so the reader loop can surface harness
    # error messages when the WebSocket closes unexpectedly.
    self._stderr_lines: collections.deque[str] = collections.deque(maxlen=100)
    self._stderr_thread: threading.Thread | None = None

    # Tracks builtin tool calls that were approved via ToolConfirmation,
    # keyed by (trajectory_id, step_index). When the step transitions to
    # STATE_DONE or STATE_ERROR, we dispatch PostToolCallHook or
    # OnToolErrorHook respectively.
    self._pending_builtin_tool_calls: dict[
        _PendingCallKey, _PendingCallValue
    ] = {}

    # Dispatch session start hook.
    if self._hook_runner and self._hook_runner.on_session_start_hooks:
      self._run_in_background(self._hook_runner.dispatch_session_start())

  @property
  def is_idle(self) -> bool:
    """Returns True if the connection is idle and ready for input."""
    return self._is_idle.is_set()

  @property
  def conversation_id(self) -> str:
    """Returns the conversation identifier, if one exists."""
    return self._cascade_id or ""

  async def send(self, prompt: types.Content | None) -> None:
    """Sends a prompt to the agent."""
    self._cancelled = False
    self._is_idle.clear()
    self._parent_idle = False
    self._active_subagent_ids.clear()
    self._subagent_responses.clear()
    if self._hook_runner:
      res, turn_context = await self._hook_runner.dispatch_pre_turn(prompt)
      self._current_turn_context = turn_context
      if not res.allow:
        logging.warning("Turn denied by hook: %s", res.message)
        self._cancelled = True
        self._cancelled_message = (
            res.message or "Turn execution denied by hook."
        )
        self._is_idle.set()
        return

    if prompt is None:
      event = localharness_pb2.InputEvent(user_input="")
    elif isinstance(prompt, str):
      event = localharness_pb2.InputEvent(user_input=prompt)
    else:
      content_list = prompt if isinstance(prompt, list) else [prompt]
      user_input_pb = localharness_pb2.UserInput(
          parts=[_to_proto_input_content(c) for c in content_list]
      )
      event = localharness_pb2.InputEvent(complex_user_input=user_input_pb)

    await self._ws.send(json_format.MessageToJson(event))

  async def receive_steps(self) -> AsyncIterator[LocalConnectionStep]:
    """Receives steps as they complete from the agent."""
    if self._cancelled:
      yield LocalConnectionStep(
          status=types.StepStatus.CANCELED,
          error=self._cancelled_message,
          source=types.StepSource.SYSTEM,
          type=types.StepType.SYSTEM_MESSAGE,
      )
      return

    if self._is_idle.is_set() and self._step_queue.empty():
      return

    # The server sends a STATE_IDLE signal when the trajectory is finalized,
    # but it may arrive before we've consumed all queued steps (the reader
    # loop and this generator run concurrently). We check idle + empty as
    # the exit condition and block on get() otherwise.
    while True:
      if self._is_idle.is_set() and self._step_queue.empty():
        return

      step_obj = await self._step_queue.get()

      if step_obj is None:
        return
      if isinstance(step_obj, Exception):
        raise step_obj

      yield step_obj

      is_from_model = step_obj.source == types.StepSource.MODEL
      is_done = step_obj.status == types.StepStatus.DONE
      is_terminal = is_done or step_obj.status in (
          types.StepStatus.ERROR,
          types.StepStatus.CANCELED,
      )
      is_target_user = getattr(step_obj, "target", None) == "TARGET_USER"

      if is_terminal and is_target_user and is_from_model:
        # Dispatch post-turn hook with the final response content.
        if self._hook_runner and self._current_turn_context:
          await self._hook_runner.dispatch_post_turn(
              self._current_turn_context, step_obj.content or ""
          )
          self._current_turn_context = None
        # Don't force idle here — wait for the TrajectoryStateUpdate
        # path to confirm that the parent and all subagent trajectories
        # have completed.

  async def wait_for_idle(self) -> None:
    """Blocks until the connection becomes idle."""
    # Drain all pending steps
    async for _ in self.receive_steps():
      pass

  def _start_stderr_reader(self, stderr_stream) -> None:
    """Starts a background daemon thread that drains the harness stderr.

    The Go harness writes diagnostic messages to stderr.  If the OS
    pipe buffer fills (typically 64 KiB on Linux), the harness blocks
    on its next write and cannot save trajectory state at shutdown.
    This thread prevents that by reading continuously and storing the
    most recent lines in a bounded deque for later diagnostics.

    Args:
      stderr_stream: The binary stderr stream from the harness process.
    """

    def _drain():
      try:
        for raw_line in stderr_stream:
          line = raw_line.decode("utf-8", errors="replace").rstrip()
          self._stderr_lines.append(line)
          logging.debug("harness stderr: %s", line)
      except ValueError:
        pass  # Stream closed.

    t = threading.Thread(target=_drain, daemon=True, name="harness-stderr")
    t.start()
    self._stderr_thread = t

  async def disconnect(self) -> None:
    """Tears down the harness connection in a careful order.

    Shutdown sequence:

    1. Dispatch the ``on_session_end`` hook so user code can react.
    2. Cancel background tasks (pending hook dispatches, etc.).
    3. Cancel the WebSocket reader task.
    4. Close the WebSocket (0.5 s timeout).  This triggers the Go
       handler's ``defer`` block, which calls ``agent.Close()`` and
       serializes the trajectory.
    5. Close stdin.  The Go main loop detects EOF and runs
       ``cleanupAllAgents`` → ``os.Exit(0)``.
    6. Wait for the process to exit.  The trajectory write is
       sub-second, so a generous 5 s is more than enough.
    7. If the process is still alive, escalate: SIGTERM (1 s wait)
       then SIGKILL (1 s wait).

    ``stdout`` is already closed after the ``OutputConfig`` handshake
    since it is only used for the initial ``OutputConfig`` handshake.

    After closing stdin the process should exit on its own.  We wait up
    to 5 s, then escalate to SIGTERM, then SIGKILL.
    """
    self._disconnecting = True
    hook_error = None

    # Dispatch session end hook before tearing down. If the hook raises,
    # capture the error but still proceed with graceful cleanup.
    if self._hook_runner and self._hook_runner.on_session_end_hooks:
      try:
        await self._hook_runner.dispatch_session_end()
      except Exception as e:  # pylint: disable=broad-except
        hook_error = e

    try:
      # Cancel and await background tasks (e.g., pending hook dispatches).
      for task in self._background_tasks:
        task.cancel()
      if self._background_tasks:
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
      self._background_tasks.clear()

      self._reader_task.cancel()
      try:
        await self._reader_task
      except asyncio.CancelledError:
        pass

      # Close the WebSocket first.  This triggers the Go handler's
      # defer block which calls agent.Close() and serializes the
      # trajectory.  The Go server does not send a response Close
      # frame, so we use a short timeout.
      try:
        await asyncio.wait_for(self._ws.close(), timeout=0.5)
      except asyncio.TimeoutError:
        pass

      # Close stdin to signal the Go main loop to exit.  On EOF the
      # harness runs cleanupAllAgents and calls os.Exit(0).
      if self._process.stdin:
        self._process.stdin.close()

      # Wait for the process to exit, escalating if needed.
      try:
        self._process.wait(timeout=5)
      except subprocess.TimeoutExpired:
        self._process.terminate()
        try:
          self._process.wait(timeout=1)
        except subprocess.TimeoutExpired:
          self._process.kill()
          self._process.wait(timeout=1)
    finally:
      if hook_error is not None:
        raise hook_error

  async def cancel(self) -> None:
    """Cancels the current turn."""
    event = localharness_pb2.InputEvent(halt_request=True)
    await self._ws.send(json_format.MessageToJson(event))

  def _get_turn_context(self) -> hooks.TurnContext:
    """Returns the current turn context, creating one if needed.

    Callers must ensure self._hook_runner is not None before calling.
    """
    assert self._hook_runner is not None
    return self._current_turn_context or hooks.TurnContext(
        self._hook_runner.session_context
    )

  def _run_in_background(self, coro) -> None:
    """Schedules a coroutine as a fire-and-forget background task."""
    t = asyncio.create_task(coro)
    self._background_tasks.add(t)
    t.add_done_callback(self._background_tasks.discard)

  async def _ws_reader_loop(self) -> None:
    """Reads OutputEvents from the WebSocket, routes steps, and dispatches tools."""
    try:
      async for raw_msg in self._ws:
        logging.info("RAW WS MSG: %s", raw_msg)
        event = localharness_pb2.OutputEvent()
        json_format.Parse(raw_msg, event)
        if event.HasField("step_update"):
          step_update = event.step_update

          # 1. Update local step tracker state to handle multiple transitions
          step_key = (step_update.trajectory_id, step_update.step_index)
          if step_key not in self._step_trackers:
            self._step_trackers[step_key] = _StepTracker()

          tracker = self._step_trackers[step_key]
          tracker.update_state(step_update.state)

          # 2. Always push the step update to the queue so that Layer 2
          #    and the UI have an accurate representation of the state.
          step_dict = json_format.MessageToDict(
              event.step_update, preserving_proto_field_name=True
          )
          step_obj = LocalConnectionStep.from_dict(step_dict)

          # 3. Attach usage_metadata from the OutputEvent.
          if event.HasField("usage_metadata"):
            step_obj.usage_metadata = _parse_usage_metadata(
                event.usage_metadata
            )
          await self._step_queue.put(step_obj)

          # Record the cascade_id for use by TrajectoryStateUpdate
          # (which does not carry a cascade_id field).
          if (
              step_update.cascade_id
              and step_update.cascade_id == step_update.trajectory_id
          ):
            self._cascade_id = step_update.cascade_id

          # 4. Dispatch observe-only hooks for special step types.
          if step_obj.type == types.StepType.COMPACTION and self._hook_runner:
            self._run_in_background(
                self._hook_runner.dispatch_compaction(
                    self._get_turn_context(), step_obj
                )
            )

          # Track the last model response from subagent trajectories so we
          # can include it in the post-tool-call ToolResult. Subagent results
          # are delivered by the harness as TrajectoryStateUpdate events, not
          # as tool responses; we capture the last model text here so the
          # PostToolCallHook receives the subagent's actual output.
          is_subagent_step = (
              self._cascade_id
              and step_obj.trajectory_id
              and step_obj.trajectory_id != self._cascade_id
          )
          if (
              is_subagent_step
              and step_obj.source == types.StepSource.MODEL
              and step_obj.content
          ):
            self._subagent_responses[step_obj.trajectory_id] = (
                step_obj.content
            )

          # Dispatch post-tool-call or on-tool-error hooks for built-in tools
          # that were approved via ToolConfirmation. The harness executes them
          # internally; we observe completion via step state transitions.
          if (
              step_key in self._pending_builtin_tool_calls
              and step_update.state
              == localharness_pb2.StepUpdate.State.STATE_DONE
          ):
            tc, op_ctx = self._pending_builtin_tool_calls.pop(step_key)
            if self._hook_runner:
              extracted = _extract_tool_result(step_update)
              result = types.ToolResult(
                  name=tc.name,
                  id=tc.id,
                  result=extracted or step_obj.content,
              )
              self._run_in_background(
                  self._hook_runner.dispatch_post_tool_call(op_ctx, result)
              )
          elif (
              step_key in self._pending_builtin_tool_calls
              and step_update.state
              == localharness_pb2.StepUpdate.State.STATE_ERROR
          ):
            tc, op_ctx = self._pending_builtin_tool_calls.pop(step_key)
            if self._hook_runner:
              error = RuntimeError(
                  step_update.error_message
                  or step_obj.content
                  or "Built-in tool failed"
              )
              self._run_in_background(
                  self._hook_runner.dispatch_on_tool_error(op_ctx, error)
              )

          # 4. Process wait requests if this is a wait state
          if (
              step_update.state
              == localharness_pb2.StepUpdate.State.STATE_WAITING_FOR_USER
          ):
            # We execute handlers as background tasks instead of awaiting them.
            # This is critical for concurrency and non-linearity:
            # - If we block the loop, other parallel subagents are starved.
            # - The local harness broadcasts the active state whenever an
            #   internal state machine tick occurs (e.g., a parallel subagent
            #   emitting text). Therefore, this branch will receive the exact
            #   same `questions_request` or `tool_confirmation_request` multiple
            #   times while waiting for a human.
            #   We use `tracker.mark_handled()` to debounce and ensure we only
            #   launch one background task per request.
            if step_update.HasField("questions_request"):
              if tracker.mark_handled("questions_request"):
                self._run_in_background(
                    self._handle_question_request(step_update)
                )

            if step_update.HasField("tool_confirmation_request"):
              if tracker.mark_handled("tool_confirmation_request"):
                self._run_in_background(
                    self._handle_tool_confirmation_request(step_update)
                )
        elif event.HasField("trajectory_state_update"):
          tsu = event.trajectory_state_update
          is_subagent = (
              self._cascade_id and tsu.trajectory_id != self._cascade_id
          )

          if (
              tsu.state
              == localharness_pb2.TrajectoryStateUpdate.State.STATE_RUNNING
          ):
            if is_subagent:
              self._active_subagent_ids.add(tsu.trajectory_id)

          elif (
              tsu.state
              == localharness_pb2.TrajectoryStateUpdate.State.STATE_IDLE
          ):
            # Dispatch post-tool-call hook if this is a subagent trajectory.
            if is_subagent:
              self._active_subagent_ids.discard(tsu.trajectory_id)
              if self._hook_runner:
                op_ctx = hooks.OperationContext(self._get_turn_context())
                response = self._subagent_responses.pop(
                    tsu.trajectory_id, ""
                )
                result = types.ToolResult(
                    name=types.BuiltinTools.START_SUBAGENT.value,
                    result=response or tsu.trajectory_id,
                )
                await self._hook_runner.dispatch_post_tool_call(
                    op_ctx, result
                )
            else:
              # Parent trajectory went idle.
              self._parent_idle = True

            # The connection is idle when the parent trajectory is idle
            # and all subagent trajectories have completed.
            if self._parent_idle and not self._active_subagent_ids:
              self._is_idle.set()
        elif event.HasField("tool_call"):
          self._run_in_background(self._handle_tool_call(event.tool_call))
    except websockets.ConnectionClosed as e:
      if self._disconnecting:
        # Expected closure — disconnect() was called.
        logging.info("WebSocket closed (code %s); normal shutdown.", e.code)
      else:
        # Unexpected closure — the harness process likely crashed.
        # Surface the harness stderr so callers get actionable context.
        stderr_tail = "\n".join(self._stderr_lines) or "(no stderr output)"
        error_msg = (
            f"Harness process exited unexpectedly (WS close code {e.code})."
            f"\nHarness stderr:\n{stderr_tail}"
        )
        logging.error(error_msg)
        await self._step_queue.put(types.AntigravityConnectionError(error_msg))

    except Exception as e:  # pylint: disable=broad-except
      logging.exception("Error in reader loop: %s", e)
      await self._step_queue.put(
          types.AntigravityConnectionError(f"Error in reader loop: {e}")
      )
    finally:
      await self._step_queue.put(None)  # Send sentinel

  async def _handle_question_request(
      self, step_update: localharness_pb2.StepUpdate
  ) -> None:
    """Handles question requests from the harness."""
    questions_list = []
    for uq in step_update.questions_request.questions:
      if uq.HasField("multiple_choice"):
        mc = uq.multiple_choice
        opts = [
            types.AskQuestionOption(id=str(i + 1), text=choice)
            for i, choice in enumerate(mc.choices)
        ]
        questions_list.append(
            types.AskQuestionEntry(question=mc.question, options=opts)
        )

    answers = []
    if self._hook_runner:
      ctx = self._current_turn_context or hooks.TurnContext(
          self._hook_runner.session_context
      )
      _, question_res, _ = await self._hook_runner.dispatch_interaction(
          turn_context=ctx,
          interaction_spec=types.AskQuestionInteractionSpec(
              questions=questions_list
          ),
      )
      for r in question_res.responses:
        ans = localharness_pb2.UserQuestionAnswer()
        if r.skipped:
          ans.unanswered = True
        else:
          mc_ans = localharness_pb2.MultipleChoiceAnswer()
          if r.selected_option_ids:
            indices = []
            for opt_id in r.selected_option_ids:
              try:
                indices.append(int(opt_id) - 1)
              except ValueError:
                pass
            mc_ans.selected_choice_indices[:] = indices
          if r.freeform_response:
            mc_ans.freeform_response = r.freeform_response
          ans.multiple_choice_answer.CopyFrom(mc_ans)
        answers.append(ans)
    else:
      logging.warning(
          "Received question_request but no HookRunner is configured. Skipping."
      )
      answers = [
          localharness_pb2.UserQuestionAnswer(unanswered=True)
          for _ in questions_list
      ]

    resp = localharness_pb2.UserQuestionsResponse(
        trajectory_id=step_update.trajectory_id,
        step_index=step_update.step_index,
        response=localharness_pb2.UserQuestionsResponse.QuestionsResponse(
            answers=answers
        ),
    )
    input_event = localharness_pb2.InputEvent(question_response=resp)
    await self._ws.send(json_format.MessageToJson(input_event))

  async def _handle_tool_confirmation_request(
      self, step_update: localharness_pb2.StepUpdate
  ) -> None:
    """Handles tool confirmation requests from the harness."""
    action_str = "unknown"
    args = {}
    found_action = False

    for tool_enum, proto_field in _BUILTIN_TOOL_PROTO_FIELDS.items():
      if step_update.HasField(proto_field):
        action_str = tool_enum.value
        found_action = True
        sub_msg = getattr(step_update, proto_field)
        args = json_format.MessageToDict(
            sub_msg, preserving_proto_field_name=True
        )
        break

    if not found_action:
      action_str = DEFAULT_HOST_TOOL_NAME

    if step_update.request_text:
      args["request_text"] = step_update.request_text

    tc = types.ToolCall(name=action_str, args=args)
    allow = True
    op_ctx = None
    # Auto-approve pre-requests for host tools because the actual tool call will
    # be sent next with its proper name and arguments, triggering its own
    # confirmation.
    if tc.name == DEFAULT_HOST_TOOL_NAME:
      allow = True
    elif self._hook_runner:
      ctx = self._current_turn_context or hooks.TurnContext(
          self._hook_runner.session_context
      )
      res, _, op_ctx = await self._hook_runner.dispatch_pre_tool_call(
          turn_context=ctx, tool_call=tc
      )
      allow = res.allow

    # Track approved built-in tool calls so we can dispatch PostToolCallHook
    # when the step transitions to STATE_DONE.
    if allow and tc.name != DEFAULT_HOST_TOOL_NAME and self._hook_runner:
      if op_ctx is None:
        ctx = self._current_turn_context or hooks.TurnContext(
            self._hook_runner.session_context
        )
        op_ctx = hooks.OperationContext(ctx)
      pending_key = _PendingCallKey(
          trajectory_id=step_update.trajectory_id,
          step_index=step_update.step_index,
      )
      self._pending_builtin_tool_calls[pending_key] = _PendingCallValue(
          tool_call=tc, operation_context=op_ctx,
      )

    resp = localharness_pb2.ToolConfirmation(
        trajectory_id=step_update.trajectory_id,
        step_index=step_update.step_index,
        accepted=allow,
    )
    input_event = localharness_pb2.InputEvent(tool_confirmation=resp)
    await self._ws.send(json_format.MessageToJson(input_event))

  async def _handle_tool_call(
      self, tool_call: localharness_pb2.ToolCall
  ) -> None:
    """Handles tool execution and hook interception."""
    args = json.loads(tool_call.arguments_json or "{}")

    tc = types.ToolCall(name=tool_call.name, args=args)
    op_context = None

    if self._hook_runner:
      ctx = self._current_turn_context or hooks.TurnContext(
          self._hook_runner.session_context
      )
      res, tc, op_context = await self._hook_runner.dispatch_pre_tool_call(
          turn_context=ctx, tool_call=tc
      )

      if not res.allow:
        reason = res.message or "No reason provided"
        err_msg = f"Tool execution denied by hook policy: {reason}"
        await self.send_tool_results([
            types.ToolResult(
                id=tool_call.id,
                name=tool_call.name,
                error=err_msg,
            ),
        ])
        return

    if self._tool_runner:
      tool_error: Exception | None = None
      try:
        results = await self._tool_runner.process_tool_calls(
            [types.ToolCall(name=tc.name, args=tc.args)]
        )
        result = results[0]
        result.id = tool_call.id
        # ToolRunner may catch exceptions internally and set result.error.
        if result.error:
          tool_error = result.exception or RuntimeError(result.error)
      except Exception as e:  # pylint: disable=broad-except
        tool_error = e
        result = types.ToolResult(
            id=tool_call.id,
            name=tool_call.name,
            error=f"Tool execution failed: {e}",
        )

      # Dispatch on-tool-error hook when the result carries an error.
      if tool_error and self._hook_runner:
        if not op_context:
          op_context = hooks.OperationContext(self._get_turn_context())
        recovery_res, recovery_val = (
            await self._hook_runner.dispatch_on_tool_error(
                op_context, tool_error
            )
        )
        if recovery_res.allow and recovery_val is not None:
          result = types.ToolResult(
              id=tool_call.id,
              name=tool_call.name,
              result=recovery_val,
          )

      # Dispatch post-tool-call hook on success.
      elif not result.error and self._hook_runner:
        if not op_context:
          op_context = hooks.OperationContext(self._get_turn_context())
        await self._hook_runner.dispatch_post_tool_call(op_context, result)

      await self.send_tool_results([result])
    else:
      logging.warning(
          "Received tool call %s but no tool runner is configured. "
          "Yielding to user.",
          tool_call.name,
      )
      step_dict = {
          "type": "tool_call",
          "state": "STATE_ACTIVE",
          "tool_call": {
              "name": tool_call.name,
              "args": args,
              "id": tool_call.id,
          },
      }
      await self._step_queue.put(LocalConnectionStep.from_dict(step_dict))

  def _tool_result_to_dict(self, result: types.ToolResult) -> dict[str, Any]:
    if result.error is not None:
      return {"error": result.error}

    output = result.result
    if hasattr(output, "model_dump"):
      output = output.model_dump()
    elif hasattr(output, "dict"):
      output = output.dict()

    if not isinstance(output, dict):
      if not isinstance(output, (str, int, float, bool, type(None))):
        output = str(output)
      return {"result": output}

    return output

  async def send_tool_results(self, results: list[types.ToolResult]) -> None:
    """Sends tool execution results back to the harness.

    Args:
      results: ToolResult instances. The id field is used to correlate each
        result with the original ToolCall.
    """
    for result in results:
      if not result.id:
        raise ValueError(
            f"ToolResult for '{result.name}' is missing an id. The"
            " LocalConnection protocol requires an id to correlate results"
            " with calls."
        )
      response = localharness_pb2.ToolResponse(
          id=result.id,
          response_json=json.dumps(self._tool_result_to_dict(result)),
      )
      input_event = localharness_pb2.InputEvent(tool_response=response)
      await self._ws.send(json_format.MessageToJson(input_event))

  async def send_trigger_notification(self, content: str) -> None:
    """Sends a trigger message to the agent."""
    event = localharness_pb2.InputEvent(automated_trigger=content)
    await self._ws.send(json_format.MessageToJson(event))


def _to_proto_input_content(
    content: types.ContentPrimitive,
) -> localharness_pb2.UserInput.Part:
  """Converts dynamic prompt fragments into proto Parts."""
  if isinstance(content, str):
    return localharness_pb2.UserInput.Part(text=content)

  is_semantic_media = isinstance(
      content, (types.Image, types.Document, types.Audio, types.Video)
  )
  if is_semantic_media:
    media_pb = localharness_pb2.UserInput.Media(
        mime_type=content.mime_type,
        data=content.data,
        description=content.description,
    )
    return localharness_pb2.UserInput.Part(media=media_pb)

  raise TypeError(f"Unsupported prompt content type: {type(content)}")


def _get_default_binary_path() -> str:
  """Finds the default binary path, supporting both internal and external wheels."""
  # 1. Check environment variable first
  if env_path := os.environ.get("ANTIGRAVITY_HARNESS_PATH"):
    return env_path


  # 3. Try importlib.resources (External Wheel)
  try:
    # Using 'google.antigravity' as the package name.
    # This assumes the binary is located at google/antigravity/bin/localharness
    # in the installed package.
    binary_path = str(
        importlib.resources.files("google.antigravity").joinpath(
            "bin/localharness"
        )
    )
    if os.path.exists(binary_path):
      return binary_path
  except (ImportError, AttributeError, KeyError):
    pass

  # 4. Fallback: Check if it's in the system PATH
  if path := shutil.which("localharness"):
    return path

  raise RuntimeError(
      "Could not find default localharness binary. "
      "Please specify binary_path explicitly or ensure it is in your PATH."
  )


class LocalConnectionStrategy(connection.ConnectionStrategy):
  """Strategy for establishing a LocalConnection."""

  def __init__(
      self,
      *,
      tool_runner: t_runner.ToolRunner | None = None,
      hook_runner: h_runner.HookRunner | None = None,
      gemini_config: str | types.GeminiConfig | None = None,
      skills_paths: list[str] | None = None,
      system_instructions: str | types.SystemInstructions | None = None,
      capabilities_config: types.CapabilitiesConfig | None = None,
      conversation_id: str | None = None,
      save_dir: str | None = None,
      workspaces: list[str] | None = None,
  ):
    self._binary_path = _get_default_binary_path()
    self._tool_runner = tool_runner
    self._hook_runner = hook_runner

    # Normalize str shorthand to GeminiConfig model.
    if isinstance(gemini_config, str):
      self._gemini_config = types.GeminiConfig(
          models=types.ModelConfig(default=types.ModelEntry(name=gemini_config))
      )
    else:
      self._gemini_config = gemini_config
    self._skills_paths = skills_paths

    # Normalize str shorthand to SystemInstructions model.
    if isinstance(system_instructions, str):
      self._system_instructions = types.TemplatedSystemInstructions(
          sections=[types.SystemInstructionSection(content=system_instructions)]
      )
    else:
      self._system_instructions = system_instructions
    self._capabilities_config = (
        capabilities_config or types.CapabilitiesConfig()
    )
    self._conversation_id = conversation_id
    self._save_dir = save_dir
    self._workspaces = workspaces or []

  def _build_harness_config(self) -> localharness_pb2.HarnessConfig:
    """Translates Pydantic config objects into a HarnessConfig proto."""
    tool_protos = []
    if self._tool_runner:
      tool_protos = [
          callable_to_tool_proto(fn, tool_runner=self._tool_runner)
          for fn in self._tool_runner.tools.values()
      ]

    system_instructions_proto = None
    if self._system_instructions:
      system_instructions_proto = localharness_pb2.SystemInstructions()
      if isinstance(self._system_instructions, types.CustomSystemInstructions):
        system_instructions_proto.custom.CopyFrom(
            localharness_pb2.CustomSystemInstructions(
                part=[
                    localharness_pb2.CustomSystemInstructions.Part(
                        text=self._system_instructions.text
                    )
                ]
            )
        )
      elif isinstance(
          self._system_instructions, types.TemplatedSystemInstructions
      ):
        appended = localharness_pb2.AppendedSystemInstructions()
        if self._system_instructions.identity:
          appended.custom_identity = self._system_instructions.identity
        for sec in self._system_instructions.sections:
          appended.appended_sections.add(title=sec.title, content=sec.content)
        system_instructions_proto.appended.CopyFrom(appended)

    gemini_config_proto = None
    if self._gemini_config:
      gemini_config_proto = localharness_pb2.GeminiConfig(
          model_name=self._gemini_config.models.default.name,
      )
      # Use per-model API key if set, otherwise fall back to shared key.
      effective_api_key = (
          self._gemini_config.models.default.api_key
          or self._gemini_config.api_key
      )
      if effective_api_key is not None:
        gemini_config_proto.api_key = effective_api_key
      thinking_level = (
          self._gemini_config.models.default.generation.thinking_level
      )
      if thinking_level is not None:
        gemini_config_proto.thinking_level = thinking_level.value

    workspace_protos = [
        localharness_pb2.Workspace(
            filesystem_workspace=localharness_pb2.FilesystemWorkspace(
                directory=p
            )
        )
        for p in self._workspaces
    ]

    cfg = self._capabilities_config

    # Determine which BuiltinTools are active.
    all_tools = set(types.BuiltinTools)
    if cfg.enabled_tools is not None:
      active_tools = set(cfg.enabled_tools)
    elif cfg.disabled_tools is not None:
      active_tools = all_tools - set(cfg.disabled_tools)
    else:
      active_tools = all_tools

    harness_side_tools = localharness_pb2.HarnessSideTools(
        subagents=localharness_pb2.SubagentsConfig(
            enabled=cfg.enable_subagents
        ),
        find=localharness_pb2.FindToolConfig(
            enabled=types.BuiltinTools.FIND_FILE in active_tools
        ),
        user_questions=localharness_pb2.UserQuestionsConfig(
            enabled=types.BuiltinTools.ASK_QUESTION in active_tools
        ),
        run_command=localharness_pb2.RunCommandToolConfig(
            enabled=types.BuiltinTools.RUN_COMMAND in active_tools
        ),
        file_edit=localharness_pb2.FileEditToolConfig(
            enabled=types.BuiltinTools.EDIT_FILE in active_tools
        ),
        view_file=localharness_pb2.ViewFileToolConfig(
            enabled=types.BuiltinTools.VIEW_FILE in active_tools
        ),
        write_to_file=localharness_pb2.WriteToFileToolConfig(
            enabled=types.BuiltinTools.CREATE_FILE in active_tools
        ),
        grep_search=localharness_pb2.GrepSearchToolConfig(
            enabled=types.BuiltinTools.SEARCH_DIR in active_tools
        ),
        list_dir=localharness_pb2.ListDirToolConfig(
            enabled=types.BuiltinTools.LIST_DIR in active_tools
        ),
        generate_image=localharness_pb2.GenerateImageToolConfig(
            enabled=types.BuiltinTools.GENERATE_IMAGE in active_tools,
            model_name=cfg.image_model,
        ),
    )

    harness_config = localharness_pb2.HarnessConfig(
        tools=tool_protos,
        system_instructions=system_instructions_proto,
        cascade_id=self._conversation_id or "",
        gemini_config=gemini_config_proto,
        workspaces=workspace_protos,
        skills_paths=self._skills_paths or [],
        harness_side_tools=harness_side_tools,
        # 0 tells the harness to use its default (30000 tokens).
        compaction_threshold=cfg.compaction_threshold or 0,
        finish_tool_schema_json=cfg.finish_tool_schema_json or "",
    )

    return harness_config

  def connect(self) -> connection.Connection:
    """Returns the established Connection."""
    if not hasattr(self, "_connection") or self._connection is None:
      raise RuntimeError(
          "Connection not established. Use as a context manager."
      )
    return self._connection

  async def __aenter__(self) -> None:
    """Starts the backend."""
    # Fail fast if no API key is available. The localharness binary requires
    # a Gemini API key to call the Gemini API; without one it silently returns
    # empty responses.
    api_key = (
        self._gemini_config.api_key if self._gemini_config else None
    ) or os.environ.get("GEMINI_API_KEY")
    if not api_key:
      raise types.AntigravityValidationError(
          "A Gemini API key is required. Set it via"
          " GeminiConfig(api_key=...) or the GEMINI_API_KEY environment"
          " variable."
      )

    harness_config = self._build_harness_config()
    input_config = localharness_pb2.InputConfig(
        storage_directory=self._save_dir or "",
    )

    process = subprocess.Popen(
        [self._binary_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    serialized = input_config.SerializeToString()
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    # Note for humans: Pack length as 4-byte uint (little-endian)
    process.stdin.write(struct.pack("<I", len(serialized)) + serialized)
    process.stdin.flush()
    raw_len = process.stdout.read(4)
    if not raw_len:
      stderr_output = process.stderr.read().decode("utf-8")
      raise RuntimeError(
          f"Failed to read length from stdout. Stderr: {stderr_output}"
      )
    length = struct.unpack("<I", raw_len)[0]
    output_config = localharness_pb2.OutputConfig()
    output_config.ParseFromString(process.stdout.read(length))
    ws_url = f"ws://localhost:{output_config.port}/"

    # Retry the WebSocket connection with backoff. The harness process may
    # need a moment to start listening after writing its OutputConfig.
    max_retries = 5
    ws = None
    for attempt in range(max_retries):
      try:
        ws = await websockets.connect(
            ws_url,
            additional_headers={"x-goog-api-key": output_config.api_key},
        )
        break
      except (OSError, websockets.WebSocketException) as e:
        if attempt == max_retries - 1:
          process.kill()
          stderr_output = process.stderr.read().decode("utf-8")
          raise RuntimeError(
              f"Failed to connect to WebSocket at {ws_url} after"
              f" {max_retries} attempts. Stderr: {stderr_output}"
          ) from e
        await asyncio.sleep(0.1 * (2 ** attempt))

    assert ws is not None
    try:
      init_event = localharness_pb2.InitializeConversationEvent(
          config=harness_config
      )
      await ws.send(json_format.MessageToJson(init_event))
    except Exception as e:
      process.kill()
      stderr_output = process.stderr.read().decode("utf-8")
      raise RuntimeError(
          f"Failed to initialize conversation at {ws_url}."
          f" Stderr: {stderr_output}"
      ) from e
    self._connection = LocalConnection(
        process=process,
        ws=ws,
        tool_runner=self._tool_runner,
        hook_runner=self._hook_runner,
    )
    self._connection._start_stderr_reader(process.stderr)

  async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
    """Tears down the backend and releases all resources."""
    if hasattr(self, "_connection") and self._connection:
      await self._connection.disconnect()
      self._connection = None


def _parse_usage_metadata(
    usage_metadata: localharness_pb2.UsageMetadata,
) -> types.UsageMetadata:
  """Extracts UsageMetadata from proto message."""
  um = types.UsageMetadata()
  if usage_metadata.HasField("prompt_token_count"):
    um.prompt_token_count = usage_metadata.prompt_token_count
  if usage_metadata.HasField("cached_content_token_count"):
    um.cached_content_token_count = usage_metadata.cached_content_token_count
  if usage_metadata.HasField("candidates_token_count"):
    um.candidates_token_count = usage_metadata.candidates_token_count
  if usage_metadata.HasField("thoughts_token_count"):
    um.thoughts_token_count = usage_metadata.thoughts_token_count
  if usage_metadata.HasField("total_token_count"):
    um.total_token_count = usage_metadata.total_token_count
  return um
