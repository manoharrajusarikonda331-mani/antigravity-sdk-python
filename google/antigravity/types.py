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

"""Type definitions for Google Antigravity SDK.

These are the canonical SDK boundary types. All public SDK interfaces use these
types. They are pure Python Pydantic V2 models with no proto dependencies.
"""

from __future__ import annotations

import enum
import itertools
import mimetypes
import pathlib
from typing import Annotated, Any, AsyncIterator, Callable, List, Literal, Optional, Union

import pydantic

# =============================================================================
# Config types
# =============================================================================

DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_IMAGE_GENERATION_MODEL = "gemini-3.1-flash-image-preview"


class ThinkingLevel(str, enum.Enum):
  """Thinking level for Gemini models that support extended thinking.

  Controls the amount of reasoning the model performs before responding.
  See https://ai.google.dev/gemini-api/docs/thinking#thinking-levels for
  details.

  Attributes:
    MINIMAL: Minimal thinking.
    LOW: Low thinking.
    MEDIUM: Medium thinking.
    HIGH: High thinking.
  """

  MINIMAL = "minimal"
  LOW = "low"
  MEDIUM = "medium"
  HIGH = "high"


class GenerationConfig(pydantic.BaseModel):
  """Generation parameters for a model.

  Attributes:
    thinking_level: Thinking level for models that support extended thinking.
      When None, the model's default level is used.
  """

  thinking_level: ThinkingLevel | None = None


def _coerce_model_entry(v: "ModelEntry | str") -> "ModelEntry":
  """Coerce a bare model name string into a ModelEntry."""
  if isinstance(v, str):
    return ModelEntry(name=v)
  return v


class ModelEntry(pydantic.BaseModel):
  """A model with optional auth and generation overrides.

  Attributes:
    name: Model name (e.g. 'gemini-3.1-pro-preview').
    api_key: Per-model API key override. Falls back to GeminiConfig.api_key.
    generation: Generation parameters for this model.
  """

  name: str
  api_key: str | None = None
  generation: GenerationConfig = pydantic.Field(
      default_factory=GenerationConfig
  )


class ModelConfig(pydantic.BaseModel):
  """Model selection for each capability.

  Slots accept a bare model name string (coerced to ModelEntry) or
  a ModelEntry for per-model overrides. After validation, all slots
  are always ModelEntry.

  Attributes:
    default: The primary reasoning model.
    image_generation: The model used for image generation.
  """

  default: Annotated[
      ModelEntry, pydantic.BeforeValidator(_coerce_model_entry)
  ] = pydantic.Field(default_factory=lambda: ModelEntry(name=DEFAULT_MODEL))
  image_generation: Annotated[
      ModelEntry, pydantic.BeforeValidator(_coerce_model_entry)
  ] = pydantic.Field(
      default_factory=lambda: ModelEntry(name=DEFAULT_IMAGE_GENERATION_MODEL)
  )


class GeminiConfig(pydantic.BaseModel):
  """Configuration for the Gemini model backend.

  Attributes:
    api_key: Shared API key for all models. Falls back to $GEMINI_API_KEY if not
      set. Individual ModelEntry instances can override this.
    models: Per-modality model selection and configuration.
  """

  api_key: str | None = None
  models: ModelConfig = pydantic.Field(default_factory=ModelConfig)


class SystemInstructionSection(pydantic.BaseModel):
  """A named section to append to the system instructions."""

  content: str
  title: str = "user_system_instructions"


class CustomSystemInstructions(pydantic.BaseModel):
  """Use this to completely replace the system instructions.

  WARNING: For advanced usage only. This replaces ALL default instructions.
  If you use this, you are responsible for providing all necessary instructions
  yourself, for example:
  - **Core Mandates**: Security and safety rules (e.g., credential protection).
  - **Engineering Standards**: Coding style, testing, and linting rules.
  - **Operational Guidelines**: Tone, brevity, and tool usage protocols.

  Most users should use TemplatedSystemInstructions instead.
  """

  text: str


class TemplatedSystemInstructions(pydantic.BaseModel):
  """Use this to override the agent's identity and append sections to the default system instructions.

  See `examples/agents/templated_system_instructions.py`
  for a full example with identity and sections.
  """

  identity: Optional[str] = None
  sections: List[SystemInstructionSection] = pydantic.Field(
      default_factory=list
  )


# Union type representing the two ways to configure system instructions.
# - CustomSystemInstructions: Full replacement (Advanced usage).
# - TemplatedSystemInstructions: Append to defaults (Recommended).
SystemInstructions = Union[
    CustomSystemInstructions, TemplatedSystemInstructions
]


class BuiltinTools(str, enum.Enum):
  """Identifiers for common connection-provided builtin tools.

  Attributes:
    LIST_DIR: List directory contents.
    SEARCH_DIR: Search within directories (grep).
    FIND_FILE: Find files by name within a directory.
    VIEW_FILE: View file contents.
    CREATE_FILE: Create a new file.
    EDIT_FILE: Edit an existing file.
    RUN_COMMAND: Execute a shell command.
    ASK_QUESTION: Ask the user a clarifying question.
    START_SUBAGENT: Invoke a subagent.
    GENERATE_IMAGE: Generate or edit images.
    FINISH: Finish the conversation and return structured output.
  """

  LIST_DIR = "list_directory"
  SEARCH_DIR = "search_directory"
  FIND_FILE = "find_file"
  VIEW_FILE = "view_file"
  CREATE_FILE = "create_file"
  EDIT_FILE = "edit_file"
  RUN_COMMAND = "run_command"
  ASK_QUESTION = "ask_question"
  START_SUBAGENT = "start_subagent"
  GENERATE_IMAGE = "generate_image"
  FINISH = "finish"

  @classmethod
  def read_only(cls) -> list["BuiltinTools"]:
    """Returns tools that only read state (no writes, deletes, or commands).

    Returns:
        A list of read-only BuiltinTools.
    """
    return [
        cls.LIST_DIR,
        cls.SEARCH_DIR,
        cls.FIND_FILE,
        cls.VIEW_FILE,
        cls.FINISH,
    ]

  @classmethod
  def nondestructive(cls) -> list["BuiltinTools"]:
    """Returns tools that cannot delete content.

    Returns:
        A list of non-destructive BuiltinTools.
    """
    return [
        cls.LIST_DIR,
        cls.SEARCH_DIR,
        cls.FIND_FILE,
        cls.VIEW_FILE,
        cls.CREATE_FILE,
        cls.EDIT_FILE,
        cls.ASK_QUESTION,
        cls.START_SUBAGENT,
        cls.GENERATE_IMAGE,
        cls.FINISH,
    ]

  @classmethod
  def all(cls) -> list["BuiltinTools"]:
    """Returns all builtin tools.

    Returns:
        A list of all BuiltinTools.
    """
    return list(cls)

  @classmethod
  def none(cls) -> list["BuiltinTools"]:
    """Returns an empty tool list (no builtin tools).

    Returns:
        An empty list of BuiltinTools.
    """
    return []


class CapabilitiesConfig(pydantic.BaseModel):
  """General agent capability configuration.

  Attributes:
    enable_subagents: Whether the agent can spawn and delegate to sub-agents.
    enabled_tools: Explicit allowlist of builtin tools to enable. Mutually
      exclusive with disabled_tools. When None, the harness defaults are used
      (all tools enabled).
    disabled_tools: Explicit denylist of builtin tools to disable. Mutually
      exclusive with enabled_tools. When None, the harness defaults are used
      (all tools enabled).
    compaction_threshold: Token count after which the context window may be
      compacted. When None, the backend's default is used.
    image_model: The model to use for image generation. Defaults to
      'gemini-3.1-flash-image-preview'.
    finish_tool_schema_json: Optional JSON schema string for the finish tool.
  """

  enable_subagents: bool = True
  enabled_tools: list[BuiltinTools] | None = None
  disabled_tools: list[BuiltinTools] | None = None
  compaction_threshold: int | None = None
  image_model: str = "gemini-3.1-flash-image-preview"
  finish_tool_schema_json: str | None = None

  @pydantic.model_validator(mode="after")
  def _check_mutually_exclusive(self) -> "CapabilitiesConfig":
    if self.enabled_tools is not None and self.disabled_tools is not None:
      raise ValueError(
          "enabled_tools and disabled_tools should be mutually exclusive."
      )
    return self


class McpStdioServer(pydantic.BaseModel):
  """Configuration for an MCP server connected via stdio.

  Attributes:
    command: The command to run to start the server.
    type: The type of connection, always "stdio".
    args: Arguments to pass to the command.
  """

  command: str
  type: Literal["stdio"] = "stdio"
  args: list[str] = pydantic.Field(default_factory=list)


class McpSseServer(pydantic.BaseModel):
  """Configuration for an MCP server connected via SSE.

  Attributes:
    url: The URL of the SSE endpoint.
    type: The type of connection, always "sse".
    headers: Optional headers to send with the connection request.
  """

  url: str
  type: Literal["sse"] = "sse"
  headers: dict[str, str] | None = None


class McpStreamableHttpServer(pydantic.BaseModel):
  """Configuration for an MCP server connected via Streamable HTTP.

  Attributes:
    url: The URL of the HTTP endpoint.
    type: The type of connection, always "http".
    headers: Optional headers to send with the connection request.
    timeout: Connection timeout in seconds.
    sse_read_timeout: SSE read timeout in seconds.
    terminate_on_close: Whether to terminate the connection on close.
  """

  url: str
  type: Literal["http"] = "http"
  headers: dict[str, str] | None = None
  timeout: float = 30.0
  sse_read_timeout: float = 300.0
  terminate_on_close: bool = True


McpServerConfig = McpStdioServer | McpSseServer | McpStreamableHttpServer


# =============================================================================
# Tool types
# =============================================================================


class ToolCall(pydantic.BaseModel):
  """A tool call to inject into the conversation.

  Attributes:
    id: Optional unique identifier for the call, often assigned by the backend.
    name: Tool identifier. Use a BuiltinTools member for Connection-provided
      tools, or an arbitrary string for custom host-side tools.
    args: Keyword arguments for the tool, as a JSON-serializable dict.
  """

  name: BuiltinTools | str
  args: dict[str, Any] = pydantic.Field(default_factory=dict)
  id: str | None = None


class ToolResult(pydantic.BaseModel):
  """Result of a single tool execution.

  Attributes:
    id: Optional identifier correlating this result with a ToolCall.id.
    name: The name of the tool that was executed. A BuiltinTools member for
      Connection-provided tools, or a string for custom host-side tools.
    result: The tool's return value. Can be any JSON-serializable value.
    error: An error message if execution failed, or None on success.
    exception: The original exception if execution failed. Not serialized.
  """

  model_config = pydantic.ConfigDict(
      extra="ignore", arbitrary_types_allowed=True
  )

  name: BuiltinTools | str
  id: str | None = None
  result: Any = None
  error: str | None = None
  exception: Exception | None = pydantic.Field(default=None, exclude=True)


PythonTool = Callable[..., Any]


# =============================================================================
# Step types
# =============================================================================


class UsageMetadata(pydantic.BaseModel):
  """Token usage metadata from the model API.

  Fields are None when the data is not available (e.g. the step did not
  involve a model call). A value of 0 means the model explicitly reported
  zero tokens for that category.

  Attributes:
    prompt_token_count: Number of tokens in the prompt.
    cached_content_token_count: Number of tokens from cached content. These are
      a subset of prompt tokens.
    candidates_token_count: Number of tokens in the generated candidates
      (excluding thinking).
    thoughts_token_count: Number of tokens used for thinking/reasoning.
    total_token_count: Sum of prompt + candidates + thinking tokens.
  """

  # Input tokens.
  prompt_token_count: int | None = None
  cached_content_token_count: int | None = None

  # Output tokens.
  candidates_token_count: int | None = None
  thoughts_token_count: int | None = None

  # Total tokens (prompt + candidates + thoughts).
  total_token_count: int | None = None


class StepType(str, enum.Enum):
  """High-level type of a step."""

  MODEL_RESPONSE = "MODEL_RESPONSE"
  TOOL_CALL = "TOOL_CALL"
  SYSTEM_MESSAGE = "SYSTEM_MESSAGE"
  COMPACTION = "COMPACTION"
  FINISH = "FINISH"
  UNKNOWN = "UNKNOWN"


class StepSource(str, enum.Enum):
  """Source of a step."""

  SYSTEM = "SYSTEM"
  USER = "USER"
  MODEL = "MODEL"
  UNKNOWN = "UNKNOWN"


class StepTarget(str, enum.Enum):
  """Target of a step interaction."""

  USER = "TARGET_USER"
  ENVIRONMENT = "TARGET_ENVIRONMENT"
  UNSPECIFIED = "TARGET_UNSPECIFIED"
  UNKNOWN = "UNKNOWN"


class StepStatus(str, enum.Enum):
  """Status of a step."""

  ACTIVE = "ACTIVE"
  DONE = "DONE"
  WAITING_FOR_USER = "WAITING_FOR_USER"
  ERROR = "ERROR"
  CANCELED = "CANCELED"
  UNKNOWN = "UNKNOWN"


class Step(pydantic.BaseModel):
  """Structure representing one action in the agent trajectory.

  Attributes:
    id: Unique string identifier for the step.
    step_index: Integer index of the step in the trajectory.
    type: The high-level type of the step.
    source: The source that generated the step.
    target: The target interacting with this step.
    status: The status of the step.
    content: The output of the step.
    thinking: Model reasoning/thinking for planner responses.
    content_delta: Text added since the last update for this step.
    thinking_delta: Thinking added since the last update for this step.
    tool_calls: List of tool calls associated with the step.
    error: Short error message if the step failed or empty string.
    is_complete_response: True if this step is a completed model response
      directed at the user, as distinct from a partial streaming chunk. Multiple
      steps per turn may have this flag set; consumers that want only the last
      response should iterate fully.
    structured_output: The structured output extracted from the finish step.
    usage_metadata: Token usage for this specific step's model invocation, or
      None if this step did not involve a model call.
  """

  id: str = ""
  step_index: int = 0
  type: StepType = StepType.UNKNOWN
  source: StepSource = StepSource.UNKNOWN
  target: StepTarget = StepTarget.UNKNOWN
  status: StepStatus = StepStatus.UNKNOWN
  content: str = ""
  content_delta: str = ""
  thinking: str = ""
  thinking_delta: str = ""
  tool_calls: list[ToolCall] = pydantic.Field(default_factory=list)
  error: str = ""
  is_complete_response: bool | None = None
  structured_output: Any | None = None
  usage_metadata: UsageMetadata | None = None

  model_config = pydantic.ConfigDict(extra="allow")


# =============================================================================
# Hook types
# =============================================================================
class HookResult(pydantic.BaseModel):
  """Result of a decision hook execution.

  Attributes:
    allow: Whether execution should proceed.
    message: Optional explanation or response message.
  """

  model_config = pydantic.ConfigDict(extra="ignore")

  allow: bool = True
  message: str = ""


class QuestionResponse(pydantic.BaseModel):
  """Individual response for an AskQuestion entry.

  Attributes:
    selected_option_ids: List of option IDs selected.
    freeform_response: Freeform text response.
    skipped: If true, the question is marked as skipped.
  """

  model_config = pydantic.ConfigDict(extra="ignore")

  selected_option_ids: list[str] | None = None
  freeform_response: str = ""
  skipped: bool = False


class QuestionHookResult(pydantic.BaseModel):
  """Result of an interaction containing a list of responses.

  Attributes:
    responses: List of QuestionResponse objects.
    cancelled: If true, the interaction was cancelled.
  """

  model_config = pydantic.ConfigDict(extra="ignore")

  responses: list[QuestionResponse]
  cancelled: bool = False


class AskQuestionOption(pydantic.BaseModel):
  """Option for an AskQuestion entry."""

  model_config = pydantic.ConfigDict(frozen=True, extra="ignore")

  id: str
  text: str


class AskQuestionEntry(pydantic.BaseModel):
  """A single question with predefined options."""

  model_config = pydantic.ConfigDict(frozen=True, extra="ignore")

  question: str
  options: list[AskQuestionOption]
  is_multi_select: bool = False


class AskQuestionInteractionSpec(pydantic.BaseModel):
  """Interaction spec for ask_question dialog."""

  model_config = pydantic.ConfigDict(frozen=True, extra="ignore")

  questions: list[AskQuestionEntry]


# =============================================================================
# Error types
# =============================================================================


class AntigravityConnectionError(Exception):
  """Base class for connection errors in the Google Antigravity SDK.

  Raised when a connection to an agent backend cannot be established or
  encounters a fatal protocol-level error.
  """


class AntigravityValidationError(Exception):
  """Wraps Pydantic ValidationError at the SDK boundary.

  SDK consumers should catch this instead of pydantic.ValidationError directly.
  This decouples the public API from the Pydantic implementation detail.

  Attributes:
    message: Human-readable error description.
    errors: The structured error list from Pydantic, if available.
  """

  def __init__(
      self,
      message: str,
      errors: list[dict[str, Any]] | None = None,
  ):
    super().__init__(message)
    self.message = message
    self.errors = errors or []

  @classmethod
  def from_pydantic(
      cls, exc: pydantic.ValidationError
  ) -> "AntigravityValidationError":
    """Constructs from a Pydantic ValidationError.

    Args:
      exc: The original Pydantic ValidationError.

    Returns:
      An AntigravityValidationError wrapping the Pydantic error.
    """
    return cls(message=str(exc), errors=exc.errors())


class TriggerDelivery(str, enum.Enum):
  """Controls how trigger messages are delivered to the agent."""

  SEND_IMMEDIATELY = "send_immediately"  # Send immediately (non-blocking).
  WAIT_IDLE = "wait_idle"  # Wait until agent is idle before sending.
  # TODO: INTERRUPT — cancel current turn, then send. Deferred due to
  # safety implications for in-flight tool calls (requires Connection.cancel()).


class FileChangeKind(str, enum.Enum):
  """Kind of filesystem change detected by a file-watching trigger."""

  ADDED = "added"
  MODIFIED = "modified"
  DELETED = "deleted"


class FileChange(pydantic.BaseModel):
  """A single filesystem change detected by a file-watching trigger.

  Attributes:
    kind: The type of change (added, modified, deleted).
    path: Absolute path to the changed file.
  """

  model_config = pydantic.ConfigDict(frozen=True)

  kind: FileChangeKind
  path: str


# =============================================================================
# Response types
# =============================================================================


class StreamChunk(pydantic.BaseModel):
  """Base class for all real-time semantic chunks yielded during agent.chat() streaming."""

  step_index: int
  model_config = pydantic.ConfigDict(frozen=True)


class Thought(StreamChunk):
  """A delta chunk representing a piece of the model's internal reasoning/thinking."""

  text: str  # Incremental thought string delta
  signature: bytes | None = None


class Text(StreamChunk):
  """A delta chunk representing a piece of the model's text output."""

  text: str  # Incremental response string delta


class ChatResponse:
  """The turn response from Agent.chat().

  An async stream of semantic chunks with lazy buffering. This class provides
  both zero-boilerplate text token streaming and advanced sugared event streams.
  """

  def __init__(
      self,
      chunk_stream: AsyncIterator[StreamChunk | ToolCall | ToolResult],
      conversation: Any,
  ):
    self._chunk_stream = chunk_stream
    self._conversation = conversation
    self._buffered_chunks: list[StreamChunk | ToolCall | ToolResult] = []
    self._is_done = False

  @property
  def chunks(self) -> AsyncIterator[StreamChunk | ToolCall | ToolResult]:
    """The rich, unfiltered semantic chunk stream for more advanced use cases."""

    async def _chunks_gen():
      # Cache Walk (yields already cached chunks for re-iteration)
      for chunk in self._buffered_chunks:
        yield chunk

      # Network Walk (pulls from the live stream and caches them)
      if not self._is_done:
        try:
          async for chunk in self._chunk_stream:
            self._buffered_chunks.append(chunk)
            yield chunk
          self._is_done = True
        except Exception:
          self._is_done = True
          raise

    return _chunks_gen()

  async def __aiter__(self) -> AsyncIterator[str]:
    """Streams conversational text token deltas directly as raw strings."""
    async for chunk in self.chunks:
      chunk_obj: Any = chunk
      if isinstance(chunk_obj, Text):
        yield chunk_obj.text

  @property
  def thoughts(self) -> AsyncIterator[str]:
    """The internal model reasoning/thinking token deltas as raw strings."""

    async def _thoughts_gen():
      async for chunk in self.chunks:
        chunk_obj: Any = chunk
        if isinstance(chunk_obj, Thought):
          yield chunk_obj.text

    return _thoughts_gen()

  @property
  def tool_calls(self) -> AsyncIterator[ToolCall]:
    """The strongly-typed ToolCall objects in real-time as they are dispatched."""

    async def _tool_calls_gen():
      async for chunk in self.chunks:
        chunk_obj: Any = chunk
        if isinstance(chunk_obj, ToolCall):
          yield chunk_obj

    return _tool_calls_gen()

  async def resolve(self) -> list[StreamChunk | ToolCall | ToolResult]:
    """Drains the underlying stream completely and returns all chunks as a flat list.

    Returns:
        A list of all chunks yielded during the turn.
    """
    return [chunk async for chunk in self.chunks]

  async def text(self) -> str:
    """Drains the stream and returns the fully aggregated conversational response text.

    Returns:
        The complete response text as a single string.
    """
    chunks = await self.resolve()
    return "".join(chunk.text for chunk in chunks if isinstance(chunk, Text))

  async def structured_output(self) -> Any | None:
    """Drains the stream and extracts the parsed structured output payload, if one exists.

    Returns:
        The parsed structured output if available, otherwise None.
    """
    if not self._is_done:
      await self.resolve()
    return self._conversation.get_last_structured_output()

  @property
  def usage_metadata(self) -> UsageMetadata | None:
    """Accumulated token usage across all model invocations in this turn."""
    return self._conversation.last_turn_usage


# =============================================================================
# Input Content Primitives
# =============================================================================

SUPPORTED_IMAGE_MIMES = frozenset({
    "image/bmp",
    "image/jpeg",
    "image/png",
    "image/webp",
})

SUPPORTED_DOCUMENT_MIMES = frozenset({
    "application/pdf",
    "application/json",
    "text/css",
    "text/csv",
    "text/html",
    "text/javascript",
    "text/plain",
    "text/rtf",
    "text/xml",
})

SUPPORTED_AUDIO_MIMES = frozenset({
    "audio/wav",
    "audio/mp3",
    "audio/aac",
    "audio/ogg",
    "audio/flac",
    "audio/opus",
    "audio/mpeg",
    "audio/m4a",
    "audio/l16",
})

SUPPORTED_VIDEO_MIMES = frozenset({
    "video/3gpp",
    "video/avi",
    "video/mp4",
    "video/mpeg",
    "video/mpg",
    "video/quicktime",
    "video/webm",
    "video/wmv",
    "video/x-flv",
})


def _read_file_safely(path: str | pathlib.Path) -> bytes:
  """Robustly loads local file bytes with comprehensive error wrapping.

  Args:
      path: The file path to read.

  Returns:
      The file contents as bytes.

  Raises:
      FileNotFoundError: If the file does not exist.
      IsADirectoryError: If the path is a directory.
      PermissionError: If the file is not readable.
      OSError: For other filesystem errors.
  """
  file_path = pathlib.Path(path)
  try:
    return file_path.read_bytes()
  except FileNotFoundError as exc:
    raise FileNotFoundError(f"File not found at path: '{file_path}'") from exc
  except IsADirectoryError as exc:
    raise IsADirectoryError(
        f"Path is a directory, not a file: '{file_path}'"
    ) from exc
  except PermissionError as exc:
    raise PermissionError(
        f"Permission denied when reading path: '{file_path}'"
    ) from exc
  except OSError as exc:
    raise OSError(f"Failed to read file at path '{file_path}': {exc}") from exc


class _BaseMedia(pydantic.BaseModel):
  """Base class for all rich multimedia content attachment primitives."""

  data: bytes
  mime_type: str
  description: str | None = None

  @classmethod
  def from_file(
      cls, path: str | pathlib.Path, description: str | None = None
  ) -> _BaseMedia:
    """Instantiates a media content primitive from a local file path.

    Args:
        path: Local file path to read.
        description: Optional text description of the media.

    Returns:
        The instantiated media object.
    """
    file_path = pathlib.Path(path)
    data = _read_file_safely(file_path)
    mime_guess, _ = mimetypes.guess_type(file_path)
    return cls(
        data=data,
        mime_type=mime_guess or "",
        description=description,
    )

  model_config = pydantic.ConfigDict(frozen=True)


class Image(_BaseMedia):
  """Image content attachment primitive."""

  @pydantic.field_validator("mime_type")
  @classmethod
  def validate_mime_type(cls, v: str) -> str:
    """Validates that the MIME type is supported for Image content."""
    if v not in SUPPORTED_IMAGE_MIMES:
      raise ValueError(f"Unsupported Image MIME type: '{v}'")
    return v


class Document(_BaseMedia):
  """Document content attachment primitive."""

  @pydantic.field_validator("mime_type")
  @classmethod
  def validate_mime_type(cls, v: str) -> str:
    """Validates that the MIME type is supported for Document content."""
    if v not in SUPPORTED_DOCUMENT_MIMES:
      raise ValueError(f"Unsupported Document MIME type: '{v}'")
    return v


class Audio(_BaseMedia):
  """Audio content attachment primitive."""

  @pydantic.field_validator("mime_type")
  @classmethod
  def validate_mime_type(cls, v: str) -> str:
    """Validates that the MIME type is supported for Audio content."""
    if v not in SUPPORTED_AUDIO_MIMES:
      raise ValueError(f"Unsupported Audio MIME type: '{v}'")
    return v


class Video(_BaseMedia):
  """Video content attachment primitive."""

  @pydantic.field_validator("mime_type")
  @classmethod
  def validate_mime_type(cls, v: str) -> str:
    """Validates that the MIME type is supported for Video content."""
    if v not in SUPPORTED_VIDEO_MIMES:
      raise ValueError(f"Unsupported Video MIME type: '{v}'")
    return v


ContentPrimitive = str | Image | Document | Audio | Video
Content = ContentPrimitive | list[ContentPrimitive]


def from_file(
    path: str | pathlib.Path, description: str | None = None
) -> Image | Document | Audio | Video:
  """Automatically resolves a local file path into the correct semantic Content primitive.

  Args:
      path: Local file path to read.
      description: Optional text description of the media.

  Returns:
      A specialized media object (Image, Document, Audio, or Video) based
      on the file's MIME type.

  Raises:
      ValueError: If the MIME type cannot be inferred or is unsupported.
  """
  file_path = pathlib.Path(path)
  data = _read_file_safely(file_path)

  mime_guess, _ = mimetypes.guess_type(file_path)
  if not mime_guess:
    raise ValueError(
        f"Could not infer a valid MIME type for extension: '{file_path.suffix}'"
    )

  if mime_guess in SUPPORTED_IMAGE_MIMES:
    return Image(
        data=data,
        mime_type=mime_guess,
        description=description,
    )
  elif mime_guess in SUPPORTED_DOCUMENT_MIMES:
    return Document(
        data=data,
        mime_type=mime_guess,
        description=description,
    )
  elif mime_guess in SUPPORTED_AUDIO_MIMES:
    return Audio(
        data=data,
        mime_type=mime_guess,
        description=description,
    )
  elif mime_guess in SUPPORTED_VIDEO_MIMES:
    return Video(
        data=data,
        mime_type=mime_guess,
        description=description,
    )
  else:
    all_supported = list(
        itertools.chain(
            SUPPORTED_IMAGE_MIMES,
            SUPPORTED_DOCUMENT_MIMES,
            SUPPORTED_AUDIO_MIMES,
            SUPPORTED_VIDEO_MIMES,
        )
    )
    raise ValueError(
        f"Unsupported MIME type: '{mime_guess}'. "
        f"Supported file formats in the SDK are: {sorted(all_supported)}"
    )
