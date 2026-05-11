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

"""In-process tool runner for the Google Antigravity SDK.

Tools are Python callables that run directly in the SDK process. The
ToolRunner is a registry and executor — it holds references to tool
functions and invokes them by name when requested.

HOW tool calls reach the runner (callback server, direct invocation,
RPC bridge) is a connection strategy concern, not a tool runner concern.

ToolContext injection is handled here: at registration time the runner
inspects each tool's signature for a ``ToolContext``-typed parameter
and caches the result. At execution time, the context is injected
automatically. Schema generation (``get_public_callable``) strips
injectable parameters so the model never sees them.
"""

import asyncio
import collections.abc
import functools
import inspect
from typing import Any, Callable

from google.antigravity import types
from google.antigravity.tools import tool_context as tool_context_module


def _find_context_param(fn: Callable[..., Any]) -> str | None:
  """Returns the name of the ToolContext-typed parameter, if any.

  Inspects the function signature for a parameter whose annotation
  resolves to ``ToolContext``. Supports both live type annotations
  and stringified annotations (``from __future__ import annotations``).

  Args:
    fn: The callable to inspect. If it's a ``ToolWithSchema``, the inner ``.fn``
      is used.

  Returns:
    The parameter name, or None if no ToolContext parameter is found.
  """
  target = fn.fn if isinstance(fn, ToolWithSchema) else fn
  try:
    sig = inspect.signature(target)
  except (ValueError, TypeError):
    return None

  for name, param in sig.parameters.items():
    ann = param.annotation
    if ann is inspect.Parameter.empty:
      continue
    # Direct type check (most common case).
    if ann is tool_context_module.ToolContext:
      return name
    # Handle stringified annotations and Optional/Union forms.
    ann_str = str(ann)
    if "ToolContext" in ann_str:
      return name
  return None


def _make_public_callable(
    fn: Callable[..., Any], context_param: str
) -> Callable[..., Any]:
  """Returns a proxy callable with the injectable parameter removed.

  The proxy has the same ``__name__``, ``__doc__``, and a modified
  ``__signature__`` that excludes the ``context_param``. This is used
  for schema generation so the model never sees the injected parameter.

  Args:
    fn: The original callable.
    context_param: Name of the parameter to hide.

  Returns:
    A wrapper callable with a cleaned signature.
  """
  target = fn.fn if isinstance(fn, ToolWithSchema) else fn
  sig = inspect.signature(target)
  new_params = [p for n, p in sig.parameters.items() if n != context_param]
  public_sig = sig.replace(parameters=new_params)

  @functools.wraps(target)
  def _proxy(**kwargs):
    return target(**kwargs)

  _proxy.__signature__ = public_sig
  return _proxy


class ToolWithSchema(collections.abc.Callable):
  """Wrapper for callables with an explicit JSON Schema."""

  def __init__(self, fn: Callable[..., Any], input_schema: dict[str, Any]):
    self.fn = fn
    self.input_schema = input_schema
    self.__name__ = fn.__name__
    self.__doc__ = fn.__doc__

  def __call__(self, **kwargs: Any) -> Any:
    return self.fn(**kwargs)


class ToolRunner:
  """Registry and executor for in-process Python tools.

  Tools are registered by name and executed on demand. Both sync and async
  tools are supported. Tools that declare a ``ToolContext``-typed parameter
  receive the context automatically at execution time.
  """

  def __init__(self, tools: list[types.PythonTool] | None = None):
    self._tools: dict[str, types.PythonTool] = {}
    self._context: tool_context_module.ToolContext | None = None
    # Maps tool name → parameter name for ToolContext injection.
    # Populated at registration time to avoid per-call introspection.
    self._context_params: dict[str, str] = {}
    if tools:
      for tool in tools:
        self.register(tool)

  def set_context(self, ctx: tool_context_module.ToolContext) -> None:
    """Sets the ToolContext for injection into tools that request it.

    Args:
      ctx: The ToolContext to inject.
    """
    self._context = ctx

  def register(self, tool: types.PythonTool, name: str | None = None) -> None:
    """Registers a tool by name.

    At registration time, the tool's signature is inspected for a
    ``ToolContext``-typed parameter. If found, the parameter name is
    cached for injection at execution time.

    Args:
      tool: The callable to register.
      name: Optional name override. Defaults to tool.__name__.

    Raises:
      ValueError: If a tool with the same name is already registered.
    """
    tool_name = name or tool.__name__
    if tool_name in self._tools:
      raise ValueError(f"Tool '{tool_name}' is already registered.")
    self._tools[tool_name] = tool

    # Cache the ToolContext parameter name (if any) at registration time.
    ctx_param = _find_context_param(tool)
    if ctx_param is not None:
      self._context_params[tool_name] = ctx_param

  def unregister(self, name: str) -> None:
    """Removes a tool by name.

    Args:
      name: The name of the tool to remove.

    Raises:
      KeyError: If no tool with the given name is registered.
    """
    if name not in self._tools:
      raise KeyError(f"Tool '{name}' is not registered.")
    del self._tools[name]
    self._context_params.pop(name, None)

  @property
  def tool_names(self) -> list[str]:
    """The names of all registered tools."""
    return list(self._tools.keys())

  @property
  def tools(self) -> dict[str, types.PythonTool]:
    """A copy of the registered tools dictionary."""
    return dict(self._tools)

  def get_public_callable(self, tool_name: str) -> Callable[..., Any]:
    """Returns a callable with injectable params hidden from the signature.

    Connections use this to generate tool schemas for the model without
    exposing internal-only parameters like ``ToolContext``.

    Args:
      tool_name: The name of the registered tool.

    Returns:
      The original callable if no context parameter, otherwise a proxy
      callable with the context parameter removed from the signature.

    Raises:
      KeyError: If no tool with the given name is registered.
    """
    if tool_name not in self._tools:
      raise KeyError(f"Tool '{tool_name}' is not registered.")
    tool = self._tools[tool_name]
    ctx_param = self._context_params.get(tool_name)
    if ctx_param is None:
      return tool
    return _make_public_callable(tool, ctx_param)

  async def _execute_fn(self, fn: Callable[..., Any], **kwargs: Any) -> Any:
    """Executes a callable, running sync functions in a separate thread."""

    def is_async(callable_obj):
      return inspect.iscoroutinefunction(callable_obj) or (
          hasattr(callable_obj, "__call__")
          and inspect.iscoroutinefunction(callable_obj.__call__)
      )

    if not is_async(fn):
      result = await asyncio.to_thread(fn, **kwargs)
    else:
      result = fn(**kwargs)

    if asyncio.iscoroutine(result):
      return await result
    return result

  def _inject_context(
      self, tool_name: str, kwargs: dict[str, Any]
  ) -> dict[str, Any]:
    """Adds the ToolContext to kwargs if the tool requests it.

    Args:
      tool_name: The registered tool name.
      kwargs: The original keyword arguments.

    Returns:
      The kwargs dict, potentially augmented with the ToolContext.
    """
    ctx_param = self._context_params.get(tool_name)
    if ctx_param is not None and self._context is not None:
      if ctx_param not in kwargs:
        kwargs[ctx_param] = self._context
    return kwargs

  async def execute(self, tool_name: str, **kwargs: Any) -> Any:
    """Executes a registered tool by name.

    If the tool declares a ``ToolContext`` parameter and a context has
    been set via ``set_context()``, the context is injected automatically.

    Args:
      tool_name: The name of the tool to execute.
      **kwargs: Arguments to pass to the tool.

    Returns:
      The tool's return value.

    Raises:
      KeyError: If no tool with the given name is registered.
    """
    if tool_name not in self._tools:
      raise KeyError(f"Tool '{tool_name}' is not registered.")

    tool_fn = self._tools[tool_name]
    kwargs = self._inject_context(tool_name, kwargs)
    try:
      return await self._execute_fn(tool_fn, **kwargs)
    except Exception as e:  # pylint: disable=broad-except
      return f"Error executing tool '{tool_name}': {e}"

  async def process_tool_calls(
      self,
      tool_calls: list[types.ToolCall],
  ) -> list[types.ToolResult]:
    """Executes a batch of normalized tool calls and returns structured results.

    Returns one ToolResult per call. Unknown tools and execution failures
    produce ToolResult with an error message rather than raising.

    Args:
      tool_calls: List of ToolCall objects.

    Returns:
      A list of ToolResult, one per input tool call.
    """
    results = []
    for tc in tool_calls:
      if tc.name not in self._tools:
        results.append(
            types.ToolResult(
                name=tc.name, error=f"Unknown tool: '{tc.name}'"
            )
        )
        continue
      tool_fn = self._tools[tc.name]
      injected_args = self._inject_context(tc.name, dict(tc.args))
      try:
        result = await self._execute_fn(tool_fn, **injected_args)
        results.append(types.ToolResult(name=tc.name, result=result))
      except Exception as e:  # pylint: disable=broad-except
        results.append(
            types.ToolResult(
                name=tc.name,
                error=f"Error executing tool '{tc.name}': {e}",
                exception=e,
            )
        )
    return results
