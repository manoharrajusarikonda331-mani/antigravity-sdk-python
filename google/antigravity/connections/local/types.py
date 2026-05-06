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

"""Structured result types for the local harness connection."""

import os
import pydantic


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
