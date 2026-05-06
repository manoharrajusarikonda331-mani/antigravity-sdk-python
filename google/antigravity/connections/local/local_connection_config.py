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

"""Configuration for the local harness connection strategy."""

import logging
import tempfile
from typing import Any

import pydantic

from google.antigravity import types
from google.antigravity.connections import connection


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
  ) -> "connection.ConnectionStrategy":
    # Late import to avoid circular dependency: local_connection.py imports
    # this config module, so we import the strategy class here at call time.
    from google.antigravity.connections.local import local_connection  # pylint: disable=g-import-not-at-top

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

    return local_connection.LocalConnectionStrategy(
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
