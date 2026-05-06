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

"""Google Antigravity SDK for building AI agents."""

from google.antigravity.agent import Agent
from google.antigravity.connections.connection import AgentConfig
from google.antigravity.connections.local import LocalAgentConfig
from google.antigravity.tools.tool_context import ToolContext
from google.antigravity.types import CapabilitiesConfig
from google.antigravity.types import GeminiConfig
from google.antigravity.types import GenerationConfig
from google.antigravity.types import ModelConfig
from google.antigravity.types import ModelEntry
from google.antigravity.types import ThinkingLevel
