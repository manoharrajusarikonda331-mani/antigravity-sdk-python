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

"""Local connection package for the Google Antigravity SDK.

Re-exports the public API so that existing import paths like
``from ...connections.local_connection import LocalAgentConfig``
continue to work without changes.
"""

from google.antigravity.connections.local.local_connection import callable_to_tool_proto
from google.antigravity.connections.local.local_connection import LocalConnection
from google.antigravity.connections.local.local_connection import LocalConnectionStep
from google.antigravity.connections.local.local_connection import LocalConnectionStrategy
from google.antigravity.connections.local.local_connection_config import LocalAgentConfig
from google.antigravity.connections.local.types import EditFileResult
from google.antigravity.connections.local.types import FindFileResult
from google.antigravity.connections.local.types import GenerateImageResult
from google.antigravity.connections.local.types import ListDirectoryEntry
from google.antigravity.connections.local.types import ListDirectoryResult
from google.antigravity.connections.local.types import RunCommandResult
from google.antigravity.connections.local.types import SearchDirectoryResult
from google.antigravity.connections.local.types import TextResult
from google.antigravity.connections.local.types import ToolOutput
