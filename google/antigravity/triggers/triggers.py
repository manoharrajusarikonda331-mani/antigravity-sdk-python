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

"""Core trigger interface for the Google Antigravity SDK.

A Trigger is a long-lived async function that runs alongside an agent
session. It reacts to external events (cron, file changes, webhooks)
and pushes messages back into the agent.
"""

from typing import Awaitable, Callable

from google.antigravity.connections import connection as connection_module


class TriggerContext:
  """Handle provided to every trigger at startup.

  Provides the capability to send messages to the agent. One
  TriggerContext is created per trigger.
  """

  def __init__(
      self,
      connection: connection_module.Connection,
  ) -> None:
    self._connection = connection

  async def send(
      self,
      content: str,
  ) -> None:
    """Sends a message to the agent.

    Args:
      content: The message content.
    """
    await self._connection.send_trigger_notification(content)


# A Trigger is any async function that accepts a TriggerContext.
Trigger = Callable[[TriggerContext], Awaitable[None]]
