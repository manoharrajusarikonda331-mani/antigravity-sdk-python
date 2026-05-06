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

"""Example demonstrating structured output from an agent using a custom tool."""

import asyncio
import logging

import pydantic

from google.antigravity import agent
from google.antigravity.connections.local.local_connection_config import LocalAgentConfig


class ActionItem(pydantic.BaseModel):
  assignee: str
  task: str
  deadline: str


class MeetingSummary(pydantic.BaseModel):
  action_items: list[ActionItem]


# A custom mock tool that retrieves unstructured text data
async def fetch_unstructured_meeting_notes(meeting_id: str) -> str:
  """Retrieves the raw unstructured notes for a given meeting ID."""
  if meeting_id == "meeting-2026-05":
    return (
        "Discussed launch timeline for project X. Alice agreed to update"
        " the textproto tests by Monday. Bob mentioned he will run the final"
        " E2E benchmarks tomorrow. I will push the release build once the"
        " tests are green."
    )
  return "Error: Meeting notes not found."


async def run():
  """Runs the structured output example."""
  logging.basicConfig(level=logging.INFO)

  try:
    config = LocalAgentConfig(
        system_instructions="You are a helpful assistant.",
        tools=[fetch_unstructured_meeting_notes],
        response_schema=MeetingSummary,
    )
    async with agent.Agent(config) as meeting_agent:

      prompt = (
          "Retrieve the notes for 'meeting-2026-05' and return structured"
          " action items for the meeting attendees in the correct format"
          " needed."
      )

      print("\nSending prompt to agent...")
      response = await meeting_agent.chat(prompt)

      if response.structured_output:
        print("\n=== Structured Meeting Action Items ===")
        for item in response.structured_output.get("action_items", []):
          print(f"- Assignee: {item.get('assignee')}")
          print(f"  Task:     {item.get('task')}")
          print(f"  Deadline: {item.get('deadline')}\n")
      else:
        print("\nFailed to extract structured summary natively.")
        print(f"Final Text Response: {response.text}")

  except Exception:  # pylint: disable=broad-exception-caught
    logging.exception("Execution failed")


if __name__ == "__main__":
  asyncio.run(run())
