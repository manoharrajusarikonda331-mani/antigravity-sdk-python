# Antigravity SDK Conversation

The `conversation` package provides the `Conversation` class, which is the Layer 2 session API in the Antigravity SDK. It wraps a `Connection` and adds stateful session management features.

## Core Concepts

### `Conversation`

`Conversation` is the primary interface for power users who need more control than the high-level `Agent` class provides, but don't want to deal with the low-level details of `Connection`.

Key features:
- **Step History Accumulation**: It automatically records all `Step` objects received from the connection.
- **History Limits**: It supports a maximum history size to prevent memory issues in long sessions, discarding oldest steps when the limit is exceeded.
- **Turn Tracking**: It tracks where each turn (user prompt) starts in the history.
- **Compaction Tracking**: It tracks where the model's context was compacted.
- **Convenience Methods**:
    - `chat(prompt)`: Sends a prompt and waits for the complete response, returning a `ChatResponse`. Natively supports both standard strings and rich `types.Content` multimodal payloads (lists of text strings and `types.Part` files).
    - `send(prompt)`: Sends a prompt turn (non-blocking). Accepts strings or complex multimodal payloads.
    - `receive_steps()`: Async iterator for receiving steps for the current turn.

## Usage Example

### Using `chat()` (High-level)

```python
from google.antigravity.conversation.conversation import Conversation
from google.antigravity.connections.local_connection import LocalConnectionStrategy
from google.antigravity import types

strategy = LocalConnectionStrategy(...)

async with Conversation.create(strategy) as conversation:
    response = await conversation.chat("What files are in the current directory?")
    print(response.text)
    
    print(f"Total steps: {len(conversation.history)}")
```

### Using `send()` and `receive_steps()` (Low-level/Streaming)

```python
async with Conversation.create(strategy) as conversation:
    await conversation.send("Tell me a story.")
    async for step in conversation.receive_steps():
        if step.type == types.StepType.MODEL_RESPONSE:
            print(step.content, end="")
```

## Files

- `conversation.py`: Defines the `Conversation` class.
