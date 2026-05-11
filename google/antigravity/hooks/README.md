# Antigravity Hook Architecture: Design and Implementation

This document describes the design decisions and current implementation of the
hook system in the Google Antigravity SDK, designed to support a
granular, secure, and symmetrical lifecycle.

## Overview

Hooks in the Google Antigravity SDK allow users and system components to intercept,
observe, and modify the behavior of the agent at various stages of its execution
lifecycle. They are essential for observability, policy enforcement, data
sanitization, and interactive decision-making.

## Hook Taxonomy

To ensure clear semantics and predictable behavior, hooks are classified into
three strict categories:

### 1. Inspect Hooks (Read-Only, Non-Blocking)

-   **Purpose**: Observability, logging, and monitoring.
-   **Behavior**: They receive data but cannot modify it. They cannot block
    execution. They are executed asynchronously or concurrently without delaying
    the main flow.
-   **Examples**: `PostToolCallHook`.

### 2. Decide Hooks (Read-Only, Blocking)

-   **Purpose**: Policy enforcement, permission checks, and guardrails.
-   **Behavior**: They receive data and return a `HookResult` indicating whether
    execution should proceed (`allow=True`) or be aborted (`allow=False`). They
    cannot modify the data.
-   **Examples**: `PreToolCallDecideHook`.

### 3. Transform Hooks (Modifying, Blocking)

-   **Purpose**: Data sanitization, prompt optimization, error recovery, and
    interactive responses.
-   **Behavior**: They receive data, can modify it, and must return the
    (potentially modified) data. They can also fail, triggering a fail-closed
    behavior.
-   **Examples**: `OnToolErrorHook`, `OnInteractionHook`.

## Execution Order and Security (TOCTOU)

For events that support multiple hook types (e.g., `PreToolCall`), the
`HookRunner` enforces a strict execution order to prevent **Time-of-Check to
Time-of-Use (TOCTOU)** vulnerabilities:

1.  **Decisions**: Executed first to validate the data. If any Decide hook
    denies, execution is aborted immediately.
2.  **Inspections**: Executed after the operation completes to log or observe
    the actual execution context.

Example for `PreToolCall`: `PreToolCallDecideHook` $\rightarrow$
(Tool Execution) $\rightarrow$ `PostToolCallHook`.

## Context Management

Hooks operate within a hierarchical context system that allows state sharing and
correlation across different lifecycle events:

1.  **`SessionContext`**: Scoped to the entire agent session.
2.  **`TurnContext`**: Scoped to a single turn (prompt/response cycle). Inherits
    from `SessionContext`.
3.  **`OperationContext`**: Scoped to a specific operation (e.g., a model call
    or tool call). Inherits from `TurnContext`.

This hierarchy ensures that state set in a broader scope is visible to narrower
scopes, but not from narrower to broader scopes, preventing cross-talk and
ensuring proper cleanup.

## Connection Compatibility

Hook behavior depends on the connection type.

### `LocalConnection`

-   **Built-in tool hooks** (view_file, run_command, edit_file, etc.):
    `PreToolCallDecideHook` runs and can **approve or deny** built-in tools.
    `PostToolCallHook` fires when the harness reports the tool as complete.
    `OnToolErrorHook` fires when the tool fails.

-   **Built-in tool results**: When `PostToolCallHook` fires for a built-in
    tool, the `ToolResult.result` field contains the tool's output as a
    string. The following table shows what each tool surfaces:

    | Tool | Result Content |
    |---|---|
    | `run_command` | Combined stdout and stderr |
    | `list_dir` | Formatted listing with names, types, and sizes |
    | `find_by_name` | Newline-separated list of matching filenames |
    | `grep_search` | Number of results found |
    | `view_file` | File content (falls back to step text) |
    | `edit_file` | Diff summary (falls back to step text) |
    | `generate_image` | Generated image name |
    | Other tools | The step's `text` field |

    Large outputs may be truncated before delivery.

-   **Host-side (custom Python and MCP) tools**: The full hook pipeline runs
    (Decide → Execute → PostToolCall / OnToolError).

-   **Subagent hooks**: Subagent invocations appear as `START_SUBAGENT` tool
    calls. `PreToolCallDecideHook` fires before the subagent starts, and
    `PostToolCallHook` fires when the subagent trajectory goes idle, with the
    subagent's final response as the result. Additionally, hooks fire for
    **every tool call within a subagent trajectory**. When a subagent calls
    `run_command` or `view_file`, the parent's `PreToolCallDecideHook` and
    `PostToolCallHook` fire for those calls. This means a policy that denies
    `run_command` will deny it whether the parent or any subagent calls it.

## Observing Model Responses

To observe model-generated text:

-   Use **`PostTurnHook`**, which receives the complete model response after
    each agent turn completes.
-   Inspect **`conversation.history`** for the full step-by-step trajectory,
    including intermediate model steps.

## Fail-Safe Strategy

For security-critical operations, the system adopts a **fail-closed** strategy-
if a **Decision Hook** denies execution, the operation is aborted.

## Policies

The `policy` module provides a declarative API for expressing tool call
policies. Rather than writing raw `PreToolCallDecideHook` implementations,
developers define policies using builder functions and let the system handle
evaluation:

```python
from google.antigravity.hooks import policy

policies = [
    policy.deny("*"),                       # Block everything by default
    policy.allow("view_file"),              # Except reading files
    policy.deny("run_command",              # Block dangerous commands
        when=lambda args: "rm" in args.get("CommandLine", "")),
    policy.ask_user("run_command",          # Ask for safe commands
        handler=my_approval_fn),
]

hook = policy.enforce(policies)
# Register: HookRunner(pre_tool_call_decide_hooks=[hook])
```

### Priority Model

Policies are evaluated using a priority model where specificity and safety
determine precedence. Within each level, **first match wins** (short-circuit):

Level | Specificity | Decision   | Example
----- | ----------- | ---------- | ------------------------------
1     | Specific    | `DENY`     | `deny("run_command")`
2     | Specific    | `ASK_USER` | `ask_user("run_command", ...)`
3     | Specific    | `APPROVE`  | `allow("run_command")`
4     | Wildcard    | `DENY`     | `deny("*")`
5     | Wildcard    | `ASK_USER` | `ask_user("*", ...)`
6     | Wildcard    | `APPROVE`  | `allow("*")`

A policy is "specific" when its tool name is an exact tool name, and "wildcard"
when the tool name is `"*"`.

### Predicates

Policies support optional `when` predicates that inspect the tool call
arguments:

```python
policy.deny("run_command",
    when=lambda args: "rm" in args.get("CommandLine", ""))
```

Predicates can be sync or async. If a predicate raises an exception, the policy
**matches** (fail-closed), ensuring safety.

### ASK_USER

`ASK_USER` policies require a handler function that receives the full `ToolCall`
and returns `True` (approve) or `False` (deny):

```python
async def confirm_with_user(tc: types.ToolCall) -> bool:
    response = input(f"Allow {tc.name}? (y/n): ")
    return response.lower() == "y"

policy.ask_user("run_command", handler=confirm_with_user)
```

`enforce()` validates at construction time that all `ASK_USER` policies have
handlers, failing fast with a `ValueError` if any are missing.

## Current Implementation

The implementation is split across the following core files:

-   **`types.py`** (SDK root): Defines the canonical Pydantic V2 boundary types
    (`ToolCall`, `Step`, `ToolResult`, `HookResult`, `QuestionResponse`,
    `QuestionHookResult`). All hook interfaces use these types. `HookResult`,
    `QuestionResponse`, and `QuestionHookResult` are re-exported from `hooks.py`
    for convenience.
-   **`hooks.py`**: Defines the base classes for `HookContext`, `HookResult`,
    and the specialized hook interfaces (e.g., `PreToolCallDecideHook`).
-   **`hook_runner.py`**: Implements the `HookRunner` class, which manages the
    hook collections and implements the strict execution order dispatch logic.
-   **`cli.py`**: Provides concrete implementations of hooks for interactive CLI
    usage, such as `ToolConfirmationHook` and `AskQuestionHook`.
-   **`policy.py`**: Declarative tool call policy system with priority-based
    evaluation. Produces a `PreToolCallDecideHook` from a list of policies.

## Tests

Comprehensive unit tests are provided in:

-   **`hooks_test.py`**: Verifies base class behavior.
-   **`hook_runner_test.py`**: Verifies execution order, context scoping,
    fail-closed behavior, and streaming dispatch.
-   **`cli_test.py`**: Verifies interactive CLI hooks.
-   **`policy_test.py`**: Verifies priority evaluation, short-circuiting,
    predicate handling, ASK_USER handlers, and HookRunner integration.

## Known Limitations

-   **Pre-turn hooks are SDK-side only.** The `pre_turn` hook intercepts
    user-initiated `send()` calls but cannot guard against Connection-initiated
    turns (e.g., background task completions, cron triggers). Full
    Connection-level turn interception requires protocol-level changes and will
    be addressed in a subsequent hooks refresh.

## See Also

-   **[Triggers](../triggers/README.md)**: For long-lived background tasks that
    react to external events (cron, file changes, webhooks) and push messages
    into the agent. Hooks handle agent lifecycle; triggers handle external
    events.
