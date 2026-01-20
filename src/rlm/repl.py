"""
REPL Environment for RLM

The REPL is the "environment" where context lives as a variable,
not as part of the LLM's input. The LLM operates ON the environment, not IN it.

Core responsibilities:
1. Hold context as a variable (not in LLM prompt)
2. Inject llm_query() for recursive sub-calls
3. Execute LLM-generated code and capture output
4. Handle errors gracefully (return error to LLM for self-correction)
5. Persist variables across iterations
6. Auto-offload large outputs to prevent context rot
"""

import io
import sys
import traceback
from typing import Callable, Any

# Default output truncation limit (from paper: ~30K chars)
DEFAULT_MAX_OUTPUT_CHARS = 30000

# Default threshold for auto-offloading output to variables
# Outputs larger than this go into _result_N variables instead of history
DEFAULT_MAX_OUTPUT_TO_HISTORY = 500

# Preview length for offloaded outputs
DEFAULT_PREVIEW_LENGTH = 200


class ReplEnvironment:
    """
    A sandboxed Python execution environment for RLM.

    The key insight: context is stored HERE, not in the LLM's context window.
    The LLM writes code to explore this environment.
    """

    def __init__(
        self,
        context: str | list[str],
        llm_query_func: Callable[[str], str] | None = None,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
        max_output_to_history: int = DEFAULT_MAX_OUTPUT_TO_HISTORY,
        preview_length: int = DEFAULT_PREVIEW_LENGTH,
    ):
        """
        Initialize REPL environment.

        Args:
            context: The long text/data to process. Can be string or list of strings.
            llm_query_func: Function for recursive LLM calls. Signature: (query: str) -> str
            max_output_chars: Maximum characters to return from execution output.
            max_output_to_history: Threshold for auto-offloading outputs. Outputs larger
                                   than this are stored as variables instead of returned.
            preview_length: Number of characters to show in preview for offloaded outputs.
        """
        self.max_output_chars = max_output_chars
        self.max_output_to_history = max_output_to_history
        self.preview_length = preview_length
        self._result_counter = 0

        # The globals dict persists across executions
        # This is where all variables live
        self.globals: dict[str, Any] = {
            'context': context,
            '__builtins__': __builtins__,
        }

        # Inject llm_query if provided
        if llm_query_func is not None:
            self.globals['llm_query'] = llm_query_func

        # Inject memory management functions
        self.globals['forget'] = self._forget
        self.globals['memory_status'] = self._memory_status

    def execute(self, code: str) -> str:
        """
        Execute Python code and return the output.

        The LLM should see what happened (stdout, errors)
        so it can adapt its strategy. This is the "observe" in execute-observe-iterate.

        Large outputs are automatically offloaded to variables to prevent context rot.
        The LLM sees a preview and can access full data via the variable name.

        Args:
            code: Python code to execute

        Returns:
            String containing stdout output (or preview if offloaded) or error message
        """
        # Capture stdout
        old_stdout = sys.stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            # Execute the code in our persistent globals
            exec(code, self.globals)
            output = captured_output.getvalue()

        except Exception as e:
            # Return error to LLM so it can self-correct
            # This is important: errors are feedback, not failures
            error_trace = traceback.format_exc()
            output = f"[ERROR] {type(e).__name__}: {e}\n\nTraceback:\n{error_trace}"

        finally:
            sys.stdout = old_stdout

        # Auto-offload large outputs to prevent context rot
        output = self._maybe_offload_output(output)

        # Truncate output if still too long (safety net)
        output = self._truncate_output(output)

        return output

    def _maybe_offload_output(self, output: str) -> str:
        """
        Auto-offload large outputs to variables to prevent context rot.

        If output exceeds max_output_to_history threshold, store it as a variable
        and return only a preview with reference. This keeps history small while
        preserving access to full data.

        Args:
            output: The raw output from code execution

        Returns:
            Original output if small, or preview with variable reference if large
        """
        # Don't offload errors - LLM needs to see full error
        if output.startswith("[ERROR]"):
            return output

        # Don't offload small outputs
        if len(output) <= self.max_output_to_history:
            return output

        # Offload large output to a variable
        var_name = f"_result_{self._result_counter}"
        self._result_counter += 1
        self.globals[var_name] = output

        # Create preview
        preview = output[:self.preview_length]
        if len(output) > self.preview_length:
            preview += "..."

        return (
            f"[OUTPUT OFFLOADED to {var_name}, {len(output)} chars]\n"
            f"Preview: {preview}\n"
            f"Access full output with: print({var_name})"
        )

    def _truncate_output(self, output: str) -> str:
        """
        Truncate output to prevent context overflow.

        We need to fit output back into LLM's context window.
        Better to truncate than to crash or overflow.
        This is a safety net after offloading.
        """
        if len(output) > self.max_output_chars:
            truncated = output[:self.max_output_chars]
            truncated += f"\n\n[OUTPUT TRUNCATED: {len(output)} chars -> {self.max_output_chars} chars]"
            return truncated
        return output

    def get_variable(self, name: str) -> Any:
        """
        Get a variable from the REPL environment.

        Useful for FINAL_VAR() - retrieve the variable to return as answer.
        """
        return self.globals.get(name)

    def get_context_info(self) -> dict:
        """
        Get metadata about the context.

        This is included in the system prompt so LLM knows what it's working with.
        """
        context = self.globals['context']

        if isinstance(context, str):
            return {
                'type': 'str',
                'total_chars': len(context),
                'preview': context[:200] + '...' if len(context) > 200 else context,
            }
        elif isinstance(context, list):
            return {
                'type': 'list',
                'length': len(context),
                'total_chars': sum(len(str(item)) for item in context),
                'item_lengths': [len(str(item)) for item in context[:10]],  # First 10 items
            }
        else:
            return {
                'type': str(type(context).__name__),
                'total_chars': len(str(context)),
            }

    def _forget(self, *var_names: str) -> str:
        """
        Clear variables from REPL memory.

        This is injected into the REPL globals so LLM can call it directly.
        Useful for garbage collection when intermediate results are no longer needed.

        Args:
            *var_names: Variable names to forget

        Returns:
            Status message
        """
        # Protected names that cannot be forgotten
        protected = {'context', 'llm_query', 'forget', 'memory_status', '__builtins__'}

        cleared = []
        skipped = []

        for name in var_names:
            if name in protected:
                skipped.append(f"{name} (protected)")
            elif name in self.globals:
                del self.globals[name]
                cleared.append(name)
            else:
                skipped.append(f"{name} (not found)")

        result = []
        if cleared:
            result.append(f"Cleared: {', '.join(cleared)}")
        if skipped:
            result.append(f"Skipped: {', '.join(skipped)}")

        return "\n".join(result) if result else "Nothing to clear"

    def _memory_status(self) -> dict:
        """
        Show current memory usage of REPL variables.

        This is injected into the REPL globals so LLM can call it directly.
        Helps LLM understand what's stored and decide what to forget.

        Returns:
            Dict mapping variable names to their sizes (in chars)
        """
        # Protected/internal names to exclude from display
        internal = {'__builtins__', 'forget', 'memory_status'}

        status = {}
        for name, value in self.globals.items():
            if name in internal:
                continue

            # Calculate size
            try:
                size = len(str(value))
            except Exception:
                size = -1  # Unknown size

            # Mark special variables
            if name == 'context':
                status[f"{name} (context)"] = size
            elif name == 'llm_query':
                status[f"{name} (function)"] = "callable"
            elif name.startswith('_result_'):
                status[f"{name} (offloaded)"] = size
            else:
                status[name] = size

        return status
