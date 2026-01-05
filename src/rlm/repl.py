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
"""

import io
import sys
import traceback
from typing import Callable, Any

# Default output truncation limit (from paper: ~30K chars)
DEFAULT_MAX_OUTPUT_CHARS = 30000


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
    ):
        """
        Initialize REPL environment.

        Args:
            context: The long text/data to process. Can be string or list of strings.
            llm_query_func: Function for recursive LLM calls. Signature: (query: str) -> str
            max_output_chars: Maximum characters to return from execution output.
        """
        self.max_output_chars = max_output_chars

        # The globals dict persists across executions
        # This is where all variables live
        self.globals: dict[str, Any] = {
            'context': context,
            '__builtins__': __builtins__,
        }

        # Inject llm_query if provided
        if llm_query_func is not None:
            self.globals['llm_query'] = llm_query_func

    def execute(self, code: str) -> str:
        """
        Execute Python code and return the output.

        The LLM should see what happened (stdout, errors)
        so it can adapt its strategy. This is the "observe" in execute-observe-iterate.

        Args:
            code: Python code to execute

        Returns:
            String containing stdout output or error message
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

        # Truncate output if too long
        output = self._truncate_output(output)

        return output

    def _truncate_output(self, output: str) -> str:
        """
        Truncate output to prevent context overflow.

        We need to fit output back into LLM's context window.
        Better to truncate than to crash or overflow.
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
