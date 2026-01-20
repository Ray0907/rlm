"""
RLM Core - The Main Execute-Observe-Iterate Loop

The loop is the "runtime" that coordinates between:
1. The LLM (decision maker - writes code, interprets results)
2. The REPL (execution environment - runs code, stores state)

The loop continues until the LLM signals it has an answer (FINAL/FINAL_VAR).
"""

import re
from typing import Any, Callable, Optional
from dataclasses import dataclass, field

import litellm

from .repl import ReplEnvironment
from .prompt import build_system_prompt, build_user_message


@dataclass
class IterationEvent:
    """Event data for each iteration step (for streaming)."""
    iteration: int
    event_type: str  # "llm_response", "code_execution", "sub_call", "final"
    content: str
    code_block: str | None = None
    output: str | None = None
    has_error: bool = False
    tokens_used: int = 0


@dataclass
class RLMResult:
    """Result from an RLM execution."""
    answer: str
    iterations: int
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    code_executions: int = 0
    sub_calls: int = 0
    history: list[dict] = field(default_factory=list)


class RLM:
    """
    Recursive Language Model implementation.

    The RLM treats context as part of the environment,
    not as input to the LLM. This allows processing arbitrarily long contexts.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        sub_model: str | None = None,
        max_iterations: int = 20,
        max_output_chars: int = 30000,
        verbose: bool = False,
        iteration_delay: float = 0,
    ):
        """
        Initialize RLM.

        Args:
            model: Primary model for the root LLM (via LiteLLM)
            sub_model: Model for recursive llm_query calls (defaults to same as model)
            max_iterations: Maximum iterations before forced termination
            max_output_chars: Maximum chars for REPL output truncation
            verbose: Print debug information
            iteration_delay: Seconds to wait between iterations (for rate limiting)
        """
        self.model = model
        self.sub_model = sub_model or model
        self.max_iterations = max_iterations
        self.max_output_chars = max_output_chars
        self.verbose = verbose
        self.iteration_delay = iteration_delay

        # Track metrics
        self._sub_call_count = 0

    def run(
        self,
        query: str,
        context: str | list[str],
        on_iteration: Optional[Callable[[IterationEvent], None]] = None,
    ) -> RLMResult:
        """
        Run RLM on a query with given context.

        This is the main entry point - the execute-observe-iterate loop.

        Args:
            query: The question/task to answer
            context: The data to process (stored in REPL, not sent to LLM)
            on_iteration: Optional callback for streaming iteration events.
                          Called with IterationEvent for each step.

        Returns:
            RLMResult with answer and metrics
        """
        self._sub_call_count = 0

        # Create REPL environment with context and llm_query function
        repl = ReplEnvironment(
            context=context,
            llm_query_func=self._create_llm_query_func(),
            max_output_chars=self.max_output_chars,
        )

        # Build initial messages
        context_info = repl.get_context_info()
        system_prompt = build_system_prompt(context_info)
        user_message = build_user_message(query)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        # Metrics
        total_input_tokens = 0
        total_output_tokens = 0
        code_executions = 0
        history = []

        # Track whether any code has been successfully executed
        has_executed_code = False

        # Main loop: execute-observe-iterate
        for iteration in range(self.max_iterations):
            # Rate limit delay (skip first iteration)
            if iteration > 0 and self.iteration_delay > 0:
                import time
                if self.verbose:
                    print(f"\n[Waiting {self.iteration_delay}s for rate limit...]")
                time.sleep(self.iteration_delay)

            if self.verbose:
                print(f"\n{'='*50}")
                print(f"Iteration {iteration + 1}/{self.max_iterations}")
                print('='*50)

            # Call LLM
            response = litellm.completion(
                model=self.model,
                messages=messages,
            )

            # Extract response content and tokens
            assistant_content = response.choices[0].message.content
            total_input_tokens += response.usage.prompt_tokens
            total_output_tokens += response.usage.completion_tokens

            if self.verbose:
                print(f"\n[LLM Response]\n{assistant_content[:500]}...")

            # Emit LLM response event
            if on_iteration:
                on_iteration(IterationEvent(
                    iteration=iteration + 1,
                    event_type="llm_response",
                    content=assistant_content,
                    tokens_used=response.usage.completion_tokens,
                ))

            # Add to history
            history.append({
                "iteration": iteration + 1,
                "role": "assistant",
                "content": assistant_content,
            })

            # Parse and execute code blocks FIRST (before checking FINAL)
            # This ensures variables are created before FINAL_VAR() is checked
            code_blocks = self._parse_code_blocks(assistant_content)

            has_error = False
            execution_output = ""

            if code_blocks:
                for i, code in enumerate(code_blocks):
                    code_executions += 1
                    if self.verbose:
                        print(f"\n[Executing Code Block {i+1}]\n{code[:300]}...")

                    output = repl.execute(code)
                    execution_output += f"[Code Block {i+1} Output]\n{output}\n\n"

                    # Check if this execution had an error
                    block_has_error = "[ERROR]" in output
                    if block_has_error:
                        has_error = True
                    else:
                        has_executed_code = True  # Mark successful code execution

                    if self.verbose:
                        print(f"\n[Output]\n{output[:500]}...")

                    # Emit code execution event
                    if on_iteration:
                        on_iteration(IterationEvent(
                            iteration=iteration + 1,
                            event_type="code_execution",
                            content=f"Code block {i+1} executed",
                            code_block=code,
                            output=output,
                            has_error=block_has_error,
                        ))

                history.append({
                    "iteration": iteration + 1,
                    "role": "execution",
                    "content": execution_output.strip(),
                })

            # Only check for final answer if:
            # 1. No code blocks (LLM is ready to answer without more computation)
            # 2. OR code ran successfully without errors
            # 3. AND at least one code block has been successfully executed
            # This prevents the LLM from giving answers before actually exploring the context
            if not code_blocks or not has_error:
                final_answer = self._extract_final(assistant_content, repl)
                if final_answer is not None:
                    # Reject FINAL if no code has been executed yet
                    if not has_executed_code:
                        if self.verbose:
                            print(f"\n[FINAL rejected - no code executed yet, must explore context first]")
                        # Force LLM to write code
                        messages.append({"role": "assistant", "content": assistant_content})
                        messages.append({
                            "role": "user",
                            "content": "You must write code to explore the context before providing a final answer. The context is stored in the `context` variable. Please write a ```repl code block to analyze it."
                        })
                        continue

                    # If there were code blocks, the LLM should see output first
                    # Only accept FINAL if there were no code blocks in this response
                    if not code_blocks:
                        if self.verbose:
                            print(f"\n[FINAL ANSWER] {final_answer}")

                        # Emit final event
                        if on_iteration:
                            on_iteration(IterationEvent(
                                iteration=iteration + 1,
                                event_type="final",
                                content=final_answer,
                            ))

                        return RLMResult(
                            answer=final_answer,
                            iterations=iteration + 1,
                            total_input_tokens=total_input_tokens,
                            total_output_tokens=total_output_tokens,
                            code_executions=code_executions,
                            sub_calls=self._sub_call_count,
                            history=history,
                        )
                    elif self.verbose:
                        print(f"\n[FINAL ignored - code blocks present, will iterate]")

            # Continue iteration - either no FINAL yet, or had errors to fix
            if not code_blocks:
                # No code blocks - prompt LLM to write code or give final answer
                messages.append({"role": "assistant", "content": assistant_content})
                messages.append({
                    "role": "user",
                    "content": "Please write code to explore the context, or provide your final answer using FINAL()."
                })
            else:
                # Add assistant response and execution output to messages
                messages.append({"role": "assistant", "content": assistant_content})
                messages.append({"role": "user", "content": execution_output.strip()})

        # Max iterations reached without answer
        return RLMResult(
            answer="[ERROR] Max iterations reached without finding an answer.",
            iterations=self.max_iterations,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            code_executions=code_executions,
            sub_calls=self._sub_call_count,
            history=history,
        )

    def _create_llm_query_func(self):
        """
        Create the llm_query function for recursive sub-calls.

        Sub-calls are regular LLM calls (not RLM).
        This prevents infinite recursion and keeps costs manageable.
        In the paper, recursion depth is limited to 1.
        """
        def llm_query(prompt: str) -> str:
            self._sub_call_count += 1

            if self.verbose:
                print(f"\n[Sub-call #{self._sub_call_count}]")
                print(f"Prompt: {prompt[:200]}...")

            response = litellm.completion(
                model=self.sub_model,
                messages=[{"role": "user", "content": prompt}],
            )

            result = response.choices[0].message.content

            if self.verbose:
                print(f"Response: {result[:200]}...")

            return result

        return llm_query

    def _parse_code_blocks(self, text: str) -> list[str]:
        """
        Extract code blocks marked with ```repl from text.

        Args:
            text: LLM response text

        Returns:
            List of code strings
        """
        # Match ```repl ... ``` blocks
        pattern = r'```repl\s*\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        return matches

    def _extract_final(self, text: str, repl: ReplEnvironment) -> str | None:
        """
        Extract final answer from FINAL() or FINAL_VAR() markers.

        Args:
            text: LLM response text
            repl: REPL environment (for getting variable values)

        Returns:
            Final answer string or None if not found
        """
        # Check for FINAL(answer)
        final_match = re.search(r'FINAL\(([^)]+)\)', text)
        if final_match:
            return final_match.group(1).strip()

        # Check for FINAL_VAR(variable_name)
        final_var_match = re.search(r'FINAL_VAR\(([^)]+)\)', text)
        if final_var_match:
            var_name = final_var_match.group(1).strip()
            value = repl.get_variable(var_name)
            if value is not None:
                return str(value)
            else:
                return f"[ERROR] Variable '{var_name}' not found"

        return None


# Convenience function
def run(
    query: str,
    context: str | list[str],
    model: str = "gpt-4o-mini",
    **kwargs,
) -> RLMResult:
    """
    Run RLM on a query with given context.

    This is the main API - simple and straightforward.

    Args:
        query: The question/task to answer
        context: The data to process
        model: LLM model to use (via LiteLLM)
        **kwargs: Additional arguments passed to RLM constructor

    Returns:
        RLMResult with answer and metrics

    Example:
        >>> import rlm
        >>> result = rlm.run(
        ...     query="What is the sum of all ages?",
        ...     context="Alice is 25. Bob is 30. Charlie is 35.",
        ...     model="gpt-4o-mini"
        ... )
        >>> print(result.answer)
        90
    """
    rlm_instance = RLM(model=model, **kwargs)
    return rlm_instance.run(query=query, context=context)
