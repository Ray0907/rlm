"""
Benchmark: Output Offloading Effect on RLM Performance

Compares RLM with and without output offloading to measure:
1. History size growth
2. Token usage
3. Answer quality
4. Iteration count
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
import rlm
from rlm.repl import ReplEnvironment

load_dotenv()


def generate_employee_data(count: int) -> str:
    """Generate employee records for testing."""
    departments = ["Engineering", "Marketing", "Sales", "HR", "Finance"]
    records = ["Employee Records:\n"]

    for i in range(count):
        dept = departments[i % len(departments)]
        salary = 50000 + (i * 1000) + (i % 5) * 5000
        records.append(f"- Employee_{i:04d}: Age {25 + i % 40}, Department: {dept}, Salary: ${salary:,}")

    return "\n".join(records)


def track_history_size(result) -> dict:
    """Analyze history to measure output sizes."""
    total_chars = 0
    execution_outputs = []

    for entry in result.history:
        if entry.get("role") == "execution":
            content = entry.get("content", "")
            total_chars += len(content)
            execution_outputs.append(len(content))

    return {
        "total_history_chars": total_chars,
        "execution_count": len(execution_outputs),
        "avg_output_size": total_chars // len(execution_outputs) if execution_outputs else 0,
        "output_sizes": execution_outputs[:5],  # First 5 for preview
    }


def run_benchmark():
    print("=" * 70)
    print("BENCHMARK: Output Offloading Effect")
    print("=" * 70)

    # Generate test data - 50 employees
    context = generate_employee_data(50)
    query = "What is the total salary of all Engineering department employees? List each Engineering employee and their salary, then calculate the sum."

    print(f"\nContext: {len(context):,} chars ({context.count(chr(10))} lines)")
    print(f"Query: {query[:80]}...")

    model = os.getenv("LITELLM_MODEL", "anthropic/claude-haiku-3-5-20241022")
    print(f"Model: {model}")

    # === RUN 1: With Offloading (default: 500 char threshold) ===
    print("\n" + "-" * 70)
    print("RUN 1: WITH Output Offloading (threshold=500)")
    print("-" * 70)

    result_with = rlm.run(
        query=query,
        context=context,
        model=model,
        verbose=False,
        max_output_chars=30000,  # Keep default truncation
    )

    stats_with = track_history_size(result_with)

    print(f"Answer: {result_with.answer[:100]}...")
    print(f"Iterations: {result_with.iterations}")
    print(f"Code executions: {result_with.code_executions}")
    print(f"Input tokens: {result_with.total_input_tokens:,}")
    print(f"Output tokens: {result_with.total_output_tokens:,}")
    print(f"History size: {stats_with['total_history_chars']:,} chars")
    print(f"Avg output size: {stats_with['avg_output_size']} chars")

    # === RUN 2: Without Offloading (very high threshold) ===
    print("\n" + "-" * 70)
    print("RUN 2: WITHOUT Output Offloading (threshold=99999)")
    print("-" * 70)

    # Create custom RLM with high threshold (effectively disabling offloading)
    rlm_no_offload = rlm.RLM(
        model=model,
        verbose=False,
        max_output_chars=30000,
    )

    # Monkey-patch the REPL creation to use high threshold
    original_run = rlm_no_offload.run
    def patched_run(query, context, on_iteration=None):
        rlm_no_offload._sub_call_count = 0
        from rlm.repl import ReplEnvironment
        from rlm.prompt import build_system_prompt, build_user_message
        import litellm

        # Create REPL with HIGH threshold (no offloading)
        repl = ReplEnvironment(
            context=context,
            llm_query_func=rlm_no_offload._create_llm_query_func(),
            max_output_chars=30000,
            max_output_to_history=99999,  # Effectively disable offloading
        )

        context_info = repl.get_context_info()
        system_prompt = build_system_prompt(context_info)
        user_message = build_user_message(query)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        total_input_tokens = 0
        total_output_tokens = 0
        code_executions = 0
        history = []
        has_executed_code = False

        for iteration in range(rlm_no_offload.max_iterations):
            response = litellm.completion(model=rlm_no_offload.model, messages=messages)
            assistant_content = response.choices[0].message.content
            total_input_tokens += response.usage.prompt_tokens
            total_output_tokens += response.usage.completion_tokens

            history.append({"iteration": iteration + 1, "role": "assistant", "content": assistant_content})

            code_blocks = rlm_no_offload._parse_code_blocks(assistant_content)
            has_error = False
            execution_output = ""

            if code_blocks:
                for i, code in enumerate(code_blocks):
                    code_executions += 1
                    output = repl.execute(code)
                    execution_output += f"[Code Block {i+1} Output]\n{output}\n\n"
                    if "[ERROR]" in output:
                        has_error = True
                    else:
                        has_executed_code = True

                history.append({"iteration": iteration + 1, "role": "execution", "content": execution_output.strip()})

            if not code_blocks or not has_error:
                final_answer = rlm_no_offload._extract_final(assistant_content, repl)
                if final_answer is not None:
                    if not has_executed_code:
                        messages.append({"role": "assistant", "content": assistant_content})
                        messages.append({"role": "user", "content": "You must write code to explore the context before providing a final answer."})
                        continue
                    if not code_blocks:
                        return rlm.RLMResult(
                            answer=final_answer,
                            iterations=iteration + 1,
                            total_input_tokens=total_input_tokens,
                            total_output_tokens=total_output_tokens,
                            code_executions=code_executions,
                            sub_calls=rlm_no_offload._sub_call_count,
                            history=history,
                        )

            if not code_blocks:
                messages.append({"role": "assistant", "content": assistant_content})
                messages.append({"role": "user", "content": "Please write code to explore the context, or provide your final answer using FINAL()."})
            else:
                messages.append({"role": "assistant", "content": assistant_content})
                messages.append({"role": "user", "content": execution_output.strip()})

        return rlm.RLMResult(
            answer="[ERROR] Max iterations reached",
            iterations=rlm_no_offload.max_iterations,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            code_executions=code_executions,
            sub_calls=rlm_no_offload._sub_call_count,
            history=history,
        )

    result_without = patched_run(query, context)
    stats_without = track_history_size(result_without)

    print(f"Answer: {result_without.answer[:100]}...")
    print(f"Iterations: {result_without.iterations}")
    print(f"Code executions: {result_without.code_executions}")
    print(f"Input tokens: {result_without.total_input_tokens:,}")
    print(f"Output tokens: {result_without.total_output_tokens:,}")
    print(f"History size: {stats_without['total_history_chars']:,} chars")
    print(f"Avg output size: {stats_without['avg_output_size']} chars")

    # === COMPARISON ===
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'With Offload':<20} {'Without Offload':<20} {'Diff':<15}")
    print("-" * 80)

    metrics = [
        ("Iterations", result_with.iterations, result_without.iterations),
        ("Code executions", result_with.code_executions, result_without.code_executions),
        ("Input tokens", result_with.total_input_tokens, result_without.total_input_tokens),
        ("Output tokens", result_with.total_output_tokens, result_without.total_output_tokens),
        ("History size (chars)", stats_with['total_history_chars'], stats_without['total_history_chars']),
        ("Avg output size", stats_with['avg_output_size'], stats_without['avg_output_size']),
    ]

    for name, with_val, without_val in metrics:
        if isinstance(with_val, int):
            diff = without_val - with_val
            diff_pct = f"({diff:+,})" if diff != 0 else "(same)"
            print(f"{name:<25} {with_val:<20,} {without_val:<20,} {diff_pct:<15}")

    # Token savings
    token_diff = result_without.total_input_tokens - result_with.total_input_tokens
    if result_without.total_input_tokens > 0:
        token_pct = (token_diff / result_without.total_input_tokens) * 100
        print(f"\nToken savings: {token_diff:,} tokens ({token_pct:.1f}%)")

    # History reduction
    history_diff = stats_without['total_history_chars'] - stats_with['total_history_chars']
    if stats_without['total_history_chars'] > 0:
        history_pct = (history_diff / stats_without['total_history_chars']) * 100
        print(f"History reduction: {history_diff:,} chars ({history_pct:.1f}%)")


if __name__ == "__main__":
    run_benchmark()
