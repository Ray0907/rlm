"""
Long Document Processing Example for RLM

Demonstrates RLM's ability to process documents that would exceed
a typical LLM's context window by using chunking and recursive calls.
"""

import os
import sys

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

import rlm

# Load environment variables
load_dotenv()


def generate_long_document(num_entries: int = 100) -> str:
    """Generate a synthetic long document with many entries."""
    entries = []
    departments = ["Engineering", "Marketing", "Sales", "HR", "Finance"]

    for i in range(num_entries):
        name = f"Employee_{i:04d}"
        age = 22 + (i % 40)
        dept = departments[i % len(departments)]
        salary = 50000 + (i * 500) + (i % 10) * 1000
        performance = ["Excellent", "Good", "Average", "Needs Improvement"][i % 4]

        entry = f"""
Entry {i + 1}:
  Name: {name}
  Age: {age}
  Department: {dept}
  Annual Salary: ${salary:,}
  Performance Rating: {performance}
  Years at Company: {i % 15}
  Projects Completed: {(i * 3) % 50}
"""
        entries.append(entry)

    header = """
=== COMPANY EMPLOYEE DATABASE ===
Generated for RLM Long Document Processing Demo

This document contains detailed employee records.
Each entry includes name, age, department, salary, and performance data.
"""

    return header + "\n".join(entries)


def main():
    # Generate a document with many entries
    num_entries = 50  # Adjust for longer tests
    context = generate_long_document(num_entries)

    query = """
    Analyze this employee database and answer:
    1. How many employees are in each department?
    2. What is the average salary per department?
    3. Which department has the highest average salary?
    """

    print("=" * 60)
    print("RLM Long Document Processing Example")
    print("=" * 60)
    print(f"\nQuery: {query}")
    print(f"\nContext length: {len(context):,} chars ({num_entries} entries)")
    print("-" * 60)

    # Run RLM with verbose output
    result = rlm.run(
        query=query,
        context=context,
        model=os.getenv("LITELLM_MODEL", "anthropic/claude-sonnet-4-5-20250929"),
        verbose=True,
        max_iterations=15,
    )

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Answer:\n{result.answer}")
    print("-" * 60)
    print(f"Iterations: {result.iterations}")
    print(f"Code executions: {result.code_executions}")
    print(f"Sub-calls (llm_query): {result.sub_calls}")
    print(f"Total input tokens: {result.total_input_tokens:,}")
    print(f"Total output tokens: {result.total_output_tokens:,}")


if __name__ == "__main__":
    main()
