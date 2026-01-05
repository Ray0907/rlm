"""
Simple Q&A Example for RLM

Demonstrates basic RLM usage with a simple context and query.
The LLM will write code to extract information from the context.
"""

import os
import sys

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

import rlm

# Load environment variables
load_dotenv()


def main():
    # Simple context with structured information
    context = """
    Employee Records:
    - Alice Johnson: Age 25, Department: Engineering, Salary: $75,000
    - Bob Smith: Age 30, Department: Marketing, Salary: $65,000
    - Charlie Brown: Age 35, Department: Engineering, Salary: $85,000
    - Diana Lee: Age 28, Department: Sales, Salary: $70,000
    - Eve Wilson: Age 32, Department: Engineering, Salary: $90,000
    """

    query = "What is the total salary of all Engineering department employees?"

    print("=" * 60)
    print("RLM Simple Q&A Example")
    print("=" * 60)
    print(f"\nQuery: {query}")
    print(f"\nContext length: {len(context)} chars")
    print("-" * 60)

    # Run RLM
    result = rlm.run(
        query=query,
        context=context,
        model=os.getenv("LITELLM_MODEL", "anthropic/claude-sonnet-4-5-20250929"),
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Answer: {result.answer}")
    print(f"Iterations: {result.iterations}")
    print(f"Code executions: {result.code_executions}")
    print(f"Sub-calls (llm_query): {result.sub_calls}")
    print(f"Total input tokens: {result.total_input_tokens}")
    print(f"Total output tokens: {result.total_output_tokens}")


if __name__ == "__main__":
    main()
