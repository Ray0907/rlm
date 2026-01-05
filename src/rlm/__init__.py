"""
RLM - Recursive Language Models

A Python implementation of the RLM paradigm for processing arbitrarily long contexts.

Context is part of the environment, not input to the LLM.
The LLM writes code to explore and process context stored in a REPL.

Usage:
    import rlm

    result = rlm.run(
        query="What is the sum of all ages?",
        context="Alice is 25. Bob is 30. Charlie is 35.",
        model="gpt-4o-mini"
    )
    print(result.answer)  # 90
"""

from .core import RLM, RLMResult, run
from .repl import ReplEnvironment

__version__ = "0.1.0"

__all__ = [
    "RLM",
    "RLMResult",
    "ReplEnvironment",
    "run",
]
