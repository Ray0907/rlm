# RLM - Recursive Language Models

Python implementation of Recursive Language Models for processing arbitrarily long contexts.

## What is RLM?

RLM is an inference strategy that lets language models recursively call themselves to handle unlimited input context length. Instead of feeding long text directly to the LLM (causing "context rot"), RLM stores context as a variable in a Python REPL environment.

```
Traditional LLM:  Context -> LLM -> Answer  (quality degrades with length)

RLM:              LLM <-> REPL Environment
                          |-- context (variable)
                          |-- llm_query() (recursive calls)
                          |-- exec() (run code)
                          v
                      FINAL(answer)
```

## Key Concepts

- **Context as Environment**: Long text stored in REPL variable, not in LLM prompt
- **Code as Tool**: LLM writes Python to explore context (`context[:1000]`, `re.findall()`, etc.)
- **Recursive Decomposition**: Complex problems split via `llm_query()` sub-calls
- **Self-Correction**: LLM sees execution errors and fixes its own code
- **Output Offloading**: Large outputs auto-stored as variables to prevent context rot

## Installation

```bash
# Clone the repository
git clone https://github.com/Ray0907/rlm.git
cd rlm

# Install with uv
uv sync

# Set up API key
cp .env.example .env
# Edit .env with your API key (OPENAI_API_KEY or ANTHROPIC_API_KEY)
```

## Usage

```python
import rlm

result = rlm.run(
    query="What is the total salary of all Engineering employees?",
    context="""
    Employee Records:
    - Alice: Engineering, $75,000
    - Bob: Marketing, $65,000
    - Charlie: Engineering, $85,000
    """,
    model="anthropic/claude-sonnet-4-5-20250929",  # or "gpt-4o-mini"
)

print(result.answer)      # $160,000
print(result.iterations)  # Number of execute-observe cycles
```

## How It Works

1. **Query sent to LLM** with system prompt explaining REPL environment
2. **LLM writes code** in ```repl blocks to explore context
3. **Code executed**, output returned to LLM (large outputs auto-offloaded)
4. **LLM iterates** until it finds the answer
5. **FINAL(answer)** signals completion

## Output Offloading (Context Rot Prevention)

RLM automatically offloads large outputs to variables to prevent context rot:

```
Without Offloading:
  Iteration 1: output (2000 chars) -> history grows
  Iteration 2: output (1500 chars) -> history grows more
  Iteration N: history bloated     -> quality degrades

With Offloading:
  Iteration 1: output -> stored as _result_0, preview (200 chars) in history
  Iteration 2: output -> stored as _result_1, preview in history
  Iteration N: history stays small -> quality maintained
```

**Benchmark Results** (50 employee records, Haiku 3.5):

| Metric | With Offload | Without Offload | Improvement |
|--------|-------------|-----------------|-------------|
| Input Tokens | 8,980 | 10,748 | **-16.4%** |
| History Size | 2,154 chars | 5,636 chars | **-61.8%** |
| Avg Output | 718 chars | 1,878 chars | **-62%** |

Data still accessible via `print(_result_0)`. Run benchmark: `uv run python examples/offload_benchmark.py`

### REPL Functions

The LLM has access to these memory management functions:

```python
# Check what's stored in memory
memory_status()  # -> {'context': 10000, '_result_0': 2000, ...}

# Clean up variables no longer needed
forget('_result_0', 'temp_var')  # -> "Cleared: _result_0, temp_var"
```

## RLM vs Traditional Agents

| Aspect                | Traditional Agent         | RLM                            |
| --------------------- | ------------------------- | ------------------------------ |
| Problem Decomposition | Human predefines workflow | Model decides how to decompose |
| Tool Usage            | Fixed tool set            | Code as universal tool         |
| Control Flow          | Human-designed (ReAct)    | Model-driven iteration         |
| Flexibility           | Limited to designed tools | Unlimited (any valid Python)   |

## Supported Models

Any model supported by [LiteLLM](https://docs.litellm.ai/docs/providers):

```bash
# Anthropic (Recommended)
LITELLM_MODEL=anthropic/claude-sonnet-4-5-20250929 uv run python examples/simple_qa.py

# OpenAI
LITELLM_MODEL=gpt-4o-mini uv run python examples/simple_qa.py

# Local (Ollama)
LITELLM_MODEL=ollama/llama2 uv run python examples/simple_qa.py
```

## Examples

```bash
# Simple Q&A
uv run python examples/simple_qa.py

# Long document processing
uv run python examples/long_document.py

# Output offloading benchmark (compare with/without)
uv run python examples/offload_benchmark.py
```

## Benchmark Results

### Weak Model + RLM vs Strong Model Direct

Comparing **Haiku 4.5 + RLM** vs **Opus 4.5 Direct** on employee salary aggregation task.

| Records | Context      | Expected   | RLM Answer | Direct Answer | RLM Cost | Direct Cost | RLM Time | Direct Time | Winner |
| ------- | ------------ | ---------- | ---------- | ------------- | -------- | ----------- | -------- | ----------- | ------ |
| 100     | 24,390 chars | $1,105,000 | $1,105,000 | $145,000      | $0.0437  | $0.0410     | 43.9s    | 3.3s        | RLM    |
| 200     | 48,835 chars | $2,822,000 | $2,822,000 | $145,000      | $0.0469  | $0.0791     | 53.8s    | 3.8s        | RLM    |

**Key Findings:**

- RLM with weak model (Haiku 4.5) achieves **correct answers** while strong model (Opus 4.5) fails on longer contexts
- Cost is comparable or lower with RLM
- Trade-off: RLM takes longer due to multiple iterations

## API Reference

### `rlm.run()`

```python
def run(
    query: str,                        # Question to answer
    context: str | list,               # Data to process (stored in REPL, not sent to LLM)
    model: str = "gpt-4o-mini",
    max_iterations: int = 20,
    max_output_chars: int = 30000,     # Max chars before truncation
    verbose: bool = False,
) -> RLMResult
```

### `rlm.RLM()` (Advanced)

```python
rlm_instance = rlm.RLM(
    model: str = "gpt-4o-mini",
    sub_model: str = None,             # Model for llm_query() calls
    max_iterations: int = 20,
    max_output_chars: int = 30000,     # Max chars before truncation
    verbose: bool = False,
)
```

### `RLMResult`

```python
@dataclass
class RLMResult:
    answer: str              # Final answer
    iterations: int          # Number of iterations
    total_input_tokens: int  # Total input tokens used
    total_output_tokens: int # Total output tokens used
    code_executions: int     # Number of code blocks executed
    sub_calls: int           # Number of llm_query() calls
    history: list[dict]      # Full execution history
```

## References

1. **Paper**: [Recursive Language Models](https://arxiv.org/pdf/2512.24601v1) - arXiv:2512.24601
2. **Blog**: [RLM: Scalable LLM Inference](https://alexzhang13.github.io/blog/2025/rlm/) - Author's implementation notes

## License

MIT
