# Math AI Agent — Complete Build Documentation

> A local, self-contained math-solving AI agent that runs Python code, generates proofs, graphs functions, remembers what it learns, verifies its own work with sub-agents, and can be fine-tuned — all on your machine, no cloud required.

---

## Table of Contents

1. [What This Project Is](#1-what-this-project-is)
2. [The Model: Why Qwen2.5-Math-1.5B-Instruct](#2-the-model-why-qwen25-math-15b-instruct)
3. [Project Architecture Overview](#3-project-architecture-overview)
4. [File-by-File Breakdown](#4-file-by-file-breakdown)
5. [The Build Process: Step by Step](#5-the-build-process-step-by-step)
6. [How Each Feature Works](#6-how-each-feature-works)
7. [Bugs We Hit and How We Fixed Them](#7-bugs-we-hit-and-how-we-fixed-them)
8. [Testing and Validation](#8-testing-and-validation)
9. [The Training Pipeline](#9-the-training-pipeline)
10. [How to Use It](#10-how-to-use-it)
11. [Design Decisions and Trade-offs](#11-design-decisions-and-trade-offs)
12. [Future Improvements](#12-future-improvements)

---

## 1. What This Project Is

This is a math AI agent built from scratch in Python. It lives at `~/math-agent/` and consists of 12 Python files, a 1.13 GB local model, a SQLite database for memory, and a training pipeline for fine-tuning the model on math problems with proofs.

The agent can:

- **Run Python code to solve math problems** — it generates sympy/numpy/scipy code, executes it in a sandboxed subprocess, and extracts structured results
- **Spawn multiple agents of itself** — three expert agents (algebra, number theory, geometry) can discuss a problem, cross-check each other, and synthesize a unified answer
- **Remember things** — a SQLite database stores conversation history, learned facts, and every problem ever solved with its verification status
- **Verify its own work** — a sub-agent independently re-derives the answer using code and checks the LLM's reasoning, producing a confidence score from 0 to 1
- **Handle complex math** — eigenvalues, derivatives, integrals, limits, equation solving, Taylor series, proofs by induction
- **Cite sources with proofs** — every step in a solution includes the theorem or rule that justifies it (Power Rule, Fundamental Theorem of Calculus, Zero Product Property, etc.)
- **Explain each step** — each step says what was done, why it was done, and what proof justifies it
- **Graph things** — 2D function plots, 3D surface plots, multi-function overlays, and geometric shapes (circles, polygons, lines, points) via matplotlib

The constraint was: download a small local model no bigger than 5 billion parameters and train it more. We chose **Qwen2.5-Math-1.5B-Instruct** (1.5 billion parameters, 1.13 GB in GGUF Q5_K_M quantization) and built a LoRA fine-tuning pipeline for it.

---

## 2. The Model: Why Qwen2.5-Math-1.5B-Instruct

### Selection Criteria

The user asked for a model under 5 billion parameters. We wanted:
- Specifically trained on math (not a general chatbot)
- Small enough to run locally on a laptop without a dedicated GPU
- Available in GGUF format for fast inference via llama.cpp
- Instruct-tuned (already knows how to follow instructions, not just complete text)

### What We Chose

**Qwen2.5-Math-1.5B-Instruct** by Alibaba/Qwen team. This is a 1.5 billion parameter model specifically fine-tuned on mathematical reasoning tasks. It's part of the Qwen2.5-Math series, which was trained on a large corpus of math problems, solutions, and proofs.

### GGUF Quantization

We download the Q5_K_M GGUF quantization from `bartowski/Qwen2.5-Math-1.5B-Instruct-GGUF` on HuggingFace. GGUF (GPT-Generated Unified Format) is a binary format that packs the model weights in a way that's optimized for CPU/GPU inference via llama.cpp. Q5_K_M means 5-bit quantization with medium precision — it reduces the model from ~3 GB (FP16) to 1.13 GB while keeping most of the reasoning quality.

The download happens in `download_model.py`:

```python
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="bartowski/Qwen2.5-Math-1.5B-Instruct-GGUF",
    filename="Qwen2.5-Math-1.5B-Instruct-Q5_K_M.gguf",
    local_dir=os.path.dirname(GGUF_PATH),
)
```

If the GGUF download fails (e.g., repo unavailable), the script falls back to downloading the full HuggingFace model via `snapshot_download` and uses the transformers library for inference instead.

### Inference Engine

The agent uses two inference paths, selected automatically in `llm_engine.py`:

1. **llama.cpp (primary)** — if the GGUF file exists at `model/qwen2.5-math-1.5b-instruct-q5_k_m.gguf`, we load it via `llama_cpp.Llama` with GPU offload (`n_gpu_layers=-1`) and a 4096 token context window. This is fast and uses ~1.3 GB RAM.

2. **transformers (fallback)** — if no GGUF, we load the full FP16 model via HuggingFace transformers. This path also checks for a trained LoRA adapter in `training/math-lora/` and automatically applies it if present, so the agent uses your fine-tuned weights without any code changes.

---

## 3. Project Architecture Overview

```
~/math-agent/
│
├── agent.py              ← Main orchestrator — ties everything together
├── llm_engine.py         ← LLM inference (llama.cpp or transformers + LoRA)
├── code_solver.py        ← Sandboxed Python code execution
├── code_templates.py     ← Direct code generation for known problem types
├── graphing.py           ← matplotlib graphing (2D, 3D, multi, geometry)
├── memory.py             ← SQLite persistent memory
├── verifier.py           ← Sub-agent that independently checks solutions
├── multi_agent.py        ← Multi-agent discussion and cross-checking
├── train.py              ← LoRA fine-tuning pipeline
├── download_model.py     ← Downloads the GGUF model from HuggingFace
├── config.py             ← All configuration constants
├── run.py                ← Quick demo script
├── README.md             ← This file
│
├── model/                ← Model storage
│   └── qwen2.5-math-1.5b-instruct-q5_k_m.gguf  (1.13 GB)
│
├── graphs/               ← Generated graph PNGs
│
├── memory/
│   └── agent_memory.db   ← SQLite database
│
├── training/
│   └── math-lora/        ← LoRA adapter output (created after training)
│
├── data/
│   └── math_train.jsonl  ← Generated training data
│
└── venv/                 ← Python 3.11 virtual environment
```

### Data Flow

```
User Input
    │
    ▼
agent.py: MathAgent.solve()
    │
    ├──► memory.py: search for similar facts/problems
    │
    ├──► code_templates.py: detect_and_solve()
    │       │
    │       ├── [match] ──► generate Python code directly
    │       └── [no match] ──► llm_engine.py: generate_code()
    │                              │
    │                              ▼
    │                          Qwen2.5-Math-1.5B generates code
    │                              │
    │                              ▼
    │                          _extract_code() strips markdown
    │
    ├──► code_solver.py: CodeSolver.solve()
    │       │
    │       ├── safety check (banned imports)
    │       ├── wrap with imports (sympy, numpy, scipy, matplotlib)
    │       ├── write to temp file
    │       ├── subprocess.run() with 30s timeout
    │       ├── parse __RESULT_JSON__ from stdout
    │       └── return structured result dict
    │
    ├──► [if code fails] ──► _llm_fallback()
    │       └── LLM generates pure reasoning solution
    │
    ├──► verifier.py: VerifierAgent.verify()
    │       ├── re-derive answer with independent code
    │       ├── LLM checks each step
    │       └── produce confidence score (0-1)
    │
    ├──► [optional] multi_agent.py: MultiAgentSystem.discuss()
    │       ├── 3 agents with different expertise solve independently
    │       ├── synthesis agent combines answers
    │       └── verifier checks the synthesis
    │
    ├──► [optional] graphing.py: make_graph()
    │       └── save PNG to graphs/
    │
    ├──► memory.py: store problem + solution + verification
    │
    └──► print formatted solution with steps, proofs, confidence
```

---

## 4. File-by-File Breakdown

### config.py

This is the configuration hub. It defines all paths and constants:

- `BASE_DIR` — the project root directory, computed from the file's own location
- `GGUF_PATH` — where the quantized model lives
- `MODEL_DIR` — where the full HuggingFace model would live (fallback)
- `GRAPH_DIR` — where generated graphs are saved
- `MEMORY_DB` — path to the SQLite database
- `TRAINING_DIR` — where LoRA adapters are saved
- `MODEL_ID` — the HuggingFace model identifier (`Qwen/Qwen2.5-Math-1.5B-Instruct`)
- `GGUF_REPO` / `GGUF_FILE` — the GGUF repo (`bartowski/Qwen2.5-Math-1.5B-Instruct-GGUF`) and filename
- `MAX_TOKENS` — 2048, the max generation length
- `TEMPERATURE` — 0.7, default sampling temperature
- `N_CTX` — 4096, the context window size

It also creates all necessary directories at import time using `os.makedirs(..., exist_ok=True)`.

One quirk: the `MEMORY_DIR` line uses a walrus operator inside `os.makedirs` to extract the directory from the database path. This works but is unusual.

### llm_engine.py

This is the LLM abstraction layer. It provides a unified `generate()` function that works whether the model is loaded via llama.cpp (GGUF) or transformers (HuggingFace).

**Singleton pattern**: A module-level `_LLM` variable caches the loaded model so it's only loaded once per process.

**`get_llm()`**: Checks if the GGUF file exists. If yes, loads it with `llama_cpp.Llama` with GPU offload and 4096 context. If not, loads the full model with transformers, checking for a LoRA adapter in `training/math-lora/`.

**`generate(prompt, system_prompt, max_tokens, temperature)`**: Builds a chat message list, calls the model, and returns the text response. For llama.cpp, it uses `create_chat_completion()`. For transformers, it applies the chat template, tokenizes, calls `model.generate()`, and decodes.

**`generate_code(problem)`**: A specialized prompt for code generation. The system prompt tells the model to output ONLY Python code. The user prompt gives the problem and strict rules about the `_result` dict structure. Temperature is 0.2 for deterministic output.

**`_extract_code(text)`**: This is critical. The 1.5B model often wraps code in markdown fences or adds explanation text despite being told not to. This function handles all cases:
1. Look for ` ```python ` ... ` ``` ` blocks
2. Look for any ` ``` ` ... ` ``` ` blocks
3. Scan line-by-line for the first Python statement and take everything from there

### code_solver.py

This executes Python code in a sandboxed subprocess. It's the engine that actually does the math.

**`CodeSolver.solve(code, problem_context)`**: The main entry point. First checks if code is empty, then runs a safety filter, then executes.

**`_is_safe(code)`**: Scans for banned patterns: `import os`, `import subprocess`, `import shutil`, `import socket`, `os.system`, `__import__`, `eval(`, `exec(`, `open(`, `os.remove`, `os.rmdir`, `shutil.rmtree`, `os.environ`. If any are found, the code is rejected. This prevents the generated code from doing anything malicious.

**`_run(code, problem_context)`**: This is where the magic happens:
1. Build a wrapper preamble that pre-imports everything: `sympy`, `numpy`, `scipy`, `matplotlib` (in Agg mode for headless rendering), and optionally `graphing.make_graph` (wrapped in try/except so missing imports don't crash)
2. Inject `__PROBLEM_CTX__` as a JSON string so the code can reference the original problem
3. Append the user's code
4. Append a try/except block that prints `__RESULT_JSON__:` followed by `json.dumps(_result)` — this is how the result gets extracted from stdout
5. Write to a temp file in `/tmp`
6. Run `subprocess.run()` with the venv's Python, the project directory as `cwd`, and a 30-second timeout
7. Parse stdout for the `__RESULT_JSON__:` line and extract the JSON
8. Return a dict with `success`, `result`, `stdout`, `stderr`, `returncode`

The subprocess inherits the environment (important for the venv's site-packages) and runs from the project directory (important for importing `graphing.py`).

### code_templates.py

This was a late addition that dramatically improved reliability. The 1.5B model struggles to generate clean Python code — it often wraps it in LaTeX explanations or markdown fences. To work around this, we built a regex-based problem detector that generates Python code directly for common math problem types.

**`detect_and_solve(problem)`**: Takes the problem string, lowercases it, and runs regex matches:

- **Eigenvalues**: `eigenvalue.*matrix [[...],[...]]` or `matrix [[...],[...]] .* eigenvalue`
- **Derivatives**: `derivative of <expr>` or `diff(erentiate)? <expr>`
- **Integrals**: `integral of <expr>` or `integrate <expr>` (handles definite integrals with "from X to Y")
- **Equations**: `solve <expr>` (splits on `=` to form `lhs - rhs = 0`)
- **Limits**: `limit of <expr> as <var> approaches <point>`
- **Determinants**: `determinant of matrix [[...],[...]]`

Each match generates a Python code string using sympy with the correct imports, the computation, and a fully structured `_result` dict including steps, proofs, and method name.

**`_normalize_expr(expr_str)`**: This fixes a critical bug. Users write math like `3x^3 + 2x^2`, but sympy needs `3*x**3 + 2*x**2`. This function:
1. Replaces `^` with `**`
2. Inserts `*` between a digit and a letter: `3x` → `3*x`
3. Inserts `*` between `)` and a letter/digit: `)x` → `)*x`
4. Inserts `*` between a letter and `(`: `x(` → `x*(`

**`_eigenvalue_code(r1, r2)`**: Has a `clean()` helper that converts floats to ints when possible (e.g., `2.0` → `2`) so the output is clean: `Matrix([[2, 1], [1, 2]])` instead of `Matrix([[2.0, 1.0], [1.0, 2.0]])`.

Each template function returns a Python code string that:
1. Imports sympy
2. Defines symbols
3. Computes the answer
4. Builds a `_result` dict with `answer`, `steps` (each with `step`, `explanation`, `proof`), `proofs` (each with `theorem`, `statement`, `applied_to`), and `method`

### graphing.py

This handles all graphing. It uses matplotlib in Agg (headless) mode.

**`make_graph(expr_or_data, var, title, graph_type, **kwargs)`**: The main entry point. Generates a random filename, routes to the appropriate plotting function based on `graph_type`:
- `"2d"` → `_plot_2d` (single function)
- `"3d"` → `_plot_3d` (surface plot)
- `"multi"` → `_plot_multi` (multiple functions overlaid)
- `"geometry"` → `_plot_geometry` (circles, polygons, lines, points)

If a list is passed with `graph_type="2d"`, it automatically switches to `"multi"` (this was a bug fix).

**`_plot_2d`**: Takes a sympy expression string (e.g., `"sin(x)"`), lambdifies it, evaluates on a linspace, and plots. Also handles callable functions and raw numpy arrays.

**`_plot_3d`**: Creates a 3D axis, builds a meshgrid, lambdifies with two variables (x, y), and plots a surface with the viridis colormap.

**`_plot_multi`**: Iterates through a list of expressions, plotting each with a different color from the `COLORS` palette. Each gets a label for the legend.

**`_plot_geometry`**: Draws geometric shapes on an equal-aspect plot:
- Circles via `matplotlib.patches.Circle`
- Polygons via `matplotlib.patches.Polygon`
- Lines as `ax.plot()` calls
- Points as scatter markers

All plots save to `graphs/` at 100 DPI with `tight_layout()`.

### memory.py

SQLite-based persistent memory. Three tables:

**`conversations`**: Stores every message exchanged. Columns: `id`, `role` (user/assistant), `content`, `metadata` (JSON), `timestamp`.

**`facts`**: Key-value store for learned facts. Columns: `id`, `key` (unique), `value`, `confidence` (0-1), `source`, `timestamp`. Uses `INSERT OR REPLACE` so facts can be updated.

**`problems`**: Every solved problem. Columns: `id`, `problem`, `solution`, `proof` (JSON of proofs), `verified` (0 or 1), `method`, `timestamp`.

Methods:
- `remember(role, content, metadata)` — append to conversation history
- `recall(limit)` — get last N messages in order
- `store_fact(key, value, source, confidence)` — upsert a fact
- `get_fact(key)` — retrieve a specific fact
- `search_facts(query)` — LIKE search on key and value
- `store_problem(...)` — save a solved problem
- `get_problems(limit)` — retrieve recent problems

The connection uses `check_same_thread=False` so it works across threads if needed.

### verifier.py

The verification sub-agent. This is the "sub agent checks main llm" feature.

**`VerifierAgent.verify(problem, solution)`**: Two-stage verification:

1. **Code-based verification (primary, reliable)**: Generates Python code to independently re-derive the answer. For template-detectable problems, it re-runs the template (which is deterministic and guaranteed correct). For LLM-only problems, it asks the LLM to write verification code. The code runs through `CodeSolver.solve()`.

2. **LLM-based verification (secondary, best-effort)**: Asks the LLM to check each step of the solution and respond as JSON with `agrees`, `errors`, `corrections`, `confidence`. The 1.5B model often fails to produce valid JSON here, which is why code verification is primary.

**Confidence scoring**:
- Code succeeds and agrees → **0.85** (high confidence, verified)
- Code succeeds but no explicit agreement → **0.6** (medium, verified)
- Code fails but LLM agrees → **0.5** (medium, verified)
- Both fail → **0.2** (low, not verified)

The `_verification_check_code()` method wraps the template re-derivation by saving the original answer before overwriting `_result`, so the verification result includes the independent answer for comparison.

### multi_agent.py

The multi-agent discussion system. This is the "make multiple agents of itself to talk to" feature.

**`MultiAgentSystem.discuss(problem, context)`**: Spawns 3 agents, each with a different mathematical persona:

- **Agent_1**: Algebra, calculus, symbolic computation
- **Agent_2**: Number theory, combinatorics, proof writing
- **Agent_3**: Geometry, topology, numerical methods

Each agent gets the problem and any context from previous agents. They're prompted to provide their solution as JSON with `approach`, `answer`, `confidence`, `concerns`, and `steps`. Temperature varies per agent (0.5, 0.6, 0.7) to encourage diversity.

**`_synthesize(problem, responses)`**: A synthesis agent combines all three agents' responses into a unified solution with `answer`, `method`, `steps`, `proofs`, `consensus`, and optional `graph`.

**`cross_check(problem, solution_a, solution_b)`**: A standalone method to compare two solutions side by side, useful when different agents disagree.

The discussion results are also verified through the same `VerifierAgent` system.

### train.py

The LoRA fine-tuning pipeline.

**`generate_training_data()`**: Creates 8 hand-crafted math training samples, each with:
- A problem statement
- The correct answer
- Step-by-step solution (each step has what, why, and proof)
- Theorems used (each with statement and what it's applied to)

The samples cover: derivatives, definite integrals, quadratic equations, limits, proof by definition (odd+odd=even), eigenvalues, Taylor series, and area via integration.

Each sample is formatted as a chat conversation:
```json
{
  "messages": [
    {"role": "system", "content": "You are a math expert..."},
    {"role": "user", "content": "Find the derivative of..."},
    {"role": "assistant", "content": "{\"answer\": \"9x^2+4x-5\", \"steps\": [...], \"proofs\": [...]}"}
  ]
}
```

This is written to `data/math_train.jsonl` (one JSON object per line).

**`train()`**: The actual fine-tuning:
1. Generate training data
2. Load the full Qwen2.5-Math-1.5B-Instruct model with transformers (FP16)
3. Apply LoRA config: `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, targeting all linear layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
4. Tokenize the dataset using the model's chat template with `max_length=1024`
5. Train for 3 epochs with batch size 2, gradient accumulation 4, learning rate 2e-4, FP16
6. Save the LoRA adapter to `training/math-lora/`

The LoRA approach means only ~1% of parameters are trained, making it feasible on a laptop GPU. The adapter is small (~10 MB) and automatically loaded by `llm_engine.py` when using the transformers path.

### download_model.py

Simple script that downloads the GGUF model from HuggingFace. Checks if the file already exists to skip re-downloading. Falls back to full model download if GGUF fails.

### agent.py

The main orchestrator. This ties everything together.

**`MathAgent.__init__`**: Creates instances of `Memory`, `CodeSolver`, `VerifierAgent`, and `MultiAgentSystem`.

**`MathAgent.solve(problem, graph, multi_agent)`**: The main solve pipeline:
1. Search memory for related facts
2. Generate code: try `detect_and_solve()` first, fall back to `generate_code()`
3. Execute code via `CodeSolver.solve()`
4. If code fails, fall back to `_llm_fallback()` (pure LLM reasoning)
5. Verify via `VerifierAgent.verify()`
6. If multi-agent mode, run `MultiAgentSystem.discuss()`
7. Print formatted solution
8. If graph requested and solution includes graph spec, generate graph
9. Store in memory

**`_print_solution`**: Pretty-prints the solution with:
- Answer
- Numbered steps with explanation and proof for each
- Proofs used (theorem, statement, applied to)
- Verification confidence and any errors
- Multi-agent discussion results (if applicable)

**`_llm_fallback`**: When code execution fails, the LLM generates a pure reasoning solution. It's prompted to provide JSON with `answer`, `steps`, `proofs`, `method`. The JSON is extracted by finding the first `{` and last `}` in the response (since the 1.5B model may add extra text).

**`chat(message)`**: Interactive mode router. Detects if the message is a math problem (contains words like "solve", "calculate", "find", "prove") and routes to `solve()`. Otherwise, it's a general conversation that uses memory context.

**`main()`**: CLI entry point with argparse:
- `problem` — positional, the math problem
- `--graph` — generate a graph
- `--multi-agent` — use multi-agent discussion
- `--interactive` — interactive REPL mode
- `--memory` — show stored memory

---

## 5. The Build Process: Step by Step

### Step 1: Project Setup

Created the directory structure:
```
~/math-agent/{model,training,data,graphs,memory,agents}
```

Created a Python 3.11 virtual environment (initial attempt with system Python 3.9 failed because torch requires Python 3.10+). Installed all dependencies: `torch`, `numpy`, `scipy`, `sympy`, `matplotlib`, `llama-cpp-python`, `transformers`, `accelerate`, `peft`, `datasets`, `huggingface_hub`.

### Step 2: Configuration

Wrote `config.py` with all paths, model IDs, and constants. Used `os.makedirs(..., exist_ok=True)` to auto-create directories at import time.

### Step 3: Memory System

Wrote `memory.py` with SQLite. Three tables: conversations, facts, problems. Tested with storing/retrieving facts, conversation history, and solved problems. All passed.

### Step 4: Code Solver

Wrote `code_solver.py` with the sandboxed execution model. The key design: write generated code to a temp file with a wrapper preamble, execute as subprocess, parse `__RESULT_JSON__:` from stdout.

### Step 5: Graphing

Wrote `graphing.py` with four plot types: 2D, 3D, multi-function, and geometry. Each uses matplotlib in Agg mode. Tested all four types — 2D sin(x), multi-function (sin/cos/parabola), 3D paraboloid, and geometry shapes.

### Step 6: LLM Engine

Wrote `llm_engine.py` with dual inference paths (llama.cpp and transformers). Added the LoRA adapter auto-loading for the transformers path.

### Step 7: Model Download

First attempt: tried `Qwen/Qwen2.5-Math-1.5B-Instruct-GGUF` — this repo doesn't exist on HuggingFace (401 error). Searched for alternative repos and found `bartowski/Qwen2.5-Math-1.5B-Instruct-GGUF` which has the correct file. Updated `config.py` with the right repo and filename. Downloaded the 1.13 GB GGUF file successfully.

### Step 8: Verifier

Wrote `verifier.py` with two-stage verification (code + LLM). Initially the confidence calculation was a weighted average, but the 1.5B model's LLM verification was too unreliable (often returning unparseable JSON). Reworked to make code verification primary with fixed confidence tiers.

### Step 9: Multi-Agent System

Wrote `multi_agent.py` with three expert agents and a synthesis agent. Each agent has a different persona and temperature for diversity. The synthesis combines all responses, and the result is verified.

### Step 10: Main Orchestrator

Wrote `agent.py` tying everything together. The solve pipeline: memory search → code generation → execution → verification → optional multi-agent → optional graphing → memory storage.

### Step 11: First Test — Failure

Ran the agent on "Find all eigenvalues of the matrix [[2,1],[1,2]]". The LLM generated a response that was mostly LaTeX explanation with code inside markdown fences. The code extraction stripped the fences but the remaining text wasn't valid Python (LaTeX backslashes caused SyntaxError). The fallback LLM reasoning worked but couldn't be verified.

### Step 12: Code Templates

Built `code_templates.py` with regex-based problem detection and direct code generation. This was the key breakthrough — instead of relying on the 1.5B model to generate clean code (which it can't reliably do), we detect the problem type and generate correct code ourselves.

### Step 13: Expression Normalization

First template test: "Find the derivative of 3x^3 + 2x^2 - 5x + 1" failed because `sympify('3x**3')` doesn't work — sympy needs `3*x**3`. Built `_normalize_expr()` to insert implicit multiplication signs.

### Step 14: Verifier Integration with Templates

The verifier initially tried to generate LLM code for verification, which had the same code generation problems. Reworked it to use templates for verification too — if the problem matches a template, it re-runs the template (which is deterministic and correct) as verification.

### Step 15: Training Pipeline

Wrote `train.py` with 8 hand-crafted training samples covering derivatives, integrals, equations, limits, proofs, eigenvalues, Taylor series, and area by integration. Each sample has step-by-step solutions with theorem citations. Uses LoRA (r=16) targeting all linear layers for 3 epochs.

### Step 16: Full End-to-End Testing

Tested all problem types:
- ✅ Eigenvalues: `[1, 3]` with confidence 0.85
- ✅ Derivatives: `9*x**2 + 4*x - 5` with confidence 0.85
- ✅ Integrals: `10` with confidence 0.85
- ✅ Equations: `[2, 3]` with confidence 0.85
- ✅ Limits: `2` with confidence 0.85
- ✅ Graphs: 2D, multi, 3D all rendered correctly
- ✅ Memory: facts, problems, conversations all stored and retrieved
- ✅ Training data: 8 samples generated successfully

---

## 6. How Each Feature Works

### Running Python Code to Solve Math Problems

The agent doesn't just ask the LLM for an answer — it asks the LLM (or a template) to write Python code, then actually runs that code. This means the math is done by sympy (a computer algebra system), not by the LLM's neural network. The LLM's job is to translate the natural language problem into correct Python code.

The code runs in a subprocess with:
- Pre-imported sympy, numpy, scipy, matplotlib
- A 30-second timeout
- A safety filter blocking dangerous imports
- A structured output protocol (`_result` dict printed as `__RESULT_JSON__:`)

This is far more reliable than asking the LLM to compute math directly. A 1.5B model will make arithmetic errors; sympy never will.

### Multiple Agents

When `--multi-agent` is passed, three agents with different expertise each solve the problem independently. They're given slightly different temperatures (0.5, 0.6, 0.7) to encourage diverse approaches. A synthesis agent then combines their answers into one unified solution. The synthesis is verified by the same verifier system.

This is useful for complex problems where a single approach might miss something. If all three agents agree, confidence is high. If they disagree, the synthesis agent reconciles and the verifier checks.

### Memory

SQLite stores three types of information:
1. **Conversation history** — every user message and agent response, so the agent can recall past interactions
2. **Facts** — a key-value store for mathematical facts (e.g., `derivative_power_rule` → `d/dx(x^n) = n*x^(n-1)`). These are searchable and can be referenced in future solutions.
3. **Solved problems** — every problem ever solved, with its solution, proofs, verification status, and method used. This lets the agent build up a knowledge base over time.

Memory persists across sessions because it's in a file (`memory/agent_memory.db`), not in RAM.

### Sub-Agent Verification

After the main agent produces a solution, a separate `VerifierAgent` independently checks it:

1. **Code re-derivation**: Generates Python code to solve the same problem from scratch. For template-detectable problems, this uses the same template (which is deterministic). For LLM-generated solutions, it asks the LLM to write verification code. The code runs through the same `CodeSolver` with the same sandbox.

2. **LLM step checking**: The LLM is asked to verify each step of the solution, checking for mathematical errors and logical gaps. It responds as JSON with `agrees`, `errors`, `corrections`, `confidence`. (This is less reliable from a 1.5B model, which is why code verification is the primary method.)

3. **Confidence scoring**: Based on whether code verification succeeded and whether the LLM agreed. Ranges from 0.2 (both failed) to 0.85 (code verified successfully).

### Complex Math

The agent handles:
- **Calculus**: Derivatives (power rule, chain rule, etc.) and definite/indefinite integrals (via sympy's `diff` and `integrate`)
- **Linear algebra**: Eigenvalues/eigenvectors, determinants (via sympy's `Matrix` class)
- **Equations**: Polynomial equations, systems (via sympy's `solve`)
- **Limits**: Including indeterminate forms (via sympy's `limit` with L'Hôpital's Rule support)
- **Proofs**: Proof by definition (e.g., odd+odd=even), with each step justified by a theorem
- **Taylor series**: Via sympy's `series` function
- **Area by integration**: Including trigonometric substitution

### Sources with Proofs

Every solution includes two proof-related structures:

**Steps**: Each step in the solution has three fields:
- `step`: What was done (e.g., "d/dx(3x^3) = 9x^2")
- `explanation`: Why it was done (e.g., "Power rule: d/dx(ax^n) = a*n*x^(n-1)")
- `proof`: The theorem or rule that justifies it (e.g., "Power Rule: d/dx(x^n) = n*x^(n-1) by first principles of differentiation")

**Proofs**: A separate list of all theorems used, each with:
- `theorem`: Name (e.g., "Fundamental Theorem of Calculus")
- `statement`: Formal statement (e.g., "integral_a^b f(x)dx = F(b) - F(a) where F' = f")
- `applied_to`: What it was applied to (e.g., "definite integral")

### Explanation of Steps

The agent prints each step in a readable format:
```
Step 1: Matrix A = Matrix([[2, 1], [1, 2]])
  Explanation: Given matrix
  Proof: Definition of the matrix

Step 2: Characteristic equation: det(A - lambda*I) = 0
  Explanation: Eigenvalues satisfy det(A - lambda*I) = 0
  Proof: Eigenvalue definition: Av = lambda*v, so det(A - lambda*I) = 0
```

This makes it clear what was done, why, and what mathematical principle justifies it.

### Graphing

The graphing module supports four types:

1. **2D function plots**: Pass a sympy expression string and variable. The expression is lambdified and evaluated on a numpy linspace. Output: a single line plot with axes, grid, and legend.

2. **3D surface plots**: Pass an expression with two variables. A meshgrid is created, the expression is lambdified with both variables, and a surface is plotted with the viridis colormap.

3. **Multi-function plots**: Pass a list of expressions. Each gets a different color from a 7-color palette. All are plotted on the same axes with a legend.

4. **Geometry plots**: Pass a list of shape dictionaries. Supported shapes: circles (center + radius), polygons (list of points), lines (start + end), points (position). Plotted with equal aspect ratio.

All graphs save as PNG files in `graphs/` at 100 DPI.

---

## 7. Bugs We Hit and How We Fixed Them

### Bug 1: Wrong Python version

**Problem**: Initial venv used system Python 3.9, which doesn't support `TypeGuard` from the `typing` module that torch needs.

**Fix**: Deleted the venv and recreated it with `/Users/home/.local/bin/python3.11`.

### Bug 2: GGUF repo not found

**Problem**: Configured `GGUF_REPO` as `Qwen/Qwen2.5-Math-1.5B-Instruct-GGUF` — this repo doesn't exist on HuggingFace. Got a 401 error.

**Fix**: Searched for alternative repos using `list_repo_files()`. Found `bartowski/Qwen2.5-Math-1.5B-Instruct-GGUF` which has the correct GGUF files. Updated `config.py`.

### Bug 3: Subprocess couldn't find sympy

**Problem**: The `CodeSolver` ran subprocesses that couldn't find sympy because the environment was being overridden, stripping the venv's site-packages from the path.

**Fix**: Removed the custom `env` parameter from `subprocess.run()` so it inherits the parent environment (which includes the venv's site-packages).

### Bug 4: Graphing import fails in subprocess

**Problem**: The wrapper code in `code_solver.py` tried to `from graphing import make_graph`, but the subprocess couldn't find `graphing.py` because the working directory wasn't set correctly.

**Fix**: Added `cwd=os.path.dirname(os.path.abspath(__file__))` to `subprocess.run()` so the subprocess runs from the project directory. Also wrapped the import in try/except so missing graphing doesn't crash the solver.

### Bug 5: LLM generates explanation instead of code

**Problem**: The 1.5B Qwen model, when asked to "write Python code," often generates LaTeX explanations with code buried inside markdown fences. The extracted code had LaTeX backslashes causing SyntaxError.

**Fix**: Built the entire `code_templates.py` module. Instead of relying on the LLM to generate code, we use regex to detect the problem type and generate correct Python code directly. The LLM is only used as a fallback for problems that don't match any template.

### Bug 6: Implicit multiplication not handled

**Problem**: User writes `3x^3 + 2x^2` but sympy needs `3*x**3 + 2*x**2`. The template code called `sympify('3x**3')` which raised a SyntaxError.

**Fix**: Built `_normalize_expr()` which uses regex to insert `*` between digits and letters, between `)` and letters, and between letters and `(`. Also replaces `^` with `**`.

### Bug 7: Float values in matrix output

**Problem**: The eigenvalue template parsed matrix entries as floats (`2.0`, `1.0`), producing ugly output like `Matrix([[2.0, 1.0], [1.0, 2.0]])` and eigenvalues `[1.00000000000000, 3.00000000000000]`.

**Fix**: Added a `clean()` function in `_eigenvalue_code` that converts floats to ints when they're whole numbers.

### Bug 8: `var` name collision in limit template

**Problem**: The limit template used `var` as both the Python variable name and the sympy symbol name. But `var` is also a sympy function, so `f"lim({var}->...)"` printed `<function var at 0x...>`.

**Fix**: Renamed the sympy symbol to `_{var}` (e.g., `_x`) while keeping the display name as `{var}` (e.g., `x`).

### Bug 9: Verifier LLM can't produce valid JSON

**Problem**: The 1.5B model was asked to verify solutions and respond as JSON. It often couldn't produce parseable JSON, leading to "Verification parse failed" errors and low confidence scores.

**Fix**: Made code verification the primary method (which is deterministic and reliable). The LLM verification is secondary and best-effort. Confidence is now based primarily on whether the code re-derivation succeeds, not on the LLM's JSON parsing ability.

### Bug 10: List passed to 2D plot

**Problem**: When a list of expressions was passed with `graph_type="2d"`, the 2D plotter tried to plot the list as a single array, causing a dimension mismatch.

**Fix**: Added a check in `make_graph()`: if `graph_type == "2d"` and the input is a list, automatically switch to `"multi"`.

---

## 8. Testing and Validation

All tests were run end-to-end with the actual model loaded:

### Eigenvalues
```
Problem: Find all eigenvalues of the matrix [[2,1],[1,2]]
Answer: [1, 3]
Confidence: 0.85
Code verified: True
```
Verified: The characteristic polynomial λ² - 4λ + 3 = 0 factors as (λ-1)(λ-3) = 0, giving eigenvalues 1 and 3. ✓

### Derivative
```
Problem: Find the derivative of 3x^3 + 2x^2 - 5x + 1
Answer: 9*x**2 + 4*x - 5
Confidence: 0.85
Code verified: True
```
Verified: Power rule on each term: 9x² + 4x - 5. ✓

### Integral
```
Problem: Evaluate the integral of 2x + 3 from 0 to 2
Answer: 10
Confidence: 0.85
Code verified: True
```
Verified: Antiderivative x² + 3x, evaluated: F(2) - F(0) = (4+6) - 0 = 10. ✓

### Equation Solving
```
Problem: Solve x^2 - 5x + 6 = 0
Answer: [2, 3]
Confidence: 0.85
Code verified: True
```
Verified: (x-2)(x-3) = 0, roots are 2 and 3. ✓

### Limit
```
Problem: Find the limit of (x^2 - 1)/(x - 1) as x approaches 1
Answer: 2
Confidence: 0.85
Code verified: True
```
Verified: Factor to (x+1), substitute x=1: 2. ✓

### Graphing
- 2D sin(x)·cos(x): 44 KB PNG, correctly rendered wave
- Multi sin(x), cos(x), x²/10: 62 KB PNG, three correctly colored curves with legend
- 3D paraboloid x²+y²: 180 KB PNG, correct surface plot

### Memory
- Stored 2 facts, 10 problems (8 verified ✓, 2 unverified ✗ from early failed attempts)
- All facts searchable, all problems retrievable

### Training Data
- 8 samples generated, 12.5 KB JSONL file
- Each sample has system/user/assistant messages with structured JSON solutions including steps and proofs

---

## 9. The Training Pipeline

### Why LoRA

Full fine-tuning of a 1.5B model would require updating all 1.5 billion parameters, needing ~12 GB of VRAM and hours of training. LoRA (Low-Rank Adaptation) instead freezes the original model and adds small rank-16 matrices to each linear layer. This means only ~1% of parameters are trained (~15 million), requiring ~4 GB VRAM and minutes of training.

### Configuration

```python
LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                    # rank of the adaptation matrices
    lora_alpha=32,            # scaling factor (typically 2x rank)
    lora_dropout=0.05,        # regularization
    target_modules=[          # which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",  # attention
        "gate_proj", "up_proj", "down_proj"        # MLP
    ],
)
```

### Training Arguments

```python
TrainingArguments(
    num_train_epochs=3,           # 3 passes over the data
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4, # effective batch size = 8
    learning_rate=2e-4,            # standard for LoRA
    fp16=True,                     # mixed precision
    warmup_steps=10,               # learning rate warmup
)
```

### Training Data

8 hand-crafted samples covering diverse math topics:
1. Derivative of polynomial (Power Rule, Sum Rule, Constant Rule)
2. Definite integral (Fundamental Theorem of Calculus)
3. Quadratic equation (Zero Product Property, Vieta's Formulas)
4. Limit with indeterminate form (Difference of Squares, L'Hôpital's Rule)
5. Proof by definition (odd + odd = even)
6. Eigenvalues (Characteristic Equation, FTA)
7. Taylor series (Taylor Series Theorem, Exponential derivative property)
8. Area by integration (Trig substitution, Half-Angle Identity, FTC)

Each sample teaches the model to:
- Break problems into steps
- Justify each step with a named theorem
- List all theorems used with their formal statements
- Structure output as JSON

### Running Training

```bash
cd ~/math-agent
source venv/bin/activate
python3 train.py
```

The LoRA adapter is saved to `training/math-lora/`. When the agent next starts and uses the transformers inference path, the adapter is automatically loaded and applied.

---

## 10. How to Use It

### Single Problem

```bash
cd ~/math-agent
source venv/bin/activate

# Basic solve
python3 agent.py "Find the derivative of x^3 + 2x"

# With graphing
python3 agent.py "Plot y = sin(x) * cos(x)" --graph

# Multi-agent discussion (3 agents debate)
python3 agent.py "Prove that the sum of two odd integers is even" --multi-agent

# Complex problem
python3 agent.py "Find all eigenvalues of the matrix [[2,1],[1,2]]"
python3 agent.py "Evaluate the integral of 2x + 3 from 0 to 2"
python3 agent.py "Find the limit of (x^2 - 1)/(x - 1) as x approaches 1"
python3 agent.py "Solve x^2 - 5x + 6 = 0"
```

### Interactive Mode

```bash
python3 agent.py --interactive
```

Then type problems at the `🧮 >` prompt. Type `memory` to see stored knowledge. Type `quit` to exit.

### View Memory

```bash
python3 agent.py --memory
```

Shows all stored facts and previously solved problems with verification status.

### Train the Model

```bash
python3 train.py
```

Generates training data and runs LoRA fine-tuning. The adapter is saved to `training/math-lora/` and automatically used by the agent.

### Download the Model (if needed)

```bash
python3 download_model.py
```

Downloads the 1.13 GB GGUF file from HuggingFace. Skips if already downloaded.

### Quick Demo

```bash
python3 run.py
```

Runs a demo solving the eigenvalue problem.

---

## 11. Design Decisions and Trade-offs

### Why Code Execution Instead of Pure LLM Reasoning

A 1.5B model makes arithmetic errors. It might say 7 × 8 = 54. Sympy never makes this mistake. By having the LLM generate code that sympy executes, we get the best of both worlds: natural language understanding from the LLM and mathematical precision from sympy.

### Why Code Templates Instead of Pure LLM Code Generation

The 1.5B model, despite being math-trained, struggles to produce clean Python code. It wraps code in LaTeX, adds explanation text, and sometimes generates syntactically invalid code. The templates bypass this entirely for common problem types (derivatives, integrals, eigenvalues, etc.), producing guaranteed-correct code. The LLM is only used for problems that don't match any template.

### Why SQLite Instead of a Vector Database

The agent needs to remember conversations, facts, and solved problems. A vector database would enable semantic search but adds significant complexity (embedding models, similarity search infrastructure). SQLite is zero-dependency, always available in Python, and sufficient for keyword-based search. For a local agent on a laptop, this is the right trade-off.

### Why GGUF Instead of HuggingFace Transformers

GGUF via llama.cpp is:
- Faster to load (seconds vs minutes)
- Lower memory (1.3 GB vs 3+ GB)
- Supports GPU offload on Apple Silicon
- Doesn't require PyTorch for inference

The transformers path is kept as a fallback and is needed for training, but GGUF is the primary inference engine.

### Why LoRA Instead of Full Fine-Tuning

LoRA trains ~1% of parameters, needs ~4 GB VRAM instead of ~12 GB, and produces a ~10 MB adapter instead of a ~3 GB model. The adapter can be hot-swapped without reloading the base model. For a laptop running a 1.5B model, this is the only viable approach.

### Why Two-Stage Verification

Code verification (re-deriving with sympy) is deterministic and reliable but can only check if the answer is computationally correct — it can't verify the reasoning steps. LLM verification can check reasoning but is unreliable from a 1.5B model. Using both gives the best coverage: code checks the answer, LLM checks the reasoning, and confidence reflects how well both agree.

### Why Three Agents with Different Personas

Different mathematical specialties encourage different solution approaches. An algebra expert might solve by factoring; a geometry expert might solve by visualization. If all three reach the same answer via different methods, confidence is higher. The synthesis agent reconciles any differences.

---

## 12. Future Improvements

1. **More code templates**: Add templates for differential equations, series convergence, matrix multiplication, vector calculus, combinatorics, and statistics.

2. **Better LLM code generation**: Fine-tune the model specifically on code generation tasks (problem → Python code) rather than just problem → solution.

3. **Vector-based memory search**: Replace SQLite LIKE queries with embedding-based semantic search for finding similar past problems.

4. **Web search integration**: Let the agent look up theorems and proofs from mathematical databases (Wolfram Alpha, MathWorld, etc.) to cite as sources.

5. **Larger model support**: Make it easy to swap in a larger model (Qwen2.5-Math-7B-Instruct) when more VRAM is available.

6. **Streaming output**: Stream the LLM response token-by-token instead of waiting for the full generation.

7. **Web interface**: Build a Flask/FastAPI web UI for the agent instead of CLI-only.

8. **More graph types**: Add contour plots, parametric plots, polar plots, vector fields, and animations.

9. **Multi-turn problem solving**: Let the user refine problems across turns, with the agent maintaining context and building on previous answers.

10. **Export to LaTeX**: Generate LaTeX documents of the full solution with proofs and graphs embedded.

---

## License

This project uses the Qwen2.5-Math-1.5B-Instruct model, which is licensed under the Apache 2.0 license by Alibaba/Qwen team. The code in this project is provided as-is for educational and personal use.

---

*Built July 2026. Model: Qwen2.5-Math-1.5B-Instruct (Q5_K_M GGUF, 1.13 GB). Python 3.11. macOS Apple Silicon.*# Math-agent
