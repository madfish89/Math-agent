"""
skills.py — All math skills for the Math AI Agent.

Skills:
  1. generate_code      — Generate Python (sympy) code to solve a problem
  2. search_web          — Search Wikipedia for formulas and theorems
  3. present_graph       — Create graph, histogram, or scatter plot (matplotlib + seaborn)
  4. memory              — Remember previous problems (browser localStorage)
  5. double_check        — Verify a solution by re-deriving independently
  6. three_agents        — Split into 3 agents for complex problems
  7. multi_model          — Multi-model discussion prompts

All functions return plain Python dicts/strings.
Called from the browser via Pyodide — no server needed.
"""

import re as _re
import json
import io
import base64
from sympy import *
from sympy import solve as sp_solve
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless mode for Pyodide
import matplotlib.pyplot as plt
import seaborn as sns

re = _re  # prevent sympy from shadowing the re module

sns.set_theme(style="darkgrid", palette="muted")
plt.style.use("dark_background")


# ──────────────────────────────────────────────
# 1. CODE GENERATION
# Generate Python (sympy) code to solve a math problem.
# ──────────────────────────────────────────────

def norm(s):
    """Normalize math notation for sympy: 3x → 3*x, ^ → **."""
    s = s.strip().replace("^", "**")
    for f in ["sin","cos","tan","exp","log","sqrt","abs","ln","sec","csc","cot",
              "asin","acos","atan","sinh","cosh","tanh","erf","gamma"]:
        s = s.replace(f+"(", "\x00"+f+"\x01")
    s = re.sub(r"([0-9])([a-zA-Z])", r"\1*\2", s)
    s = re.sub(r"\)([a-zA-Z0-9])", r")*\1", s)
    s = re.sub(r"([a-zA-Z])\(", r"\1*(", s)
    for f in ["sin","cos","tan","exp","log","sqrt","abs","ln","sec","csc","cot",
              "asin","acos","atan","sinh","cosh","tanh","erf","gamma"]:
        s = s.replace("\x00"+f+"\x01", f+"(")
    return s


def fmt(s):
    """Format sympy output for humans: x**2 → x², * → ·, sqrt → √."""
    s = str(s)
    sup = {"0":"⁰","1":"¹","2":"²","3":"³","4":"⁴","5":"⁵","6":"⁶","7":"⁷","8":"⁸","9":"⁹"}
    s = re.sub(r"\*\*([0-9]+)", lambda m: "".join(sup.get(d,d) for d in m.group(1)), s)
    s = s.replace("sqrt(", "√(").replace("pi", "π").replace("oo", "∞")
    s = re.sub(r"Rational\(([0-9]+),\s*([0-9]+)\)", r"\1/\2", s)
    s = s.replace("*", "·")
    if s.startswith("[") and s.endswith("]"): s = s[1:-1]
    return s.strip()


def generate_code(problem):
    """Generate Python code (as a string) to solve the problem with sympy."""
    p = problem.lower().strip()
    m = re.search(r"derivative\s+of\s+(.+)|differentiate\s+(.+)", p)
    if m:
        e = norm((m.group(1) or m.group(2)).strip().rstrip("."))
        return f"from sympy import *\nx = symbols('x')\n_result = str(diff(sympify('{e}'), x))"
    m = re.search(r"integral\s+of\s+(.+)|integrate\s+(.+)", p)
    if m:
        raw = (m.group(1) or m.group(2)).strip()
        mi = re.search(r"(.+?)\s+from\s+(\S+)\s+to\s+(\S+)", raw)
        if mi:
            e = norm(mi.group(1).strip())
            return f"from sympy import *\nx = symbols('x')\n_result = str(integrate(sympify('{e}'), (x, sympify('{mi.group(2)}'), sympify('{mi.group(3)}'))))"
        return f"from sympy import *\nx = symbols('x')\n_result = str(integrate(sympify('{norm(raw)}'), x))"
    m = re.search(r"solve\s+(.+)", p)
    if m:
        e = norm(m.group(1).strip())
        if "=" in e: parts = e.split("="); e = f"({parts[0]})-({parts[1]})"
        return f"from sympy import *\nx = symbols('x')\n_result = str(sp_solve(sympify('{e}'), x))"
    m = re.search(r"([0-9]+)\s+choose\s+([0-9]+)", p)
    if m: return f"from sympy import *\n_result = str(binomial({m.group(1)}, {m.group(2)}))"
    m = re.search(r"factorial\s+of\s+([0-9]+)|([0-9]+)!", p)
    if m: return f"from sympy import *\n_result = str(factorial({m.group(1) or m.group(2)}))"
    return None


# ──────────────────────────────────────────────
# 2. WEB SEARCH
# ──────────────────────────────────────────────

def search_web(theorem):
    """Return a Wikipedia URL for the given theorem/formula."""
    title = theorem.replace(" ", "_").replace("(", "").replace(")", "")
    return f"https://en.wikipedia.org/wiki/{title}"


# ──────────────────────────────────────────────
# 3. GRAPH PRESENTER (matplotlib + seaborn)
# Returns a base64 PNG image that the browser displays directly.
# ──────────────────────────────────────────────

def present_graph(problem):
    """Detect graph/histogram/scatter requests and return a base64 PNG image."""
    p = problem.lower().strip()

    m = re.search(r"graph\s+y\s*=\s*(.+?)\s+from\s+(-?[0-9]+)\s+to\s+(-?[0-9]+)", p)
    if m:
        return _line_graph(m.group(1).strip(), int(m.group(2)), int(m.group(3)))

    m = re.search(r"graph\s+(.+?)\s+and\s+(.+?)\s+from\s+(-?[0-9]+)\s+to\s+(-?[0-9]+)", p)
    if m:
        return _line_graph_multi(m.group(1).strip(), m.group(2).strip(),
                                  int(m.group(3)), int(m.group(4)))

    m = re.search(r"graph\s+y\s*=\s*(.+)", p)
    if m:
        return _line_graph(m.group(1).strip(), -10, 10)

    m = re.search(r"histogram\s+of\s+\[([^\]]+)\]", p)
    if m:
        return _histogram([float(x) for x in m.group(1).split(",")])

    m = re.search(r"scatter\s+plot\s+of\s+\[([^\]]+)\]|scatter\s+\[([^\]]+)\]", p)
    if m:
        nums = [float(x) for x in (m.group(1) or m.group(2)).split(",")]
        return _scatter(nums)

    return None


def _fig_to_base64(fig):
    """Convert a matplotlib figure to a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
               facecolor="#161b22", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def _line_graph(expr_str, a, b, pts=400):
    """Line graph using plt.plot + seaborn style."""
    x = symbols("x")
    expr = sympify(norm(expr_str))
    f = lambdify(x, expr, "numpy")
    xs = np.linspace(a, b, pts)
    ys = np.array([float(f(xi)) for xi in xs])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, color="#58a6ff", linewidth=2, label=f"y = {expr_str}")
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title(f"Graph of y = {expr_str}", fontsize=14, pad=12)
    ax.legend(fontsize=11)
    ax.axhline(0, color="#30363d", linewidth=0.8)
    ax.axvline(0, color="#30363d", linewidth=0.8)
    fig.tight_layout()
    return {"image": _fig_to_base64(fig), "title": f"Graph of y = {expr_str}"}


def _line_graph_multi(e1, e2, a, b, pts=400):
    """Multi-line graph using plt.plot + seaborn palette."""
    x = symbols("x")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#58a6ff", "#bc8cff"]
    for i, e in enumerate([e1, e2]):
        expr = sympify(norm(e))
        f = lambdify(x, expr, "numpy")
        xs = np.linspace(a, b, pts)
        ys = np.array([float(f(xi)) for xi in xs])
        ax.plot(xs, ys, color=colors[i], linewidth=2, label=f"y = {e}")
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title("Graph Comparison", fontsize=14, pad=12)
    ax.legend(fontsize=11)
    ax.axhline(0, color="#30363d", linewidth=0.8)
    fig.tight_layout()
    return {"image": _fig_to_base64(fig), "title": "Graph Comparison"}


def _histogram(data):
    """Histogram using sns.histplot + plt."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data, bins=min(len(data), 15), color="#58a6ff", alpha=0.8, ax=ax)
    ax.set_xlabel("Value", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Histogram", fontsize=14, pad=12)
    fig.tight_layout()
    return {"image": _fig_to_base64(fig), "title": "Histogram"}


def _scatter(data):
    """Scatter plot using sns.scatterplot + plt."""
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = list(range(len(data)))
    sns.scatterplot(x=xs, y=data, s=80, color="#3fb950", ax=ax)
    ax.set_xlabel("Index", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("Scatter Plot", fontsize=14, pad=12)
    fig.tight_layout()
    return {"image": _fig_to_base64(fig), "title": "Scatter Plot"}


# ──────────────────────────────────────────────
# 4. MEMORY
# ──────────────────────────────────────────────

def memory_format(problem, answer, confidence):
    """Format a solved problem for storage."""
    return {"problem": problem, "answer": answer, "confidence": confidence}


# ──────────────────────────────────────────────
# 5. DOUBLE CHECK
# ──────────────────────────────────────────────

def double_check(problem, llm_answer):
    """Re-derive the answer with sympy and compare to the LLM's answer."""
    code = generate_code(problem)
    if not code:
        return {"verified": False, "reason": "Could not generate verification code"}
    try:
        local = {}
        exec(code, local)
        py_answer = str(local.get("_result", ""))
        match = _answers_match(llm_answer, py_answer)
        return {"verified": match, "python_answer": fmt(py_answer)}
    except Exception as e:
        return {"verified": False, "reason": str(e)}


def _answers_match(a, b):
    """Check if two answer strings are numerically equivalent."""
    nums_a = re.findall(r"-?[0-9]+\.?[0-9]*", str(a))
    nums_b = re.findall(r"-?[0-9]+\.?[0-9]*", str(b))
    if not nums_a or not nums_b:
        return str(a).strip() == str(b).strip()
    try:
        return abs(float(nums_a[0]) - float(nums_b[0])) < 0.01
    except:
        return str(a).strip() == str(b).strip()


# ──────────────────────────────────────────────
# 6 & 7. THREE AGENTS / MULTI-MODEL DISCUSSION
# ──────────────────────────────────────────────

def three_agents(problem):
    """Return 3 agent prompts + 1 synthesis prompt for multi-model discussion."""
    agents = [
        {"name": "Agent 1 — Algebra & Calculus",
         "system": 'You are Agent 1, expert in algebra, calculus, and symbolic computation. Respond ONLY in JSON: {"answer": str, "approach": str, "confidence": float, "concerns": list}',
         "user": problem},
        {"name": "Agent 2 — Number Theory & Proofs",
         "system": 'You are Agent 2, expert in number theory, combinatorics, and proof writing. Respond ONLY in JSON: {"answer": str, "approach": str, "confidence": float, "concerns": list}',
         "user": problem},
        {"name": "Agent 3 — Geometry & Numerics",
         "system": 'You are Agent 3, expert in geometry, topology, and numerical methods. Respond ONLY in JSON: {"answer": str, "approach": str, "confidence": float, "concerns": list}',
         "user": problem},
    ]
    synthesis = {
        "system": 'You are a synthesis agent. Combine expert opinions into one rigorous solution with proofs. Respond ONLY in JSON: {"answer": str, "steps": [{"step": str, "explanation": str, "proof": str}], "confidence": float, "consensus": bool}',
        "user": problem
    }
    return {"agents": agents, "synthesis": synthesis}


# ──────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────

def handle(problem):
    """Main entry point called from the browser via Pyodide."""
    result = {"problem": problem}
    graph = present_graph(problem)
    if graph:
        result["graph_image"] = graph["image"]
        result["graph_title"] = graph.get("title", "Graph")
    if "graph_image" in result:
        return json.dumps(result)
    return None