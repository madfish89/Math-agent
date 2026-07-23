"""
skills.py — All math skills for the Math AI Agent.

Skills:
  1. generate_code      — Generate Python (sympy) code to solve a problem
  2. search_web          — Search Wikipedia for formulas and theorems
  3. present_graph       — Create graph, histogram, or scatter plot data
  4. memory              — Remember previous problems (browser localStorage)
  5. double_check        — Verify a solution by re-deriving independently
  6. three_agents        — Split into 3 agents for complex problems
  7. graph_presenter     — Graph/histogram/scatter using matplotlib + seaborn
  8. multi_model         — Multi-model discussion prompts

All functions return plain Python dicts/strings.
Called from the browser via Pyodide — no server needed.
"""

import re as _re
import json
from sympy import *
from sympy import solve as sp_solve
import numpy as np

re = _re  # prevent sympy from shadowing the re module


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

    # Derivative
    m = re.search(r"derivative\s+of\s+(.+)|differentiate\s+(.+)", p)
    if m:
        e = norm((m.group(1) or m.group(2)).strip().rstrip("."))
        return f"from sympy import *\nx = symbols('x')\nexpr = sympify('{e}')\n_result = str(diff(expr, x))"

    # Integral
    m = re.search(r"integral\s+of\s+(.+)|integrate\s+(.+)", p)
    if m:
        raw = (m.group(1) or m.group(2)).strip()
        mi = re.search(r"(.+?)\s+from\s+(\S+)\s+to\s+(\S+)", raw)
        if mi:
            e = norm(mi.group(1).strip())
            return f"from sympy import *\nx = symbols('x')\nexpr = sympify('{e}')\n_result = str(integrate(expr, (x, sympify('{mi.group(2)}'), sympify('{mi.group(3)}'))))"
        e = norm(raw)
        return f"from sympy import *\nx = symbols('x')\nexpr = sympify('{e}')\n_result = str(integrate(expr, x))"

    # Solve equation
    m = re.search(r"solve\s+(.+)", p)
    if m:
        e = norm(m.group(1).strip())
        if "=" in e: parts = e.split("="); e = f"({parts[0]})-({parts[1]})"
        return f"from sympy import *\nx = symbols('x')\n_result = str(sympify('{e}'))"

    # Eigenvalues
    m = re.search(r"eigenvalue.*\[([^\]]+),\s*([^\]]+)\]", p) or re.search(r"\[([^\]]+),\s*([^\]]+)\].*eigenvalue", p)
    if m:
        r1 = [int(float(x)) for x in m.group(1).split(",")]
        r2 = [int(float(x)) for x in m.group(2).split(",")]
        return f"from sympy import *\nlam = symbols('l')\nA = Matrix({r1}, {r2})\n_result = str(solve(A.charpoly(lam), lam))"

    # Combinations
    m = re.search(r"([0-9]+)\s+choose\s+([0-9]+)", p)
    if m:
        return f"from sympy import *\n_result = str(binomial({m.group(1)}, {m.group(2)}))"

    # Factorial
    m = re.search(r"factorial\s+of\s+([0-9]+)|([0-9]+)!", p)
    if m:
        n = m.group(1) or m.group(2)
        return f"from sympy import *\n_result = str(factorial({n}))"

    return None


# ──────────────────────────────────────────────
# 2. WEB SEARCH
# Search Wikipedia for formulas and theorems.
# (Called from JS fetch — Python returns the search query.)
# ──────────────────────────────────────────────

def search_web(theorem):
    """Return a Wikipedia URL for the given theorem/formula."""
    title = theorem.replace(" ", "_").replace("(", "").replace(")", "")
    return f"https://en.wikipedia.org/wiki/{title}"


# ──────────────────────────────────────────────
# 3 & 7. GRAPH PRESENTER
# Create graph, histogram, or scatter plot data.
# Uses matplotlib + seaborn style (in browser: converts to Plotly traces).
# ──────────────────────────────────────────────

def present_graph(problem):
    """Detect graph requests and return Plotly-compatible trace data."""
    p = problem.lower().strip()

    # Graph y = f(x) from a to b
    m = re.search(r"graph\s+y\s*=\s*(.+?)\s+from\s+(-?[0-9]+)\s+to\s+(-?[0-9]+)", p)
    if m:
        e, a, b = m.group(1).strip(), int(m.group(2)), int(m.group(3))
        return _line_graph(e, a, b)

    # Graph f and g from a to b
    m = re.search(r"graph\s+(.+?)\s+and\s+(.+?)\s+from\s+(-?[0-9]+)\s+to\s+(-?[0-9]+)", p)
    if m:
        return _line_graph_multi(m.group(1).strip(), m.group(2).strip(), int(m.group(3)), int(m.group(4)))

    # Graph y = f(x)
    m = re.search(r"graph\s+y\s*=\s*(.+)", p)
    if m:
        return _line_graph(m.group(1).strip(), -10, 10)

    # Histogram of [data]
    m = re.search(r"histogram\s+of\s+\[([^\]]+)\]", p)
    if m:
        return _histogram([float(x) for x in m.group(1).split(",")])

    # Scatter plot of [data]
    m = re.search(r"scatter\s+plot\s+of\s+\[([^\]]+)\]", p) or re.search(r"scatter\s+\[([^\]]+)\]", p)
    if m:
        nums = [float(x) for x in m.group(1).split(",")]
        return _scatter(nums)

    return None


def _line_graph(expr_str, a, b, pts=200):
    """Generate line graph trace data (matplotlib plt.plot equivalent)."""
    x = symbols("x")
    expr = sympify(norm(expr_str))
    f = lambdify(x, expr, "numpy")
    xs = np.linspace(a, b, pts)
    ys = [float(f(xi)) for xi in xs]
    return {"traces": [{"x": xs.tolist(), "y": ys, "name": expr_str, "type": "scatter", "mode": "lines"}],
            "title": "Graph", "x_range": [a, b]}


def _line_graph_multi(e1, e2, a, b, pts=200):
    """Generate multi-line graph (plt.plot with multiple lines + seaborn style)."""
    x = symbols("x")
    traces = []
    for e in [e1, e2]:
        expr = sympify(norm(e))
        f = lambdify(x, expr, "numpy")
        xs = np.linspace(a, b, pts)
        ys = [float(f(xi)) for xi in xs]
        traces.append({"x": xs.tolist(), "y": ys, "name": e, "type": "scatter", "mode": "lines"})
    return {"traces": traces, "title": "Graph", "x_range": [a, b]}


def _histogram(data):
    """Generate histogram trace data (plt.hist + seaborn sns.histplot equivalent)."""
    return {"traces": [{"x": data, "type": "histogram", "name": "Data", "marker": {"color": "#58a6ff"}}],
            "title": "Histogram", "x_range": [min(data) - 1, max(data) + 1]}


def _scatter(data):
    """Generate scatter plot trace data (plt.scatter + seaborn equivalent)."""
    x = list(range(len(data)))
    return {"traces": [{"x": x, "y": data, "type": "scatter", "mode": "markers",
                         "marker": {"size": 8, "color": "#3fb950"}, "name": "Data"}],
            "title": "Scatter Plot", "x_range": [0, len(data)]}


# ──────────────────────────────────────────────
# 4. MEMORY
# Remember previous problems (stored in browser localStorage by JS).
# Python provides the data structure.
# ──────────────────────────────────────────────

def memory_format(problem, answer, confidence):
    """Format a solved problem for storage."""
    return {"problem": problem, "answer": answer, "confidence": confidence}


# ──────────────────────────────────────────────
# 5. DOUBLE CHECK
# Verify a solution by re-deriving with sympy independently.
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
        # Simple comparison: check if key numbers match
        match = _answers_match(llm_answer, py_answer)
        return {"verified": match, "python_answer": fmt(py_answer), "llm_answer": llm_answer}
    except Exception as e:
        return {"verified": False, "reason": str(e)}


def _answers_match(a, b):
    """Check if two answer strings are numerically equivalent."""
    # Extract numbers from both
    nums_a = re.findall(r"-?[0-9]+\.?[0-9]*", str(a))
    nums_b = re.findall(r"-?[0-9]+\.?[0-9]*", str(b))
    if not nums_a or not nums_b:
        return str(a).strip() == str(b).strip()
    # Check if the main numbers match
    try:
        return abs(float(nums_a[0]) - float(nums_b[0])) < 0.01
    except:
        return str(a).strip() == str(b).strip()


# ──────────────────────────────────────────────
# 6 & 8. THREE AGENTS / MULTI-MODEL DISCUSSION
# Split into 3 agents with different expertise.
# Returns prompts for JS to send to the LLM.
# ──────────────────────────────────────────────

def three_agents(problem):
    """Return 3 agent prompts + 1 synthesis prompt for multi-model discussion."""
    agents = [
        {"name": "Agent 1 — Algebra & Calculus",
         "system": "You are Agent 1, expert in algebra, calculus, and symbolic computation. Respond ONLY in JSON: {\"answer\": str, \"approach\": str, \"confidence\": float, \"concerns\": list}",
         "user": problem},
        {"name": "Agent 2 — Number Theory & Proofs",
         "system": "You are Agent 2, expert in number theory, combinatorics, and proof writing. Respond ONLY in JSON: {\"answer\": str, \"approach\": str, \"confidence\": float, \"concerns\": list}",
         "user": problem},
        {"name": "Agent 3 — Geometry & Numerics",
         "system": "You are Agent 3, expert in geometry, topology, and numerical methods. Respond ONLY in JSON: {\"answer\": str, \"approach\": str, \"confidence\": float, \"concerns\": list}",
         "user": problem},
    ]
    synthesis = {
        "system": "You are a synthesis agent. Combine expert opinions into one rigorous solution with proofs. Respond ONLY in JSON: {\"answer\": str, \"steps\": [{\"step\": str, \"explanation\": str, \"proof\": str}], \"confidence\": float, \"consensus\": bool}",
        "user": problem  # JS will append agent responses
    }
    return {"agents": agents, "synthesis": synthesis}


# ──────────────────────────────────────────────
# MAIN ENTRY POINT
# Called by the browser. Returns JSON string.
# ──────────────────────────────────────────────

def handle(problem):
    """
    Main entry point called from the browser via Pyodide.
    Returns graph data if the problem is a graph/histogram/scatter request.
    Returns None if no Python skill applies (JS will use LLM only).
    """
    result = {"problem": problem}

    # Check for graph request
    graph = present_graph(problem)
    if graph:
        result["graph_data"] = graph

    # Check for web search query
    # (JS handles the actual fetch; Python just identifies what to search)

    if "graph_data" in result:
        return json.dumps(result)

    return None