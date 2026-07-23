import re as _re
import json
import io
import base64
from sympy import *
from sympy import solve as sp_solve
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

re = _re
plt.style.use("dark_background")

# Colors matching the UI theme
C_BLUE = "#58a6ff"
C_PURPLE = "#bc8cff"
C_GREEN = "#3fb950"
C_BG = "#161b22"
C_GRID = "#30363d"


# ──────────────────────────────────────────────
# 1. CODE GENERATION
# ──────────────────────────────────────────────

def norm(s):
    s = s.strip().replace("^", "**")
    FUNCS = ["sin","cos","tan","exp","log","sqrt","abs","ln","sec","csc","cot",
             "asin","acos","atan","sinh","cosh","tanh","erf","gamma"]
    c0, c1 = chr(0), chr(1)
    for f in FUNCS:
        s = s.replace(f+"(", c0+f+c1)
    s = re.sub(r"([0-9])([a-zA-Z])", r"\1*\2", s)
    s = re.sub(r"\)([a-zA-Z0-9])", r")*\1", s)
    s = re.sub(r"([a-zA-Z])\(", r"\1*(", s)
    for f in FUNCS:
        s = s.replace(c0+f+c1, f+"(")
    return s


def fmt(s):
    s = str(s)
    sup = {"0":"⁰","1":"¹","2":"²","3":"³","4":"⁴","5":"⁵","6":"⁶","7":"⁷","8":"⁸","9":"⁹"}
    s = re.sub(r"\*\*([0-9]+)", lambda m: "".join(sup.get(d,d) for d in m.group(1)), s)
    s = s.replace("sqrt(", "√(").replace("pi", "π").replace("oo", "∞")
    s = re.sub(r"Rational\(([0-9]+),\s*([0-9]+)\)", r"\1/\2", s)
    s = s.replace("*", "·")
    if s.startswith("[") and s.endswith("]"): s = s[1:-1]
    return s.strip()


def generate_code(problem):
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
    title = theorem.replace(" ", "_").replace("(", "").replace(")", "")
    return f"https://en.wikipedia.org/wiki/{title}"


# ──────────────────────────────────────────────
# 3. GRAPH PRESENTER (matplotlib)
# ──────────────────────────────────────────────

def present_graph(problem):
    p = problem.lower().strip()

    # Multi-graph first
    m = re.search(r"graph\s+y\s*=\s*(.+?)\s+and\s+y\s*=\s*(.+?)\s+from\s+(-?[0-9]+)\s+to\s+(-?[0-9]+)", p)
    if m:
        return _multi(m.group(1).strip(), m.group(2).strip(), int(m.group(3)), int(m.group(4)))
    m = re.search(r"graph\s+(.+?)\s+and\s+(.+?)\s+from\s+(-?[0-9]+)\s+to\s+(-?[0-9]+)", p)
    if m:
        return _multi(m.group(1).strip(), m.group(2).strip(), int(m.group(3)), int(m.group(4)))

    # Single graph with range
    m = re.search(r"graph\s+y\s*=\s*(.+?)\s+from\s+(-?[0-9]+)\s+to\s+(-?[0-9]+)", p)
    if m:
        return _line(m.group(1).strip(), int(m.group(2)), int(m.group(3)))

    # Single graph no range
    m = re.search(r"graph\s+y\s*=\s*(.+)", p)
    if m:
        return _line(m.group(1).strip(), -10, 10)

    # Histogram
    m = re.search(r"histogram\s+of\s+\[([^\]]+)\]", p)
    if m:
        return _hist([float(x) for x in m.group(1).split(",")])

    # Scatter
    m = re.search(r"scatter\s+plot\s+of\s+\[([^\]]+)\]|scatter\s+\[([^\]]+)\]", p)
    if m:
        nums = [float(x) for x in (m.group(1) or m.group(2)).split(",")]
        return _scatter(nums)

    return None


def _strip_y(e):
    e = e.strip()
    if e.lower().startswith("y ="): e = e[3:].strip()
    elif e.lower().startswith("y="): e = e[2:].strip()
    return e


def _fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=C_BG, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def _line(expr_str, a, b, pts=400):
    expr_str = _strip_y(expr_str)
    x = symbols("x")
    expr = sympify(norm(expr_str))
    f = lambdify(x, expr, "numpy")
    xs = np.linspace(a, b, pts)
    ys = np.array([float(f(xi)) for xi in xs])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, ys, color=C_BLUE, linewidth=2.5, label=f"y = {expr_str}")
    ax.set_xlabel("x", fontsize=13, color="#e6edf3")
    ax.set_ylabel("y", fontsize=13, color="#e6edf3")
    ax.set_title(f"y = {expr_str}", fontsize=15, pad=12, color="#e6edf3")
    ax.legend(fontsize=12, facecolor=C_BG, edgecolor=C_GRID, labelcolor="#e6edf3")
    ax.axhline(0, color=C_GRID, linewidth=0.8)
    ax.axvline(0, color=C_GRID, linewidth=0.8)
    ax.grid(True, color=C_GRID, alpha=0.3)
    ax.set_facecolor(C_BG)
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values(): spine.set_color(C_GRID)
    fig.tight_layout()
    return {"image": _fig_b64(fig), "title": f"y = {expr_str}"}


def _multi(e1, e2, a, b, pts=400):
    e1, e2 = _strip_y(e1), _strip_y(e2)
    x = symbols("x")
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [C_BLUE, C_PURPLE]
    for i, e in enumerate([e1, e2]):
        expr = sympify(norm(e))
        f = lambdify(x, expr, "numpy")
        xs = np.linspace(a, b, pts)
        ys = np.array([float(f(xi)) for xi in xs])
        ax.plot(xs, ys, color=colors[i], linewidth=2.5, label=f"y = {e}")
    ax.set_xlabel("x", fontsize=13, color="#e6edf3")
    ax.set_ylabel("y", fontsize=13, color="#e6edf3")
    ax.set_title("Graph Comparison", fontsize=15, pad=12, color="#e6edf3")
    ax.legend(fontsize=12, facecolor=C_BG, edgecolor=C_GRID, labelcolor="#e6edf3")
    ax.axhline(0, color=C_GRID, linewidth=0.8)
    ax.grid(True, color=C_GRID, alpha=0.3)
    ax.set_facecolor(C_BG)
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values(): spine.set_color(C_GRID)
    fig.tight_layout()
    return {"image": _fig_b64(fig), "title": "Graph Comparison"}


def _hist(data):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(data, bins=min(len(data), 12), color=C_BLUE, alpha=0.85, edgecolor=C_PURPLE)
    ax.set_xlabel("Value", fontsize=13, color="#e6edf3")
    ax.set_ylabel("Count", fontsize=13, color="#e6edf3")
    ax.set_title("Histogram", fontsize=15, pad=12, color="#e6edf3")
    ax.grid(True, color=C_GRID, alpha=0.3, axis="y")
    ax.set_facecolor(C_BG)
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values(): spine.set_color(C_GRID)
    fig.tight_layout()
    return {"image": _fig_b64(fig), "title": "Histogram"}


def _scatter(data):
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = list(range(len(data)))
    ax.scatter(xs, data, s=100, color=C_GREEN, edgecolor=C_BLUE, linewidth=1.5, zorder=5)
    ax.set_xlabel("Index", fontsize=13, color="#e6edf3")
    ax.set_ylabel("Value", fontsize=13, color="#e6edf3")
    ax.set_title("Scatter Plot", fontsize=15, pad=12, color="#e6edf3")
    ax.grid(True, color=C_GRID, alpha=0.3)
    ax.set_facecolor(C_BG)
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values(): spine.set_color(C_GRID)
    fig.tight_layout()
    return {"image": _fig_b64(fig), "title": "Scatter Plot"}


# ──────────────────────────────────────────────
# 4. MEMORY
# ──────────────────────────────────────────────

def memory_format(problem, answer, confidence):
    return {"problem": problem, "answer": answer, "confidence": confidence}


# ──────────────────────────────────────────────
# 5. DOUBLE CHECK
# ──────────────────────────────────────────────

def double_check(problem, llm_answer):
    code = generate_code(problem)
    if not code:
        return {"verified": False, "reason": "No code"}
    try:
        local = {}
        exec(code, local)
        py_answer = str(local.get("_result", ""))
        match = _match(llm_answer, py_answer)
        return {"verified": match, "python_answer": fmt(py_answer)}
    except Exception as e:
        return {"verified": False, "reason": str(e)}


def _match(a, b):
    nums_a = re.findall(r"-?[0-9]+\.?[0-9]*", str(a))
    nums_b = re.findall(r"-?[0-9]+\.?[0-9]*", str(b))
    if not nums_a or not nums_b:
        return str(a).strip() == str(b).strip()
    try:
        return abs(float(nums_a[0]) - float(nums_b[0])) < 0.01
    except:
        return str(a).strip() == str(b).strip()


# ──────────────────────────────────────────────
# 6. THREE AGENTS
# ──────────────────────────────────────────────

def three_agents(problem):
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
    result = {"problem": problem}
    graph = present_graph(problem)
    if graph:
        result["graph_image"] = graph["image"]
        result["graph_title"] = graph.get("title", "Graph")
    if "graph_image" in result:
        return json.dumps(result)
    return None