import re as _re
import json
import io
import base64
from sympy import *
from sympy import solve as sp_solve
from sympy import limit as sp_limit
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

re = _re
plt.style.use("dark_background")

C_BLUE = "#58a6ff"
C_PURPLE = "#bc8cff"
C_GREEN = "#3fb950"
C_BG = "#161b22"
C_GRID = "#30363d"
C_ORANGE = "#d29922"
C_RED = "#f85149"


# ──────────────────────────────────────────────
# 1. EXPRESSION NORMALIZATION
# ──────────────────────────────────────────────

def norm(s):
    s = s.strip()
    sup_map = {"⁰":"**0","¹":"**1","²":"**2","³":"**3","⁴":"**4","⁵":"**5","⁶":"**6","⁷":"**7","⁸":"**8","⁹":"**9"}
    for k, v in sup_map.items():
        s = s.replace(k, v)
    s = s.replace(chr(0x2212), "-").replace(chr(0x2013), "-").replace(chr(0x2014), "-")
    s = s.replace("^", "**")
    s = s.replace("·", "*").replace("×", "*")
    s = re.sub(r"^y\s*=\s*", "", s)
    s = re.sub(r"^\(y\)\s*=\s*", "", s)
    FUNCS = ["sin","cos","tan","exp","log","sqrt","abs","ln","sec","csc","cot",
             "asin","acos","atan","sinh","cosh","tanh","erf","gamma","log10","log2"]
    c0, c1 = chr(0), chr(1)
    for f in FUNCS:
        s = s.replace(f+"(", c0+f+c1)
    s = re.sub(r"([0-9])([a-zA-Z])", r"\1*\2", s)
    s = re.sub(r"\)([a-zA-Z0-9])", r")*\1", s)
    s = re.sub(r"([a-zA-Z])\(", r"\1*(", s)
    for f in FUNCS:
        s = s.replace(c0+f+c1, f+"(")
    s = s.replace("ln(", "log(")
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


# ──────────────────────────────────────────────
# 2. CODE GENERATION — detect problem type, generate SymPy code
# ──────────────────────────────────────────────

def generate_code(problem):
    p = problem.lower().strip()

    # ── Partial derivative: d/dx of x²*y at point ──
    m = re.search(r"partial\s+derivative\s+of\s+(.+?)\s+with\s+respect\s+to\s+(\w+)", p)
    if m:
        e = norm(m.group(1).strip().rstrip("."))
        var = m.group(2).strip()
        return f"from sympy import *\n{var} = symbols('{var}')\n_result = str(diff(sympify('{e}'), {var}))"

    # ── Derivative ──
    m = re.search(r"derivative\s+of\s+(.+)|differentiate\s+(.+)", p)
    if m:
        e = norm((m.group(1) or m.group(2)).strip().rstrip("."))
        return f"from sympy import *\nx = symbols('x')\n_result = str(diff(sympify('{e}'), x))"

    # ── Second derivative ──
    m = re.search(r"second\s+derivative\s+of\s+(.+)", p)
    if m:
        e = norm(m.group(1).strip().rstrip("."))
        return f"from sympy import *\nx = symbols('x')\n_result = str(diff(sympify('{e}'), x, 2))"

    # ── Integral ──
    m = re.search(r"integral\s+of\s+(.+)|integrate\s+(.+)", p)
    if m:
        raw = (m.group(1) or m.group(2)).strip()
        mi = re.search(r"(.+?)\s+from\s+(\S+)\s+to\s+(\S+)", raw)
        if mi:
            e = norm(mi.group(1).strip())
            return f"from sympy import *\nx = symbols('x')\n_result = str(integrate(sympify('{e}'), (x, sympify('{mi.group(2)}'), sympify('{mi.group(3)}'))))"
        return f"from sympy import *\nx = symbols('x')\n_result = str(integrate(sympify('{norm(raw)}'), x))"

    # ── Double integral ──
    m = re.search(r"double\s+integral\s+of\s+(.+)", p)
    if m:
        e = norm(m.group(1).strip())
        return f"from sympy import *\nx, y = symbols('x y')\n_result = str(integrate(sympify('{e}'), (x, -oo, oo), (y, -oo, oo)))"

    # ── Limit ──
    m = re.search(r"limit\s+of\s+(.+?)\s+as\s+(\w+)\s+approaches\s+(.+?)(?:\s*$|\.)", p)
    if m:
        e = norm(m.group(1).strip())
        var = m.group(2).strip()
        val = m.group(3).strip()
        return f"from sympy import *\n{var} = symbols('{var}')\n_result = str(sp_limit(sympify('{e}'), {var}, sympify('{val}')))"

    # ── Solve equation ──
    m = re.search(r"solve\s+(.+)", p)
    if m:
        e = norm(m.group(1).strip())
        if "=" in e and "==" not in e:
            parts = e.split("=")
            e = f"({parts[0]})-({parts[1]})"
        return f"from sympy import *\nx = symbols('x')\n_result = str(sp_solve(sympify('{e}'), x))"

    # ── System of equations ──
    m = re.search(r"system\s+of\s+equations\s*[:\s]*(.+)", p)
    if m:
        eqs = m.group(1).strip()
        return f"from sympy import *\nx, y = symbols('x y')\n_result = str(sp_solve([sympify('{eqs}')], [x, y]))"

    # ── Combinations ──
    m = re.search(r"([0-9]+)\s+choose\s+([0-9]+)|c\(([0-9]+)\s*,\s*([0-9]+)\)", p)
    if m:
        n = m.group(1) or m.group(3)
        k = m.group(2) or m.group(4)
        return f"from sympy import *\n_result = str(binomial({n}, {k}))"

    # ── Permutations ──
    m = re.search(r"(\d+)\s+permutations?\s+of\s+(\d+)|p\((\d+)\s*,\s*(\d+)\)", p)
    if m:
        n = m.group(1) or m.group(3)
        k = m.group(2) or m.group(4)
        return f"from sympy import *\n_result = str(factorial({n})//factorial({n}-{k}))"

    # ── Factorial ──
    m = re.search(r"factorial\s+of\s+([0-9]+)|([0-9]+)!", p)
    if m: return f"from sympy import *\n_result = str(factorial({m.group(1) or m.group(2)}))"

    # ── Matrix operations ──
    m = re.search(r"determinant\s+of\s+(\[\[.*?\]\])", p)
    if m: return f"from sympy import *\n_result = str(Matrix({m.group(1)}).det())"
    m = re.search(r"inverse\s+of\s+(\[\[.*?\]\])", p)
    if m: return f"from sympy import *\n_result = str(Matrix({m.group(1)}).inv())"
    m = re.search(r"transpose\s+of\s+(\[\[.*?\]\])", p)
    if m: return f"from sympy import *\n_result = str(Matrix({m.group(1)}).T)"
    m = re.search(r"eigenvalues?\s+of\s+(\[\[.*?\]\])", p)
    if m: return f"from sympy import *\n_result = str(Matrix({m.group(1)}).eigenvals())"
    m = re.search(r"trace\s+of\s+(\[\[.*?\]\])", p)
    if m: return f"from sympy import *\n_result = str(Matrix({m.group(1)}).trace())"
    m = re.search(r"rank\s+of\s+(\[\[.*?\]\])", p)
    if m: return f"from sympy import *\n_result = str(Matrix({m.group(1)}).rank())"

    # ── Taylor series ──
    m = re.search(r"taylor\s+series\s+of\s+(.+?)\s+around\s+(\S+)(?:\s+order\s+(\d+))?(?:\s*$|\.)", p)
    if m:
        e = norm(m.group(1).strip())
        pt = m.group(2).strip()
        n = int(m.group(3)) if m.group(3) else 6
        return f"from sympy import *\nx = symbols('x')\n_result = str(series(sympify('{e}'), x, sympify('{pt}'), {n+1}).removeO())"

    # ── Differential equations ──
    m = re.search(r"(?:solve\s+ode|differential\s+equation|solve\s+the\s+ode)\s*[:\s]*(.+)", p)
    if m:
        eq = norm(m.group(1).strip())
        if "=" in eq and "Eq(" not in eq:
            parts = eq.split("=")
            eq = f"Eq({parts[0]},{parts[1]})"
        return f"from sympy import *\nx = symbols('x'); y = Function('y')\n_result = str(dsolve(sympify('{eq}'), y(x)))"

    # ── Statistics ──
    m = re.search(r"mean\s+of\s+\[([^\]]+)\]", p)
    if m: return f"from statistics import mean as _m\n_result = str(_m([{m.group(1)}]))"
    m = re.search(r"median\s+of\s+\[([^\]]+)\]", p)
    if m: return f"from statistics import median as _m\n_result = str(_m([{m.group(1)}]))"
    m = re.search(r"(?:std|standard\s+deviation)\s+of\s+\[([^\]]+)\]", p)
    if m: return f"from statistics import stdev as _s\n_result = str(_s([{m.group(1)}]))"
    m = re.search(r"variance\s+of\s+\[([^\]]+)\]", p)
    if m: return f"from statistics import variance as _v\n_result = str(_v([{m.group(1)}]))"
    m = re.search(r"mode\s+of\s+\[([^\]]+)\]", p)
    if m: return f"from statistics import mode as _mo\n_result = str(_mo([{m.group(1)}]))"

    # ── Trig simplification ──
    m = re.search(r"simplify\s+(.+)", p)
    if m:
        e = norm(m.group(1).strip().rstrip("."))
        return f"from sympy import *\nx = symbols('x')\n_result = str(simplify(sympify('{e}')))"

    # ── Expand ──
    m = re.search(r"expand\s+(.+)", p)
    if m:
        e = norm(m.group(1).strip().rstrip("."))
        return f"from sympy import *\nx = symbols('x')\n_result = str(expand(sympify('{e}')))"

    # ── Factor ──
    m = re.search(r"factor\s+(.+)", p)
    if m:
        e = norm(m.group(1).strip().rstrip("."))
        return f"from sympy import *\nx = symbols('x')\n_result = str(factor(sympify('{e}')))"

    # ── Sum / series sum ──
    m = re.search(r"sum\s+of\s+(.+?)\s+from\s+(\d+)\s+to\s+(\d+)", p)
    if m:
        e = norm(m.group(1).strip())
        a, b = m.group(2), m.group(3)
        return f"from sympy import *\nn = symbols('n')\n_result = str(summation(sympify('{e}'), (n, {a}, {b})))"

    # ── Logarithm ──
    m = re.search(r"log(?:arithm)?\s+base\s+(\d+)\s+of\s+(\d+)", p)
    if m:
        base, val = m.group(1), m.group(2)
        return f"from sympy import *\n_result = str(log({val}, {base}))"

    # ── GCD / LCM ──
    m = re.search(r"gcd\s+of\s+(\d+)\s+and\s+(\d+)", p)
    if m: return f"from sympy import *\n_result = str(gcd({m.group(1)}, {m.group(2)}))"
    m = re.search(r"lcm\s+of\s+(\d+)\s+and\s+(\d+)", p)
    if m: return f"from sympy import *\n_result = str(lcm({m.group(1)}, {m.group(2)}))"

    # ── Is prime ──
    m = re.search(r"is\s+(\d+)\s+prime", p)
    if m: return f"from sympy import *\n_result = str(isprime({m.group(1)}))"

    # ── Prime factorization ──
    m = re.search(r"prime\s+factor(?:ization|s)\s+of\s+(\d+)", p)
    if m: return f"from sympy import *\n_result = str(factorint({m.group(1)}))"

    # ── Normal distribution probability ──
    m = re.search(r"normal\s+(?:distribution|prob|cdf)\s+mean\s+(-?\d+)\s+std\s+(\d+)\s+x\s*(?:<=?|<)\s*(-?\d+)", p)
    if m:
        mu, sigma, xval = m.group(1), m.group(2), m.group(3)
        return f"from sympy import *\n_result = str(N(Erf(sqrt(pi/2)*(1-erf(({xval}-{mu})/({sigma}*sqrt(2)))))/2))"

    return None


# ──────────────────────────────────────────────
# 3. WEB SEARCH
# ──────────────────────────────────────────────

def search_web(theorem):
    title = theorem.replace(" ", "_").replace("(", "").replace(")", "")
    return f"https://en.wikipedia.org/wiki/{title}"


# ──────────────────────────────────────────────
# 4. GRAPH PRESENTER
# ──────────────────────────────────────────────

def present_graph(problem):
    p = problem.lower().strip()

    # ── 3D surface ──
    m = re.search(r"(?:3d\s+graph|surface\s+plot)\s+z\s*=\s*(.+?)(?:\s+from\s+(-?[0-9]+)\s+to\s+(-?[0-9]+))?\s*$", p)
    if m:
        a = int(m.group(2)) if m.group(2) else -5
        b = int(m.group(3)) if m.group(3) else 5
        return _surface(m.group(1).strip(), a, b)

    # ── 3D parametric ──
    m = re.search(r"parametric\s+3d\s+x\s*=\s*(.+?)\s+y\s*=\s*(.+?)\s+z\s*=\s*(.+?)(?:\s+from\s+(-?[0-9]+)\s+to\s+(-?[0-9]+))?\s*$", p)
    if m:
        return _parametric3d(m.group(1).strip(), m.group(2).strip(), m.group(3).strip(),
                             int(m.group(4)) if m.group(4) else -5,
                             int(m.group(5)) if m.group(5) else 5)

    # ── Contour plot ──
    m = re.search(r"contour\s+plot\s+(?:of\s+)?z\s*=\s*(.+?)(?:\s+from\s+(-?[0-9]+)\s+to\s+(-?[0-9]+))?\s*$", p)
    if m:
        a = int(m.group(2)) if m.group(2) else -5
        b = int(m.group(3)) if m.group(3) else 5
        return _contour(m.group(1).strip(), a, b)

    # ── Polar plot: "polar plot r = ..." ──
    m = re.search(r"polar\s+plot\s+r\s*=\s*(.+?)(?:\s+from\s+(-?[0-9]+)\s+to\s+(-?[0-9]+))?\s*$", p)
    if m:
        a = int(m.group(2)) if m.group(2) else 0
        b = int(m.group(3)) if m.group(3) else 12
        return _polar(m.group(1).strip(), a, b)

    # ── Parametric 2D: "parametric x = f(t) y = g(t) from a to b" ──
    m = re.search(r"parametric\s+(?:2d\s+)?x\s*=\s*(.+?)\s+y\s*=\s*(.+?)(?:\s+from\s+(-?[0-9]+)\s+to\s+(-?[0-9]+))?\s*$", p)
    if m:
        return _parametric2d(m.group(1).strip(), m.group(2).strip(),
                             int(m.group(3)) if m.group(3) else 0,
                             int(m.group(4)) if m.group(4) else 10)

    # ── Bar chart with labels: "bar chart of [labels]: [values]" (check first) ──
    m = re.search(r"bar\s+chart\s+of\s+\[([^\]]+)\]\s*:\s*\[([^\]]+)\]", p)
    if m:
        labels = [x.strip().strip("'\"") for x in m.group(1).split(",")]
        values = [float(x) for x in m.group(2).split(",")]
        return _bar(labels, values)

    # ── Bar chart from comma-separated values (simple) ──
    m = re.search(r"bar\s+chart\s+(?:of\s+)?\[([^\]]+)\]", p)
    if m:
        values = [float(x) for x in m.group(1).split(",")]
        labels = [str(i+1) for i in range(len(values))]
        return _bar(labels, values)

    # ── Box plot: "box plot of [data]" ──
    m = re.search(r"box\s+plot\s+(?:of\s+)?\[([^\]]+)\]", p)
    if m:
        return _box([float(x) for x in m.group(1).split(",")])

    # ── Area/filled plot: "area plot y = ..." ──
    m = re.search(r"area\s+plot\s+y\s*=\s*(.+?)(?:\s+from\s+(-?[0-9]+)\s+to\s+(-?[0-9]+))?\s*$", p)
    if m:
        a = int(m.group(2)) if m.group(2) else -10
        b = int(m.group(3)) if m.group(3) else 10
        return _area(m.group(1).strip(), a, b)

    # ── Multi-line graph ──
    m = re.search(r"graph\s+y\s*=\s*(.+?)\s+and\s+y\s*=\s*(.+?)\s+from\s+(-?[0-9]+)\s+to\s+(-?[0-9]+)", p)
    if m:
        return _multi(m.group(1).strip(), m.group(2).strip(), int(m.group(3)), int(m.group(4)))
    m = re.search(r"graph\s+(.+?)\s+and\s+(.+?)\s+from\s+(-?[0-9]+)\s+to\s+(-?[0-9]+)", p)
    if m:
        return _multi(m.group(1).strip(), m.group(2).strip(), int(m.group(3)), int(m.group(4)))

    # ── Multi-line without range ──
    m = re.search(r"graph\s+y\s*=\s*(.+?)\s+and\s+y\s*=\s*(.+?)\s*$", p)
    if m:
        return _multi(m.group(1).strip(), m.group(2).strip(), -10, 10)

    # ── Single line graph with range ──
    m = re.search(r"graph\s*:?\s*y\s*=\s*(.+?)\s+from\s+(-?[0-9]+)\s+to\s+(-?[0-9]+)", p)
    if m:
        return _line(m.group(1).strip(), int(m.group(2)), int(m.group(3)))

    # ── Single line graph no range — auto-range ──
    m = re.search(r"graph\s*:?\s*y\s*=\s*(.+?)\s*$", p)
    if m:
        return _line(m.group(1).strip(), -10, 10)

    # ── "plot y = ..." (alias for graph) ──
    m = re.search(r"plot\s+y\s*=\s*(.+?)\s+from\s+(-?[0-9]+)\s+to\s+(-?[0-9]+)", p)
    if m:
        return _line(m.group(1).strip(), int(m.group(2)), int(m.group(3)))
    m = re.search(r"plot\s+y\s*=\s*(.+?)\s*$", p)
    if m:
        return _line(m.group(1).strip(), -10, 10)

    # ── Histogram ──
    m = re.search(r"histogram\s+of\s+\[([^\]]+)\]", p)
    if m:
        return _hist([float(x) for x in m.group(1).split(",")])

    # ── Scatter ──
    m = re.search(r"scatter\s+plot\s+of\s+\[([^\]]+)\]|scatter\s+\[([^\]]+)\]", p)
    if m:
        nums = [float(x) for x in (m.group(1) or m.group(2)).split(",")]
        return _scatter(nums)

    # ── Scatter with x,y pairs: "scatter [[x1,y1],[x2,y2],...]" ──
    m = re.search(r"scatter\s+(?:plot\s+)?(\[\[.*?\]\])", p)
    if m:
        import json as _json
        pairs = _json.loads(m.group(1).replace(" ", ""))
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        return _scatter_xy(xs, ys)

    return None


# ── Graph helpers ──

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


def _style_ax(ax):
    ax.set_facecolor(C_BG)
    ax.tick_params(colors="#8b949e")
    ax.grid(True, color=C_GRID, alpha=0.3)
    for spine in ax.spines.values(): spine.set_color(C_GRID)
    ax.axhline(0, color=C_GRID, linewidth=0.8)
    ax.axvline(0, color=C_GRID, linewidth=0.8)


def _line(expr_str, a, b, pts=400):
    expr_str = _strip_y(expr_str)
    x = symbols("x")
    expr = sympify(norm(expr_str))
    f = lambdify(x, expr, "numpy")
    xs = np.linspace(a, b, pts)
    ys = np.array([float(f(xi)) for xi in xs])
    # Auto-clip extreme values for better display
    mask = np.abs(ys) < 1e15
    xs, ys = xs[mask], ys[mask]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, ys, color=C_BLUE, linewidth=2.5, label=f"y = {expr_str}")
    ax.set_xlabel("x", fontsize=13, color="#e6edf3")
    ax.set_ylabel("y", fontsize=13, color="#e6edf3")
    ax.set_title(f"y = {expr_str}", fontsize=15, pad=12, color="#e6edf3")
    ax.legend(fontsize=12, facecolor=C_BG, edgecolor=C_GRID, labelcolor="#e6edf3")
    _style_ax(ax)
    fig.tight_layout()
    return {"image": _fig_b64(fig), "title": f"y = {expr_str}"}


def _multi(e1, e2, a, b, pts=400):
    e1, e2 = _strip_y(e1), _strip_y(e2)
    x = symbols("x")
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [C_BLUE, C_PURPLE, C_GREEN, C_ORANGE]
    for i, e in enumerate([e1, e2][:4]):
        try:
            expr = sympify(norm(e))
            f = lambdify(x, expr, "numpy")
            xs = np.linspace(a, b, pts)
            ys = np.array([float(f(xi)) for xi in xs])
            mask = np.abs(ys) < 1e15
            ax.plot(xs[mask], ys[mask], color=colors[i % len(colors)], linewidth=2.5, label=f"y = {e}")
        except: pass
    ax.set_xlabel("x", fontsize=13, color="#e6edf3")
    ax.set_ylabel("y", fontsize=13, color="#e6edf3")
    ax.set_title("Graph Comparison", fontsize=15, pad=12, color="#e6edf3")
    ax.legend(fontsize=12, facecolor=C_BG, edgecolor=C_GRID, labelcolor="#e6edf3")
    _style_ax(ax)
    fig.tight_layout()
    return {"image": _fig_b64(fig), "title": "Graph Comparison"}


def _hist(data):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(data, bins=min(len(data), 12), color=C_BLUE, alpha=0.85, edgecolor=C_PURPLE)
    ax.set_xlabel("Value", fontsize=13, color="#e6edf3")
    ax.set_ylabel("Count", fontsize=13, color="#e6edf3")
    ax.set_title("Histogram", fontsize=15, pad=12, color="#e6edf3")
    ax.set_facecolor(C_BG)
    ax.tick_params(colors="#8b949e")
    ax.grid(True, color=C_GRID, alpha=0.3, axis="y")
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
    _style_ax(ax)
    fig.tight_layout()
    return {"image": _fig_b64(fig), "title": "Scatter Plot"}


def _scatter_xy(xs, ys):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(xs, ys, s=100, color=C_GREEN, edgecolor=C_BLUE, linewidth=1.5, zorder=5)
    ax.set_xlabel("x", fontsize=13, color="#e6edf3")
    ax.set_ylabel("y", fontsize=13, color="#e6edf3")
    ax.set_title("Scatter Plot (XY)", fontsize=15, pad=12, color="#e6edf3")
    _style_ax(ax)
    fig.tight_layout()
    return {"image": _fig_b64(fig), "title": "Scatter Plot (XY)"}


def _surface(expr_str, a, b, pts=80):
    x, y = symbols("x y")
    expr = sympify(norm(expr_str))
    f = lambdify((x, y), expr, "numpy")
    xv = np.linspace(a, b, pts)
    yv = np.linspace(a, b, pts)
    X, Y = np.meshgrid(xv, yv)
    Z = f(X, Y)
    if not isinstance(Z, np.ndarray): Z = np.full_like(X, float(Z))
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.set_xlabel("x", fontsize=11, color="#e6edf3")
    ax.set_ylabel("y", fontsize=11, color="#e6edf3")
    ax.set_zlabel("z", fontsize=11, color="#e6edf3")
    ax.set_title(f"z = {expr_str}", fontsize=14, pad=12, color="#e6edf3")
    ax.tick_params(colors="#8b949e")
    fig.tight_layout()
    return {"image": _fig_b64(fig), "title": f"3D Surface: z = {expr_str}"}


def _parametric3d(ex_str, ey_str, ez_str, a, b, pts=400):
    t = symbols("t")
    ex, ey, ez = sympify(norm(ex_str)), sympify(norm(ey_str)), sympify(norm(ez_str))
    fx, fy, fz = lambdify(t, ex, "numpy"), lambdify(t, ey, "numpy"), lambdify(t, ez, "numpy")
    tv = np.linspace(a, b, pts)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(fx(tv), fy(tv), fz(tv), color=C_BLUE, linewidth=2.5)
    ax.set_xlabel("x", fontsize=11, color="#e6edf3")
    ax.set_ylabel("y", fontsize=11, color="#e6edf3")
    ax.set_zlabel("z", fontsize=11, color="#e6edf3")
    ax.set_title("Parametric 3D Curve", fontsize=14, pad=12, color="#e6edf3")
    ax.tick_params(colors="#8b949e")
    fig.tight_layout()
    return {"image": _fig_b64(fig), "title": "Parametric 3D Curve"}


def _contour(expr_str, a, b, pts=100):
    x, y = symbols("x y")
    expr = sympify(norm(expr_str))
    f = lambdify((x, y), expr, "numpy")
    xv = np.linspace(a, b, pts)
    yv = np.linspace(a, b, pts)
    X, Y = np.meshgrid(xv, yv)
    Z = f(X, Y)
    if not isinstance(Z, np.ndarray): Z = np.full_like(X, float(Z))
    fig, ax = plt.subplots(figsize=(9, 7))
    cs = ax.contourf(X, Y, Z, levels=20, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="z")
    ax.set_xlabel("x", fontsize=13, color="#e6edf3")
    ax.set_ylabel("y", fontsize=13, color="#e6edf3")
    ax.set_title(f"Contour: z = {expr_str}", fontsize=15, pad=12, color="#e6edf3")
    ax.set_facecolor(C_BG)
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values(): spine.set_color(C_GRID)
    fig.tight_layout()
    return {"image": _fig_b64(fig), "title": f"Contour: z = {expr_str}"}


def _polar(expr_str, a, b, pts=400):
    expr_str = expr_str.replace("x", "t")  # normalize variable to t
    t = symbols("t")
    expr = sympify(norm(expr_str))
    f = lambdify(t, expr, modules=["numpy"])
    tv = np.linspace(a, b, pts)
    try:
        r = np.array(f(tv), dtype=float)
    except:
        r = np.array([float(expr.subs(t, float(ti)).evalf()) for ti in tv])
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    ax.plot(tv, r, color=C_BLUE, linewidth=2.5)
    ax.set_title(f"r = {expr_str}", fontsize=15, pad=20, color="#e6edf3")
    ax.tick_params(colors="#8b949e")
    ax.grid(True, color=C_GRID, alpha=0.4)
    fig.tight_layout()
    return {"image": _fig_b64(fig), "title": f"Polar: r = {expr_str}"}


def _parametric2d(ex_str, ey_str, a, b, pts=400):
    t = symbols("t")
    ex, ey = sympify(norm(ex_str)), sympify(norm(ey_str))
    fx, fy = lambdify(t, ex, "numpy"), lambdify(t, ey, "numpy")
    tv = np.linspace(a, b, pts)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fx(tv), fy(tv), color=C_BLUE, linewidth=2.5)
    ax.set_xlabel("x", fontsize=13, color="#e6edf3")
    ax.set_ylabel("y", fontsize=13, color="#e6edf3")
    ax.set_title("Parametric Curve", fontsize=15, pad=12, color="#e6edf3")
    _style_ax(ax)
    fig.tight_layout()
    return {"image": _fig_b64(fig), "title": "Parametric Curve"}


def _bar(labels, values):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [C_BLUE, C_PURPLE, C_GREEN, C_ORANGE, C_RED, "#58a6ff", "#bc8cff"]
    bar_colors = [colors[i % len(colors)] for i in range(len(values))]
    ax.bar(labels, values, color=bar_colors, edgecolor=C_GRID, linewidth=0.8)
    ax.set_xlabel("Category", fontsize=13, color="#e6edf3")
    ax.set_ylabel("Value", fontsize=13, color="#e6edf3")
    ax.set_title("Bar Chart", fontsize=15, pad=12, color="#e6edf3")
    ax.set_facecolor(C_BG)
    ax.tick_params(colors="#8b949e")
    ax.grid(True, color=C_GRID, alpha=0.3, axis="y")
    for spine in ax.spines.values(): spine.set_color(C_GRID)
    fig.tight_layout()
    return {"image": _fig_b64(fig), "title": "Bar Chart"}


def _box(data):
    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(data, patch_artist=True, widths=0.5)
    for patch in bp["boxes"]:
        patch.set_facecolor(C_BLUE)
        patch.set_alpha(0.7)
        patch.set_edgecolor(C_PURPLE)
    for whisker in bp["whiskers"]: whisker.set_color(C_GRID)
    for cap in bp["caps"]: cap.set_color(C_GRID)
    for median in bp["medians"]: median.set_color(C_ORANGE)
    ax.set_ylabel("Value", fontsize=13, color="#e6edf3")
    ax.set_title("Box Plot", fontsize=15, pad=12, color="#e6edf3")
    ax.set_facecolor(C_BG)
    ax.tick_params(colors="#8b949e")
    ax.grid(True, color=C_GRID, alpha=0.3, axis="y")
    for spine in ax.spines.values(): spine.set_color(C_GRID)
    fig.tight_layout()
    return {"image": _fig_b64(fig), "title": "Box Plot"}


def _area(expr_str, a, b, pts=400):
    expr_str = _strip_y(expr_str)
    x = symbols("x")
    expr = sympify(norm(expr_str))
    f = lambdify(x, expr, "numpy")
    xs = np.linspace(a, b, pts)
    ys = np.array([float(f(xi)) for xi in xs])
    mask = np.abs(ys) < 1e15
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(xs[mask], ys[mask], alpha=0.4, color=C_BLUE)
    ax.plot(xs[mask], ys[mask], color=C_BLUE, linewidth=2)
    ax.set_xlabel("x", fontsize=13, color="#e6edf3")
    ax.set_ylabel("y", fontsize=13, color="#e6edf3")
    ax.set_title(f"y = {expr_str} (area)", fontsize=15, pad=12, color="#e6edf3")
    _style_ax(ax)
    fig.tight_layout()
    return {"image": _fig_b64(fig), "title": f"Area: y = {expr_str}"}


# ──────────────────────────────────────────────
# 5. MEMORY
# ──────────────────────────────────────────────

def memory_format(problem, answer, confidence):
    return {"problem": problem, "answer": answer, "confidence": confidence}


# ──────────────────────────────────────────────
# 6. DOUBLE CHECK
# ──────────────────────────────────────────────

def double_check(problem, llm_answer):
    code = generate_code(problem)
    if not code:
        return {"verified": None, "reason": "Python cannot verify this problem type"}
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
        return None
    if len(nums_a) != len(nums_b):
        return False
    try:
        for na, nb in zip(nums_a, nums_b):
            if abs(float(na) - float(nb)) >= 0.01:
                return False
        return True
    except:
        return False


# ──────────────────────────────────────────────
# 7. THREE AGENTS
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
    # Try explicit graph detection first
    graph = present_graph(problem)
    if graph:
        result["graph_image"] = graph["image"]
        result["graph_title"] = graph.get("title", "Graph")
    else:
        # Auto-detect: if the problem contains "y =" or "f(x) =" and looks
        # like a function (not "solve" or "derivative" etc), try to graph it
        p = problem.lower().strip()
        is_graphable = bool(re.search(r"y\s*=\s*[0-9a-z\+\-\*\^/·×\s\(\)sincoxtan]+", p) or
                           re.search(r"f\(x\)\s*=\s*[0-9a-z\+\-\*\^/·×\s\(\)sincoxtan]+", p))
        is_computation = any(kw in p for kw in [
            "derivative", "integral", "solve", "limit", "factorial", "choose",
            "determinant", "inverse", "transpose", "eigenvalue", "trace", "rank",
            "taylor", "ode", "differential", "mean of", "median of", "std of",
            "variance of", "mode of", "permutation", "simplify", "expand",
            "factor ", "gcd", "lcm", "prime", "sum of", "log base",
        ])
        if is_graphable and not is_computation:
            try:
                # Extract the expression after y = or f(x) =
                m = re.search(r"y\s*=\s*(.+?)(?:\s+from\s+(-?[0-9]+)\s+to\s+(-?[0-9]+))?\s*$", p)
                if not m:
                    m = re.search(r"f\(x\)\s*=\s*(.+?)(?:\s+from\s+(-?[0-9]+)\s+to\s+(-?[0-9]+))?\s*$", p)
                if m:
                    expr = m.group(1).strip().rstrip(".")
                    a = int(m.group(2)) if m.group(2) else -10
                    b = int(m.group(3)) if m.group(3) else 10
                    graph = _line(expr, a, b)
                    if graph and "image" in graph:
                        result["graph_image"] = graph["image"]
                        result["graph_title"] = graph.get("title", "Graph")
            except Exception:
                pass  # Not graphable, just solve with LLM
    if "graph_image" in result:
        return json.dumps(result)
    return None