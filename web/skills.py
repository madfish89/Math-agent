import re as _re
from sympy import *
from sympy import solve as sp_solve
import numpy as np
re = _re

def norm(s):
    s = s.strip().replace("^", "**")
    for f in ["sin","cos","tan","exp","log","sqrt","abs","ln","sec","csc","cot","asin","acos","atan","sinh","cosh","tanh","erf","gamma"]:
        s = s.replace(f+"(", "\x00"+f+"\x01")
    s = re.sub(r"([0-9])([a-zA-Z])", r"\1*\2", s)
    s = re.sub(r"\)([a-zA-Z0-9])", r")*\1", s)
    s = re.sub(r"([a-zA-Z])\(", r"\1*(", s)
    for f in ["sin","cos","tan","exp","log","sqrt","abs","ln","sec","csc","cot","asin","acos","atan","sinh","cosh","tanh","erf","gamma"]:
        s = s.replace("\x00"+f+"\x01", f+"(")
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

def graph_data(expr_str, multi=None, xr=(-10,10), pts=200):
    x = symbols("x")
    traces = []
    for e in (multi or [expr_str]):
        expr = sympify(norm(e))
        f = lambdify(x, expr, "numpy")
        xs = np.linspace(xr[0], xr[1], pts)
        ys = [float(f(xi)) for xi in xs]
        traces.append({"x": xs.tolist(), "y": ys, "name": e, "type": "scatter", "mode": "lines"})
    return {"traces": traces, "title": "Graph", "x_range": list(xr)}

def solve(problem):
    p = problem.lower().strip()

    def out(answer, steps, conf, graph=None):
        r = {"answer": fmt(answer), "steps": [{"step": fmt(s["step"]), "explanation": s.get("explanation",""), "proof": s.get("proof","")} for s in steps], "confidence": conf}
        if graph: r["graph_data"] = graph
        return r

    # Graph
    m = re.search(r"graph\s+y\s*=\s*(.+?)\s+from\s+(-?[0-9]+)\s+to\s+(-?[0-9]+)", p)
    if m:
        e, a, b = m.group(1).strip(), int(m.group(2)), int(m.group(3))
        x = symbols("x"); expr = sympify(norm(e))
        return out(str(expr), [{"step": f"f(x) = {expr}", "explanation": "Function to graph", "proof": "Given"},
                               {"step": f"f'(x) = {diff(expr,x)}", "explanation": "Derivative for critical points", "proof": "Power Rule"},
                               {"step": f"Graph from {a} to {b}", "explanation": "Plot the function", "proof": "Visual analysis"}], 0.95, graph_data(e, xr=(a,b)))
    m = re.search(r"graph\s+(.+?)\s+and\s+(.+?)\s+from\s+(-?[0-9]+)\s+to\s+(-?[0-9]+)", p)
    if m:
        e1, e2, a, b = m.group(1).strip(), m.group(2).strip(), int(m.group(3)), int(m.group(4))
        return out(f"{e1} and {e2}", [{"step": f"f₁ = {e1}", "explanation": "First function", "proof": "Given"},
                                       {"step": f"f₂ = {e2}", "explanation": "Second function", "proof": "Given"},
                                       {"step": f"Graph both {a} to {b}", "explanation": "Compare on same axes", "proof": "Visual comparison"}], 0.95, graph_data(None, multi=[e1,e2], xr=(a,b)))
    m = re.search(r"graph\s+y\s*=\s*(.+)", p)
    if m:
        e = m.group(1).strip(); x = symbols("x"); expr = sympify(norm(e))
        return out(str(expr), [{"step": f"f(x) = {expr}", "explanation": "Function to graph", "proof": "Given"},
                               {"step": f"f'(x) = {diff(expr,x)}", "explanation": "Derivative", "proof": "Power Rule"},
                               {"step": "Graph", "explanation": "Plot from -10 to 10", "proof": "Visual analysis"}], 0.95, graph_data(e))

    # Eigenvalues
    m = re.search(r"eigenvalue.*\[([^\]]+),\s*([^\]]+)\]" , p) or re.search(r"\[([^\]]+),\s*([^\]]+)\].*eigenvalue", p)
    if m:
        r1 = [int(float(x)) for x in m.group(1).split(",")]; r2 = [int(float(x)) for x in m.group(2).split(",")]
        lam = symbols("l"); A = Matrix([r1,r2]); cp = A.charpoly(lam); ev = sp_solve(cp, lam)
        return out(str(ev), [{"step": f"A = {A}", "explanation": "Given matrix", "proof": "Eigenvalues λ satisfy Av = λv"},
                             {"step": f"det(A - λI) = 0 → {cp.as_expr()}", "explanation": "Characteristic equation", "proof": "det([[a,b],[c,d]]) = ad-bc"},
                             {"step": f"λ = {ev}", "explanation": "Solve polynomial", "proof": "Zero Product Property"}], 0.95)

    # Partial derivative
    m = re.search(r"partial\s+derivative\s+of\s+(.+?)\s+with\s+respect\s+to\s+([a-zA-Z]+)", p)
    if m:
        v = m.group(2).strip(); vs = symbols(v); expr = sympify(norm(m.group(1).strip())); pd = diff(expr, vs)
        return out(str(pd), [{"step": f"∂f/∂{v} = {pd}", "explanation": f"Differentiate w.r.t. {v}", "proof": "Partial differentiation"}], 0.95)

    # Derivative
    m = re.search(r"derivative\s+of\s+(.+)", p) or re.search(r"differentiate\s+(.+)", p)
    if m:
        x = symbols("x"); expr = sympify(norm(m.group(1).strip().rstrip("."))); d = diff(expr, x)
        return out(str(d), [{"step": f"d/dx({expr}) = {d}", "explanation": "Apply differentiation rules", "proof": "Power Rule: d/dx(xⁿ) = n·xⁿ⁻¹"}], 0.95)

    # Integral
    m = re.search(r"integral\s+of\s+(.+)|integrate\s+(.+)", p)
    if m:
        raw = (m.group(1) or m.group(2)).strip(); mi = re.search(r"(.+?)\s+from\s+(\S+)\s+to\s+(\S+)", raw); x = symbols("x")
        if mi:
            expr = sympify(norm(mi.group(1).strip())); a, b = sympify(mi.group(2).strip()), sympify(mi.group(3).strip())
            r = integrate(expr, (x, a, b))
            return out(str(r), [{"step": f"∫{expr}dx = {integrate(expr,x)}", "explanation": "Antiderivative", "proof": "Power Rule for Integration"},
                                {"step": f"F({b})-F({a}) = {r}", "explanation": "Evaluate at bounds", "proof": "FTC: ∫ₐᵇf(x)dx = F(b)-F(a)"}], 0.95)
        expr = sympify(norm(raw)); anti = integrate(expr, x)
        return out(str(anti), [{"step": f"∫{expr}dx = {anti} + C", "explanation": "Antiderivative", "proof": "Power Rule for Integration"}], 0.95)

    # Solve equation
    m = re.search(r"solve\s+(.+)", p)
    if m:
        e = norm(m.group(1).strip())
        if "=" in e: parts = e.split("="); e = f"({parts[0]})-({parts[1]})"
        x = symbols("x"); eq = sympify(e); sol = sp_solve(eq, x)
        return out(str(sol), [{"step": f"Solve {eq} = 0", "explanation": "Find all roots", "proof": "Definition"},
                              {"step": f"x = {sol}", "explanation": "All solutions", "proof": "Zero Product Property"}], 0.95)

    # Limit
    m = re.search(r"limit\s+of\s+(.+?)\s+as\s+([a-zA-Z]+)\s+approaches\s+(.+)", p)
    if m:
        v, pt = m.group(2).strip(), m.group(3).strip(); vs = symbols(v)
        expr = sympify(norm(m.group(1).strip())); lim = limit(expr, vs, sympify(pt))
        return out(str(lim), [{"step": f"lim({v}→{pt}) {expr}", "explanation": f"As {v} approaches {pt}", "proof": "Direct Substitution"},
                              {"step": f"= {lim}", "explanation": "Apply limit rules", "proof": "L'Hôpital's Rule"}], 0.95)

    # Taylor series
    m = re.search(r"taylor\s+series\s+of\s+(.+?)\s+around\s+([a-zA-Z]+)\s*=\s*(.+)", p) or re.search(r"taylor\s+series\s+of\s+(.+)", p)
    if m:
        v = m.group(2) if len(m.groups()) > 1 else "x"; pt = m.group(3).strip() if len(m.groups()) > 2 else "0"
        vs = symbols(v); expr = sympify(norm(m.group(1).strip())); ts = series(expr, vs, sympify(pt), n=5)
        return out(str(ts), [{"step": f"f({v}) = {expr}", "explanation": f"Expand around {v}={pt}", "proof": "Taylor Series: f(x) = Σf⁽ⁿ⁾(a)/n!·(x-a)ⁿ"},
                              {"step": f"= {ts}", "explanation": "To order 5", "proof": "Converges within radius"}], 0.95)

    # Combinations
    m = re.search(r"([0-9]+)\s+choose\s+([0-9]+)", p)
    if m:
        n, k = int(m.group(1)), int(m.group(2)); r = binomial(n, k)
        return out(str(r), [{"step": f"C({n},{k}) = {n}!/({k}!×({n}-{k})!) = {r}", "explanation": "Ways to choose k from n", "proof": "C = P/k!"}], 0.95)

    # Permutations
    m = re.search(r"([0-9]+)\s+permutations?\s+of\s+([0-9]+)", p)
    if m:
        n, k = int(m.group(1)), int(m.group(2)); r = factorial(n)//factorial(n-k)
        return out(str(r), [{"step": f"P({n},{k}) = {n}!/({n}-{k})! = {r}", "explanation": "Ordered arrangements", "proof": "Multiplication Principle"}], 0.95)

    # Factorial
    m = re.search(r"factorial\s+of\s+([0-9]+)|([0-9]+)!", p)
    if m:
        n = int(m.group(1) or m.group(2)); r = factorial(n)
        return out(str(r), [{"step": f"{n}! = {r}", "explanation": f"Product 1 to {n}", "proof": "n! = n×(n-1)×...×1"}], 0.95)

    # Mean
    m = re.search(r"(?:mean|average)\s+of\s+\[([^\]]+)\]", p)
    if m:
        d = [float(x) for x in m.group(1).split(",")]; v = sum(d)/len(d)
        return out(str(v), [{"step": f"Mean = {v}", "explanation": f"Sum / count = {sum(d)}/{len(d)}", "proof": "x̄ = (1/n)Σxᵢ"}], 0.95)

    # Median
    m = re.search(r"median\s+of\s+\[([^\]]+)\]", p)
    if m:
        d = sorted(float(x) for x in m.group(1).split(",")); n = len(d)
        med = d[n//2] if n%2 else (d[n//2-1]+d[n//2])/2
        return out(str(med), [{"step": f"Median = {med}", "explanation": "Middle of sorted data", "proof": "Median definition"}], 0.95)

    # Std dev
    m = re.search(r"(?:standard\s+deviation|std\s*dev|variance)\s+of\s+\[([^\]]+)\]", p)
    if m:
        import math
        d = [float(x) for x in m.group(1).split(",")]; n = len(d)
        mv = sum(d)/n; var = sum((x-mv)**2 for x in d)/(n-1); sd = math.sqrt(var)
        return out(str(round(sd,4)), [{"step": f"Mean = {mv}", "explanation": "Average", "proof": "x̄ = (1/n)Σxᵢ"},
                                       {"step": f"Variance = {var:.4f}", "explanation": "Squared deviations / (n-1)", "proof": "s² = (1/(n-1))Σ(xᵢ-x̄)²"},
                                       {"step": f"Std dev = {sd:.4f}", "explanation": "√variance", "proof": "s = √s²"}], 0.95)

    # Determinant
    m = re.search(r"determinant\s+of\s*(?:matrix\s*)?\[([^\]]+),\s*([^\]]+)\]", p)
    if m:
        r1 = [int(float(x)) for x in m.group(1).split(",")]; r2 = [int(float(x)) for x in m.group(2).split(",")]
        A = Matrix([r1,r2]); d = A.det()
        return out(str(d), [{"step": f"det({A}) = {d}", "explanation": "ad - bc", "proof": "det([[a,b],[c,d]]) = ad-bc"}], 0.95)

    # Gradient
    m = re.search(r"gradient\s+of\s+(.+)", p)
    if m:
        x,y,z = symbols("x y z"); expr = sympify(norm(m.group(1).strip()))
        vs = [v for v in [x,y,z] if v in expr.free_symbols]; g = [diff(expr,v) for v in vs]
        return out(str(g), [{"step": f"∇f = {g}", "explanation": "Partial derivatives", "proof": "∇f = (∂f/∂x, ∂f/∂y)"}], 0.95)

    return None