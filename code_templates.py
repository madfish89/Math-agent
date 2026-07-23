import re
from sympy import sympify, symbols, diff, integrate, solve, Matrix, Rational, sqrt, pi, sin, cos, tan, exp, log, limit, Symbol, Function, dsolve, Eq, oo, factorial, binomial, series, summation, Sum, gamma, erf, E, S


def detect_and_solve(problem):
    """Try to detect problem type and generate code directly. Returns code string or None."""
    p = problem.lower().strip()

    # Eigenvalues
    m = re.search(r'eigenvalue.*matrix\s*\[\s*\[([^\]]+)\]\s*,\s*\[([^\]]+)\]\s*\]', p)
    if not m:
        m = re.search(r'matrix\s*\[\s*\[([^\]]+)\]\s*,\s*\[([^\]]+)\]\s*\].*eigenvalue', p)
    if m:
        r1 = [float(x.strip()) for x in m.group(1).split(',')]
        r2 = [float(x.strip()) for x in m.group(2).split(',')]
        return _eigenvalue_code(r1, r2)

    # Partial derivative (must check BEFORE regular derivative)
    m = re.search(r'partial\s+derivative\s+of\s+(.+?)\s+with\s+respect\s+to\s+(\w+)', p)
    if m:
        return _partial_derivative_code(m.group(1).strip(), m.group(2).strip())

    # Derivative
    m = re.search(r'derivative\s+of\s+(.+)', p)
    if m:
        expr_str = m.group(1).rstrip('.').strip()
        return _derivative_code(expr_str)

    m = re.search(r'diff(?:erentiate)?\s+(.+)', p)
    if m:
        expr_str = m.group(1).rstrip('.').strip()
        return _derivative_code(expr_str)

    # Integral
    m = re.search(r'integral\s+of\s+(.+)', p)
    if m:
        expr_str = m.group(1).rstrip('.').strip()
        return _integral_code(expr_str)

    m = re.search(r'integrate\s+(.+)', p)
    if m:
        expr_str = m.group(1).rstrip('.').strip()
        return _integral_code(expr_str)

    # Solve equation
    m = re.search(r'solve\s+(.+)', p)
    if m:
        expr_str = m.group(1).rstrip('.').strip()
        return _solve_code(expr_str)

    # Limit
    m = re.search(r'limit\s+of\s+(.+?)\s+as\s+(\w+)\s+approaches\s+(.+)', p)
    if m:
        expr_str = m.group(1).strip()
        var = m.group(2).strip()
        point = m.group(3).strip()
        return _limit_code(expr_str, var, point)

    # --- NEW TEMPLATES ---

    # Differential equation
    m = re.search(r'(?:solve\s+)?(?:the\s+)?differential\s+equation\s+(.+)', p)
    if m:
        eq_str = m.group(1).rstrip('.').strip()
        return _differential_eq_code(eq_str)

    # Series convergence
    m = re.search(r'(?:does\s+|is\s+)?the\s+series\s+(.+?)\s+(?:converge|convergent|diverge|divergent)', p)
    if m:
        series_str = m.group(1).strip()
        return _series_convergence_code(series_str)
    m = re.search(r'convergence\s+of\s+(?:the\s+)?series\s+(.+)', p)
    if m:
        series_str = m.group(1).rstrip('.').strip()
        return _series_convergence_code(series_str)

    # Taylor/Laurent series
    m = re.search(r'taylor\s+series\s+of\s+(.+?)\s+around\s+(\w+)\s*=\s*(.+)', p)
    if m:
        return _taylor_series_code(m.group(1).strip(), m.group(2).strip(), m.group(3).strip())
    m = re.search(r'taylor\s+series\s+of\s+(.+)', p)
    if m:
        return _taylor_series_code(m.group(1).rstrip('.').strip(), 'x', '0')

    # Matrix multiplication
    m = re.search(r'(?:multiply|product|matmul|matrix\s+multiplication)\s+(.+?)\s*(?:and|times|\*|x)\s*(.+)', p)
    if not m:
        m = re.search(r'matrix\s*\[\s*\[([^\]]+)\]\s*,\s*\[([^\]]+)\]\s*\]\s*(?:times|\*|x|multiplied\s+by)\s*matrix\s*\[\s*\[([^\]]+)\]\s*,\s*\[([^\]]+)\]\s*\]', p)
    if m:
        return _matrix_mult_code(m)

    # Gradient
    m = re.search(r'gradient\s+of\s+(.+)', p)
    if m:
        expr_str = m.group(1).rstrip('.').strip()
        return _gradient_code(expr_str)

    # Divergence
    m = re.search(r'divergence\s+of\s+(.+)', p)
    if m:
        expr_str = m.group(1).rstrip('.').strip()
        return _divergence_code(expr_str)

    # Curl
    m = re.search(r'curl\s+of\s+(.+)', p)
    if m:
        expr_str = m.group(1).rstrip('.').strip()
        return _curl_code(expr_str)

    # Combinatorics: permutations
    m = re.search(r'(\d+)\s+permutations?\s+of\s+(\d+)', p)
    if m:
        return _permutation_code(int(m.group(1)), int(m.group(2)))
    m = re.search(r'(?:permutations?|p\s*\()\s*(?:of\s+)?(\d+)\s*(?:,|from|choose|out\s+of)\s*(\d+)', p)
    if m:
        return _permutation_code(int(m.group(1)), int(m.group(2)))

    # Combinatorics: combinations
    m = re.search(r'(?:combinations?|c\s*\()\s*(?:of\s+)?(\d+)\s*(?:,|from|choose|out\s+of)\s*(\d+)', p)
    if m:
        return _combination_code(int(m.group(1)), int(m.group(2)))
    m = re.search(r'(\d+)\s+choose\s+(\d+)', p)
    if m:
        return _combination_code(int(m.group(1)), int(m.group(2)))

    # Factorial
    m = re.search(r'factorial\s+of\s+(\d+)|(\d+)!', p)
    if m:
        n = int(m.group(1) or m.group(2))
        return _factorial_code(n)

    # Statistics: mean
    m = re.search(r'(?:mean|average)\s+of\s+(.+)', p)
    if m:
        return _mean_code(m.group(1).rstrip('.').strip())

    # Statistics: median
    m = re.search(r'median\s+of\s+(.+)', p)
    if m:
        return _median_code(m.group(1).rstrip('.').strip())

    # Statistics: standard deviation
    m = re.search(r'(?:standard\s+deviation|std\s*dev|variance)\s+of\s+(.+)', p)
    if m:
        return _stddev_code(m.group(1).rstrip('.').strip())

    # Statistics: normal distribution probability
    m = re.search(r'(?:normal\s+distribution|prob(?:ability)?).*mean\s*=\s*(\S+).*std\s*=\s*(\S+).*x\s*=\s*(\S+)', p)
    if m:
        return _normal_dist_code(float(m.group(1)), float(m.group(2)), float(m.group(3)))

    # System of equations
    if 'system of equations' in p or 'simultaneous' in p:
        return None  # too complex for template

    # Matrix operations
    m = re.search(r'determinant\s+of\s+(?:matrix\s+)?\[\s*\[([^\]]+)\]\s*,\s*\[([^\]]+)\]\s*\]', p)
    if m:
        r1 = [float(x.strip()) for x in m.group(1).split(',')]
        r2 = [float(x.strip()) for x in m.group(2).split(',')]
        return _determinant_code(r1, r2)

    return None


def _normalize_expr(expr_str):
    """Normalize math expressions for sympy: ^ -> **, implicit multiplication."""
    s = expr_str.strip()
    s = s.replace('^', '**')
    # Don't add * for known function names: sin(, cos(, exp(, log(, etc.
    funcs = ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs', 'ln', 'sec', 'csc', 'cot',
             'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'erf', 'gamma', 'floor', 'ceil']
    # Protect function calls from getting * inserted
    for f in funcs:
        s = s.replace(f'{f}(', f'\x00{f}\x01')
    # Insert * between number and variable: 3x -> 3*x
    s = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', s)
    # Insert * between close paren and variable/number: )x -> )*x
    s = re.sub(r'\)([a-zA-Z\d])', r')*\1', s)
    # Insert * between variable and open paren: x( -> x*(  (but not function calls)
    s = re.sub(r'([a-zA-Z])\(', r'\1*(', s)
    # Restore function calls
    for f in funcs:
        s = s.replace(f'\x00{f}\x01', f'{f}(')
    return s


def _derivative_code(expr_str):
    expr_str = _normalize_expr(expr_str)
    return f"""
from sympy import *
x = symbols('x')
expr = sympify('{expr_str}')
deriv = diff(expr, x)

_result = {{
    "answer": str(deriv),
    "steps": [
        {{"step": f"d/dx({{expr}}) = {{deriv}}",
          "explanation": "Apply differentiation rules term by term",
          "proof": "Power Rule: d/dx(x^n) = n*x^(n-1); Chain Rule for composite functions"}}
    ],
    "proofs": [
        {{"theorem": "Power Rule", "statement": "d/dx(x^n) = n*x^(n-1)", "applied_to": "polynomial terms"}},
        {{"theorem": "Sum Rule", "statement": "d/dx(f+g) = d/dx(f) + d/dx(g)", "applied_to": "sum of terms"}},
        {{"theorem": "Chain Rule", "statement": "d/dx(f(g(x))) = f'(g(x))*g'(x)", "applied_to": "composite functions"}}
    ],
    "method": "symbolic_differentiation"
}}
"""


def _integral_code(expr_str):
    expr_str = _normalize_expr(expr_str)
    # Check for definite integral
    m = re.search(r'(.+?)\s+from\s+(\S+)\s+to\s+(\S+)', expr_str)
    if m:
        expr_str2 = m.group(1).strip().replace('^', '**')
        a, b = m.group(2).strip(), m.group(3).strip()
        return f"""
from sympy import *
x = symbols('x')
expr = sympify('{expr_str2}')
a_val = sympify('{a}')
b_val = sympify('{b}')
antideriv = integrate(expr, x)
result = integrate(expr, (x, a_val, b_val))

_result = {{
    "answer": str(result),
    "steps": [
        {{"step": f"Find antiderivative of {{expr}}",
          "explanation": f"integral of {{expr}} dx = {{antideriv}}",
          "proof": "Power Rule for Integration: integral(x^n)dx = x^(n+1)/(n+1) + C"}},
        {{"step": f"Evaluate from {{a_val}} to {{b_val}}",
          "explanation": f"F({{b_val}}) - F({{a_val}}) = {{result}}",
          "proof": "Fundamental Theorem of Calculus: integral_a^b f(x)dx = F(b) - F(a)"}}
    ],
    "proofs": [
        {{"theorem": "Fundamental Theorem of Calculus", "statement": "integral_a^b f(x)dx = F(b) - F(a) where F' = f", "applied_to": "definite integral"}},
        {{"theorem": "Power Rule for Integration", "statement": "integral(x^n)dx = x^(n+1)/(n+1) + C for n != -1", "applied_to": "polynomial terms"}}
    ],
    "method": "definite_integration"
}}
"""
    return f"""
from sympy import *
x = symbols('x')
expr = sympify('{expr_str}')
antideriv = integrate(expr, x)

_result = {{
    "answer": str(antideriv),
    "steps": [
        {{"step": f"integral of {{expr}} dx = {{antideriv}} + C",
          "explanation": "Find antiderivative using integration rules",
          "proof": "Power Rule for Integration: integral(x^n)dx = x^(n+1)/(n+1) + C"}}
    ],
    "proofs": [
        {{"theorem": "Power Rule for Integration", "statement": "integral(x^n)dx = x^(n+1)/(n+1) + C for n != -1", "applied_to": "polynomial terms"}},
        {{"theorem": "Sum Rule for Integration", "statement": "integral(f+g)dx = integral(f)dx + integral(g)dx", "applied_to": "sum of terms"}}
    ],
    "method": "indefinite_integration"
}}
"""


def _solve_code(expr_str):
    expr_str = _normalize_expr(expr_str)
    # Simpler: split on =
    if '=' in expr_str:
        parts = expr_str.split('=')
        expr_str = f"({parts[0]}) - ({parts[1]})"
    return f"""
from sympy import *
x = symbols('x')
eq = sympify('{expr_str}')
solutions = solve(eq, x)

_result = {{
    "answer": str(solutions),
    "steps": [
        {{"step": f"Equation: {{eq}} = 0",
          "explanation": "Set expression equal to zero",
          "proof": "Definition of solving an equation"}},
        {{"step": f"Solutions: {{solutions}}",
          "explanation": "Find all values of x that satisfy the equation",
          "proof": "Zero Product Property: if ab=0 then a=0 or b=0"}}
    ],
    "proofs": [
        {{"theorem": "Zero Product Property", "statement": "If a*b = 0 then a=0 or b=0", "applied_to": "factored equations"}},
        {{"theorem": "Fundamental Theorem of Algebra", "statement": "A polynomial of degree n has exactly n roots (counting multiplicity)", "applied_to": "polynomial equations"}}
    ],
    "method": "symbolic_solving"
}}
"""


def _limit_code(expr_str, var, point):
    expr_str = _normalize_expr(expr_str)
    return f"""
from sympy import *
_{var} = symbols('{var}')
expr = sympify('{expr_str}')
point_val = sympify('{point}')
lim = limit(expr, _{var}, point_val)

_result = {{
    "answer": str(lim),
    "steps": [
        {{"step": f"lim({var}->{{point_val}}) {{expr}}",
          "explanation": f"Evaluate the limit as {var} approaches {point}",
          "proof": "Direct Substitution Property: if f is continuous at a, lim(x->a) f(x) = f(a)"}},
        {{"step": f"Result: {{lim}}",
          "explanation": "Apply limit rules",
          "proof": "L'Hopital's Rule for indeterminate forms (0/0 or inf/inf)"}}
    ],
    "proofs": [
        {{"theorem": "Direct Substitution Property", "statement": "If f is continuous at a, lim(x->a) f(x) = f(a)", "applied_to": "continuous functions"}},
        {{"theorem": "L'Hopital's Rule", "statement": "If lim is 0/0 or inf/inf, then lim f/g = lim f'/g'", "applied_to": "indeterminate forms"}}
    ],
    "method": "limit_computation"
}}
"""


def _eigenvalue_code(r1, r2):
    # Convert to int if possible for cleaner output
    def clean(v):
        try:
            iv = int(v)
            if iv == v:
                return iv
        except (ValueError, TypeError):
            pass
        return v
    r1 = [clean(v) for v in r1]
    r2 = [clean(v) for v in r2]
    return f"""
from sympy import *
lam = symbols('lambda')
A = Matrix([{r1}, {r2}])
char_poly = A.charpoly(lam)
eigenvals = solve(char_poly, lam)
eigenvects = A.eigenvects()

steps = []
steps.append({{"step": f"Matrix A = {{A}}",
    "explanation": "Given matrix",
    "proof": "Definition of the matrix"}})
steps.append({{"step": f"Characteristic equation: det(A - lambda*I) = 0",
    "explanation": "Eigenvalues satisfy det(A - lambda*I) = 0",
    "proof": "Eigenvalue definition: Av = lambda*v, so det(A - lambda*I) = 0"}})
steps.append({{"step": f"Characteristic polynomial: {{char_poly.as_expr()}}",
    "explanation": "Compute determinant of (A - lambda*I)",
    "proof": "Determinant formula for 2x2: det([[a,b],[c,d]]) = ad - bc"}})
steps.append({{"step": f"Eigenvalues: {{eigenvals}}",
    "explanation": "Solve characteristic polynomial = 0",
    "proof": "Zero Product Property and Fundamental Theorem of Algebra"}})

proofs = [
    {{"theorem": "Characteristic Equation", "statement": "det(A - lambda*I) = 0 gives eigenvalues", "applied_to": "matrix A"}},
    {{"theorem": "Determinant of 2x2", "statement": "det([[a,b],[c,d]]) = ad - bc", "applied_to": "characteristic matrix"}},
    {{"theorem": "Fundamental Theorem of Algebra", "statement": "A polynomial of degree n has n roots", "applied_to": "characteristic polynomial"}}
]

_result = {{
    "answer": str(eigenvals),
    "steps": steps,
    "proofs": proofs,
    "method": "eigenvalue_computation"
}}
"""


def _determinant_code(r1, r2):
    return f"""
from sympy import *
A = Matrix([{r1}, {r2}])
det_val = A.det()

_result = {{
    "answer": str(det_val),
    "steps": [
        {{"step": f"det({{A}}) = {{det_val}}",
          "explanation": "Compute determinant using cofactor expansion",
          "proof": "Determinant formula for 2x2: det([[a,b],[c,d]]) = ad - bc"}}
    ],
    "proofs": [
        {{"theorem": "Determinant of 2x2 Matrix", "statement": "det([[a,b],[c,d]]) = ad - bc", "applied_to": "given matrix"}},
        {{"theorem": "Cofactor Expansion", "statement": "det(A) = sum of a_ij * C_ij for any row/column", "applied_to": "general determinant"}}
    ],
    "method": "determinant_computation"
}}
"""


# ========== NEW TEMPLATES ==========

def _differential_eq_code(eq_str):
    eq_str = _normalize_expr(eq_str)
    eq_str = eq_str.replace("y''", "Derivative(y(x), x, 2)").replace("y'", "Derivative(y(x), x)")
    return f"""
from sympy import *
x = symbols('x')
y = Function('y')
eq = sympify('{eq_str}'.replace('Derivative(y(x), x, 2)', "y''").replace('Derivative(y(x), x)', "y'"), locals={{
    'y': y, 'Derivative': Derivative, 'y(x)': y(x)
}})
try:
    ode = Eq(y(x).diff(x) - eq, 0) if not isinstance(eq, Eq) else eq
    sol = dsolve(ode, y(x))
except Exception:
    sol = dsolve(Eq(eq, 0), y(x))

_result = {{
    "answer": str(sol),
    "steps": [
        {{"step": f"ODE: {{ode}}",
          "explanation": "Identify the differential equation",
          "proof": "Definition of an ordinary differential equation"}},
        {{"step": f"General solution: {{sol}}",
          "explanation": "Solve using sympy dsolve (integrating factor or separation)",
          "proof": "Existence and Uniqueness Theorem for first-order linear ODEs"}}
    ],
    "proofs": [
        {{"theorem": "Existence and Uniqueness Theorem", "statement": "For y' + P(x)y = Q(x), a unique solution exists on intervals where P and Q are continuous", "applied_to": "first-order linear ODE"}},
        {{"theorem": "Integrating Factor Method", "statement": "Multiply by exp(integral(P dx)) to make the equation exact", "applied_to": "linear first-order ODE"}}
    ],
    "method": "differential_equation"
}}
"""


def _series_convergence_code(series_str):
    series_str = _normalize_expr(series_str)
    return f"""
from sympy import *
n = symbols('n')
expr = sympify('{series_str}'.replace('n', 'n'))

# Test ratio test
ratio = simplify(abs(expr.subs(n, n+1) / expr.subs(n, n)))
ratio_limit = limit(ratio, n, oo)

# Test root test
root = simplify(abs(expr) ** (S(1)/n))
root_limit = limit(root, n, oo)

converges = None
test_used = ""
if ratio_limit < 1:
    converges = True
    test_used = "Ratio Test"
elif ratio_limit > 1:
    converges = False
    test_used = "Ratio Test"
elif root_limit < 1:
    converges = True
    test_used = "Root Test"
elif root_limit > 1:
    converges = False
    test_used = "Root Test"
else:
    test_used = "Inconclusive (ratio and root tests give limit = 1)"

_result = {{
    "answer": f"Series {{'converges' if converges else 'diverges' if converges is False else 'inconclusive'}} ({{test_used}})",
    "steps": [
        {{"step": f"General term: a_n = {{expr}}",
          "explanation": "Identify the general term of the series",
          "proof": "Definition of an infinite series: sum of a_n from n=1 to infinity"}},
        {{"step": f"Ratio test: |a_(n+1)/a_n| -> {{ratio_limit}}",
          "explanation": f"Compute limit of the ratio of consecutive terms",
          "proof": "Ratio Test: if lim|a_(n+1)/a_n| < 1, series converges absolutely; if > 1, diverges"}},
        {{"step": f"Root test: |a_n|^(1/n) -> {{root_limit}}",
          "explanation": "Compute limit of the nth root of absolute value",
          "proof": "Root Test: if lim|a_n|^(1/n) < 1, converges; if > 1, diverges"}},
        {{"step": f"Conclusion: {{'Converges' if converges else 'Diverges' if converges is False else 'Inconclusive'}} by {{test_used}}",
          "explanation": "Apply the appropriate convergence test",
          "proof": test_used + " criterion"}}
    ],
    "proofs": [
        {{"theorem": "Ratio Test", "statement": "If lim|a_(n+1)/a_n| = L < 1, the series converges absolutely; if L > 1, it diverges", "applied_to": "ratio of consecutive terms"}},
        {{"theorem": "Root Test", "statement": "If lim|a_n|^(1/n) = L < 1, the series converges; if L > 1, it diverges", "applied_to": "nth root of terms"}},
        {{"theorem": "Divergence Test", "statement": "If lim a_n != 0, the series diverges", "applied_to": "necessary condition for convergence"}}
    ],
    "method": "series_convergence"
}}
"""


def _taylor_series_code(expr_str, var, point):
    expr_str = _normalize_expr(expr_str)
    return f"""
from sympy import *
{var} = symbols('{var}')
expr = sympify('{expr_str}')
point_val = sympify('{point}')
taylor = series(expr, {var}, point_val, n=5)
taylor_remove_o = taylor.removeO()

_result = {{
    "answer": str(taylor),
    "steps": [
        {{"step": f"f({var}) = {{expr}}",
          "explanation": f"Function to expand around {var} = {point}",
          "proof": "Taylor Series Theorem: f(x) = sum f^(n)(a)/n! * (x-a)^n"}},
        {{"step": f"Taylor series: {{taylor}}",
          "explanation": f"Expand around {var} = {point} to order 5",
          "proof": "Taylor Series Theorem: f(x) approx sum f^(n)(a)/n! * (x-a)^n"}}
    ],
    "proofs": [
        {{"theorem": "Taylor Series Theorem", "statement": "f(x) = sum from n=0 to inf of f^(n)(a)/n! * (x-a)^n", "applied_to": f"expansion around {var} = {point}"}},
        {{"theorem": "Convergence of Taylor Series", "statement": "Taylor series converges to f(x) within the radius of convergence", "applied_to": "series validity"}}
    ],
    "method": "taylor_series"
}}
"""


def _matrix_mult_code(m):
    groups = m.groups()
    if len(groups) == 4:
        r1 = [float(x.strip()) for x in groups[0].split(',')]
        r2 = [float(x.strip()) for x in groups[1].split(',')]
        r3 = [float(x.strip()) for x in groups[2].split(',')]
        r4 = [float(x.strip()) for x in groups[3].split(',')]
    else:
        return None
    return f"""
from sympy import *
A = Matrix([{r1}, {r2}])
B = Matrix([{r3}, {r4}])
C = A * B

_result = {{
    "answer": str(C),
    "steps": [
        {{"step": f"A = {{A}}, B = {{B}}",
          "explanation": "Define the two matrices",
          "proof": "Definition of matrix multiplication: (AB)_ij = sum_k A_ik * B_kj"}},
        {{"step": f"C = A * B = {{C}}",
          "explanation": "Multiply matrices: each entry C_ij is the dot product of row i of A with column j of B",
          "proof": "Matrix Multiplication: (AB)_ij = sum from k=1 to n of A_ik * B_kj"}}
    ],
    "proofs": [
        {{"theorem": "Matrix Multiplication", "statement": "(AB)_ij = sum_k A_ik * B_kj", "applied_to": "product of A and B"}},
        {{"theorem": "Dimension Compatibility", "statement": "A is m x n and B is n x p, then AB is m x p", "applied_to": "2x2 times 2x2 = 2x2"}}
    ],
    "method": "matrix_multiplication"
}}
"""


def _gradient_code(expr_str):
    expr_str = _normalize_expr(expr_str)
    return f"""
from sympy import *
x, y, z = symbols('x y z')
expr = sympify('{expr_str}')
vars_present = [v for v in [x, y, z] if v in expr.free_symbols]
grad = [diff(expr, v) for v in vars_present]

_result = {{
    "answer": str(grad),
    "steps": [
        {{"step": f"f = {{expr}}",
          "explanation": "Identify the scalar field",
          "proof": "Definition of gradient: grad f = (df/dx, df/dy, df/dz)"}},
        {{"step": f"gradient = {{grad}}",
          "explanation": "Compute partial derivatives with respect to each variable",
          "proof": "Gradient Theorem: grad f is the vector of partial derivatives"}}
    ],
    "proofs": [
        {{"theorem": "Gradient Definition", "statement": "grad f = (df/dx1, df/dx2, ..., df/dxn)", "applied_to": "scalar field f"}},
        {{"theorem": "Directional Derivative", "statement": "D_u f = grad f . u_hat", "applied_to": "gradient gives direction of steepest increase"}}
    ],
    "method": "gradient"
}}
"""


def _divergence_code(expr_str):
    return f"""
from sympy import *
x, y, z = symbols('x y z')
# Parse vector field components
parts = [p.strip() for p in '{expr_str}'.split(',')]
field = [sympify(_normalize_expr(p)) for p in parts]
vars_present = [x, y, z][:len(field)]
div = sum(diff(field[i], vars_present[i]) for i in range(len(field)))

_result = {{
    "answer": str(div),
    "steps": [
        {{"step": f"F = ({{', '.join(str(f) for f in field)}})",
          "explanation": "Identify the vector field",
          "proof": "Definition of divergence: div F = sum dF_i/dx_i"}},
        {{"step": f"div F = {{div}}",
          "explanation": "Sum the partial derivatives of each component with respect to its variable",
          "proof": "Divergence Theorem: integral of div F over volume = flux through boundary"}}
    ],
    "proofs": [
        {{"theorem": "Divergence Definition", "statement": "div F = dF1/dx + dF2/dy + dF3/dz", "applied_to": "vector field F"}},
        {{"theorem": "Divergence Theorem (Gauss)", "statement": "integral div F dV = integral F . n_hat dS", "applied_to": "relates flux to volume integral"}}
    ],
    "method": "divergence"
}}
"""


def _curl_code(expr_str):
    return f"""
from sympy import *
x, y, z = symbols('x y z')
parts = [p.strip() for p in '{expr_str}'.split(',')]
field = [sympify(_normalize_expr(p)) for p in parts]
if len(field) == 3:
    Fx, Fy, Fz = field
    curl_x = diff(Fz, y) - diff(Fy, z)
    curl_y = diff(Fx, z) - diff(Fz, x)
    curl_z = diff(Fy, x) - diff(Fx, y)
    curl_vec = [curl_x, curl_y, curl_z]
else:
    curl_vec = [diff(field[1], x) - diff(field[0], y)]

_result = {{
    "answer": str(curl_vec),
    "steps": [
        {{"step": f"F = ({{', '.join(str(f) for f in field)}})",
          "explanation": "Identify the vector field",
          "proof": "Definition of curl: curl F = nabla x F"}},
        {{"step": f"curl F = {{curl_vec}}",
          "explanation": "Compute the cross product of nabla with F",
          "proof": "Curl formula: curl F = (dFz/dy - dFy/dz, dFx/dz - dFz/dx, dFy/dx - dFx/dy)"}}
    ],
    "proofs": [
        {{"theorem": "Curl Definition", "statement": "curl F = nabla x F = (dFz/dy - dFy/dz, dFx/dz - dFz/dx, dFy/dx - dFx/dy)", "applied_to": "3D vector field"}},
        {{"theorem": "Stokes' Theorem", "statement": "integral curl F . dS = integral F . dr", "applied_to": "relates circulation to surface integral"}}
    ],
    "method": "curl"
}}
"""


def _partial_derivative_code(expr_str, var):
    expr_str = _normalize_expr(expr_str)
    return f"""
from sympy import *
x, y, z, t = symbols('x y z t')
expr = sympify('{expr_str}')
{var} = symbols('{var}')
pderiv = diff(expr, {var})

_result = {{
    "answer": str(pderiv),
    "steps": [
        {{"step": f"f = {{expr}}",
          "explanation": "Identify the function",
          "proof": "Definition of partial derivative: df/d{var} with other variables held constant"}},
        {{"step": f"partial f / partial {var} = {{pderiv}}",
          "explanation": f"Differentiate with respect to {var}, treating other variables as constants",
          "proof": "Partial Differentiation: d/d{var}(f) treating other variables as constants"}}
    ],
    "proofs": [
        {{"theorem": "Partial Derivative", "statement": f"partial f / partial {var} = lim(h->0) [f(...,{var}+h,...) - f(...,{var},...)] / h", "applied_to": "multivariable function"}},
        {{"theorem": "Clairaut's Theorem", "statement": "If mixed partials are continuous, then d2f/dxdy = d2f/dydx", "applied_to": "order of differentiation doesn't matter"}}
    ],
    "method": "partial_derivative"
}}
"""


def _permutation_code(n, k):
    return f"""
from sympy import *
n_val, k_val = {n}, {k}
result = factorial(n_val) // factorial(n_val - k_val)

_result = {{
    "answer": str(result),
    "steps": [
        {{"step": f"P({{n_val}}, {{k_val}}) = {{n_val}}! / ({{n_val}} - {{k_val}})! = {{result}}",
          "explanation": "Number of ordered arrangements of k items from n",
          "proof": "Multiplication Principle: n choices for first, n-1 for second, ..., n-k+1 for kth"}}
    ],
    "proofs": [
        {{"theorem": "Multiplication Principle", "statement": "If task A has m ways and task B has n ways, combined task has m*n ways", "applied_to": "counting ordered arrangements"}},
        {{"theorem": "Permutation Formula", "statement": "P(n,k) = n!/(n-k)!", "applied_to": f"P({n},{k})"}}
    ],
    "method": "permutation"
}}
"""


def _combination_code(n, k):
    return f"""
from sympy import *
n_val, k_val = {n}, {k}
result = binomial(n_val, k_val)

_result = {{
    "answer": str(result),
    "steps": [
        {{"step": f"C({{n_val}}, {{k_val}}) = {{n_val}}! / ({{k_val}}! * ({{n_val}} - {{k_val}})!) = {{result}}",
          "explanation": "Number of unordered selections of k items from n",
          "proof": "Combinations are permutations divided by k! (order doesn't matter)"}}
    ],
    "proofs": [
        {{"theorem": "Combination Formula", "statement": "C(n,k) = n!/(k!(n-k)!) = binomial(n,k)", "applied_to": f"C({n},{k})"}},
        {{"theorem": "Pascal's Identity", "statement": "C(n,k) = C(n-1,k-1) + C(n-1,k)", "applied_to": "recursive relationship"}}
    ],
    "method": "combination"
}}
"""


def _factorial_code(n):
    return f"""
from sympy import *
n_val = {n}
result = factorial(n_val)

_result = {{
    "answer": str(result),
    "steps": [
        {{"step": f"{{n_val}}! = {{result}}",
          "explanation": f"Product of all integers from 1 to {{n_val}}",
          "proof": "Definition of factorial: n! = n * (n-1) * ... * 1, with 0! = 1"}}
    ],
    "proofs": [
        {{"theorem": "Factorial Definition", "statement": "n! = n * (n-1) * ... * 2 * 1, with 0! = 1 by convention", "applied_to": f"{n}!"}},
        {{"theorem": "Recursive Property", "statement": "n! = n * (n-1)!", "applied_to": "recursive computation"}}
    ],
    "method": "factorial"
}}
"""


def _parse_numbers(s):
    s = s.strip().replace('[', '').replace(']', '')
    return [float(x.strip()) for x in s.split(',')]


def _mean_code(data_str):
    return f"""
from sympy import *
import numpy as np
data = np.array({_parse_numbers(data_str)})
mean_val = float(np.mean(data))

_result = {{
    "answer": str(mean_val),
    "steps": [
        {{"step": f"Data: {{data.tolist()}}",
          "explanation": "Given dataset",
          "proof": "Definition of arithmetic mean: sum of values / count"}},
        {{"step": f"Mean = sum(data) / n = {{mean_val}}",
          "explanation": "Sum all values and divide by the number of values",
          "proof": "Arithmetic Mean: x_bar = (1/n) * sum(x_i)"}}
    ],
    "proofs": [
        {{"theorem": "Arithmetic Mean", "statement": "x_bar = (1/n) * sum from i=1 to n of x_i", "applied_to": "dataset"}},
        {{"theorem": "Linearity of Expectation", "statement": "E[aX + bY] = aE[X] + bE[Y]", "applied_to": "mean as expected value"}}
    ],
    "method": "mean"
}}
"""


def _median_code(data_str):
    return f"""
from sympy import *
import numpy as np
data = np.array({_parse_numbers(data_str)})
median_val = float(np.median(data))
n = len(data)

_result = {{
    "answer": str(median_val),
    "steps": [
        {{"step": f"Data (sorted): {{sorted(data.tolist())}}",
          "explanation": "Sort the data",
          "proof": "Definition of median: middle value of sorted data"}},
        {{"step": f"Median = {{median_val}}",
          "explanation": f"{{'Middle value (n is odd)' if n % 2 else 'Average of two middle values (n is even)'}}",
          "proof": "Median: for odd n, the (n+1)/2 th value; for even n, average of n/2 and n/2+1 th values"}}
    ],
    "proofs": [
        {{"theorem": "Median Definition", "statement": "Middle value of sorted data; for even n, average of two middle values", "applied_to": "dataset"}}
    ],
    "method": "median"
}}
"""


def _stddev_code(data_str):
    return f"""
from sympy import *
import numpy as np
data = np.array({_parse_numbers(data_str)})
mean_val = float(np.mean(data))
variance_val = float(np.var(data, ddof=1))
std_val = float(np.std(data, ddof=1))

_result = {{
    "answer": str(std_val),
    "steps": [
        {{"step": f"Data: {{data.tolist()}}",
          "explanation": "Given dataset",
          "proof": "Definition of variance: average of squared deviations from mean"}},
        {{"step": f"Mean = {{mean_val}}",
          "explanation": "Compute the sample mean",
          "proof": "Arithmetic Mean: x_bar = (1/n) * sum(x_i)"}},
        {{"step": f"Variance = (1/(n-1)) * sum((x_i - x_bar)^2) = {{variance_val}}",
          "explanation": "Compute sample variance with Bessel's correction (n-1)",
          "proof": "Sample Variance: s^2 = (1/(n-1)) * sum from i=1 to n of (x_i - x_bar)^2"}},
        {{"step": f"Standard deviation = sqrt(variance) = {{std_val}}",
          "explanation": "Take the square root of variance",
          "proof": "Standard Deviation: s = sqrt(s^2) = sqrt(variance)"}}
    ],
    "proofs": [
        {{"theorem": "Sample Variance", "statement": "s^2 = (1/(n-1)) * sum (x_i - x_bar)^2", "applied_to": "dataset"}},
        {{"theorem": "Bessel's Correction", "statement": "Using n-1 instead of n gives unbiased estimator of population variance", "applied_to": "degrees of freedom"}},
        {{"theorem": "Chebyshev's Inequality", "statement": "At least 1 - 1/k^2 of data lies within k standard deviations of mean", "applied_to": "standard deviation interpretation"}}
    ],
    "method": "standard_deviation"
}}
"""


def _normal_dist_code(mean, std, x_val):
    return f"""
from sympy import *
import numpy as np
mu, sigma, x = {mean}, {std}, {x_val}
# Z-score
z = (x - mu) / sigma
# CDF using erf
from math import erf as _erf
cdf_val = 0.5 * (1 + _erf(z / np.sqrt(2)))
pdf_val = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * z**2)

_result = {{
    "answer": f"P(X <= {{x}}) = {{cdf_val:.6f}}, PDF = {{pdf_val:.6f}}, Z-score = {{z:.4f}}",
    "steps": [
        {{"step": f"X ~ N(mu={{mu}}, sigma={{sigma}}), find P(X <= {{x}})",
          "explanation": "Given normal distribution parameters",
          "proof": "Normal Distribution: f(x) = (1/(sigma*sqrt(2pi))) * exp(-(x-mu)^2/(2sigma^2))"}},
        {{"step": f"Z-score = (x - mu) / sigma = ({{x}} - {{mu}}) / {{sigma}} = {{z:.4f}}",
          "explanation": "Standardize to Z ~ N(0,1)",
          "proof": "Standardization: Z = (X - mu) / sigma transforms N(mu,sigma) to N(0,1)"}},
        {{"step": f"P(X <= {{x}}) = Phi({{z:.4f}}) = {{cdf_val:.6f}}",
          "explanation": "Use the standard normal CDF (error function)",
          "proof": "CDF: P(X <= x) = integral from -inf to x of f(t) dt = Phi(z)"}}
    ],
    "proofs": [
        {{"theorem": "Normal Distribution PDF", "statement": "f(x) = (1/(sigma*sqrt(2pi))) * exp(-(x-mu)^2/(2*sigma^2))", "applied_to": "probability density"}},
        {{"theorem": "Standardization", "statement": "Z = (X - mu) / sigma transforms any normal to standard normal", "applied_to": "Z-score computation"}},
        {{"theorem": "68-95-99.7 Rule", "statement": "68% within 1 sigma, 95% within 2 sigma, 99.7% within 3 sigma of mean", "applied_to": "normal distribution properties"}}
    ],
    "method": "normal_distribution"
}}
"""