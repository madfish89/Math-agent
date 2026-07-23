import subprocess
import sys
import os
import tempfile
import json
import traceback
from io import StringIO

from sympy import latex, srepr
import sympy as sp


class CodeSolver:
    def __init__(self):
        self.timeout = 30

    def solve(self, code, problem_context=""):
        if not code.strip():
            return {"success": False, "error": "Empty code"}
        if not self._is_safe(code):
            return {"success": False, "error": "Code blocked by safety filter"}
        return self._run(code, problem_context)

    def _is_safe(self, code):
        banned = ["import os", "import subprocess", "import shutil", "import socket",
                  "os.system", "__import__", "eval(", "exec(", "open(", "os.remove",
                  "os.rmdir", "shutil.rmtree", "os.environ"]
        for b in banned:
            if b in code:
                return False
        return True

    def _run(self, code, problem_context=""):
        wrapper = (
            "import sys, json, math, cmath\n"
            "from sympy import *\n"
            "from sympy.geometry import *\n"
            "import numpy as np\n"
            "import scipy\n"
            "import scipy.integrate\n"
            "import scipy.optimize\n"
            "import scipy.linalg\n"
            "from matplotlib import pyplot as plt\n"
            "import matplotlib\n"
            "matplotlib.use('Agg')\n"
            "try:\n"
            "    from graphing import make_graph\n"
            "except Exception:\n"
            "    make_graph = None\n"
            f"__PROBLEM_CTX__ = {json.dumps(problem_context)}\n"
        )
        full = wrapper + "\n" + code + "\n\n"
        full += "try:\n"
        full += "    print('__RESULT_JSON__:' + json.dumps(_result, default=str))\n"
        full += "except NameError:\n"
        full += "    print('__RESULT_JSON__:{\"error\": \"no _result variable set\"}')\n"
        fd, tmp = tempfile.mkstemp(suffix=".py", dir="/tmp")
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(full)
            proc = subprocess.run(
                [sys.executable, tmp], capture_output=True, text=True, timeout=self.timeout,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
            stdout = proc.stdout
            stderr = proc.stderr
            result = None
            for line in stdout.splitlines():
                if line.startswith("__RESULT_JSON__:"):
                    result = json.loads(line[len("__RESULT_JSON__:"):])
                    break
            return {
                "success": proc.returncode == 0 and result is not None,
                "result": result,
                "stdout": stdout,
                "stderr": stderr,
                "returncode": proc.returncode
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Code timed out"}
        finally:
            os.unlink(tmp)

    @staticmethod
    def generate_code_prompt(problem):
        return f"""Solve this math problem by writing Python code. Use sympy for symbolic math.
Assign the final answer to a variable called `_result` (a dict with keys: "answer", "steps", "proofs").
Each step in "steps" must have: "step", "explanation", "proof" (justification).
Each proof in "proofs" should cite the theorem or rule used.

Problem: {problem}

Write only Python code, no markdown fences. Use sympy. Set _result = {{...}}.
For the answer, prefer exact symbolic forms (e.g. Rational(1,3) not 0.333).
"""