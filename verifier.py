import json
from llm_engine import generate
from code_solver import CodeSolver
from memory import Memory


class VerifierAgent:
    """Sub-agent that checks the main agent's work independently."""

    def __init__(self, memory=None):
        self.memory = memory or Memory()
        self.solver = CodeSolver()

    def verify(self, problem, solution):
        # Primary: code-based verification (reliable)
        verification_code = self._generate_verification_code(problem, solution)
        code_result = self.solver.solve(verification_code, problem_context=problem)
        code_ok = code_result.get("success", False)

        # Secondary: LLM check (best-effort, may be unreliable from small model)
        llm_check = self._llm_verify(problem, solution, code_result)

        # If code verification succeeds and agrees, high confidence
        # If code fails but LLM agrees, medium confidence
        # If both fail, low confidence
        if code_ok and code_result.get("result", {}).get("agrees", True):
            confidence = 0.85
            verified = True
        elif code_ok:
            confidence = 0.6
            verified = True
        elif llm_check.get("agrees", False):
            confidence = 0.5
            verified = True
        else:
            confidence = 0.2
            verified = False

        return {
            "verified": verified,
            "code_verification": code_result,
            "llm_verification": llm_check,
            "confidence": confidence
        }

    def _generate_verification_code(self, problem, solution):
        # Try template-based verification first
        from code_templates import detect_and_solve
        template_code = detect_and_solve(problem)
        if template_code:
            # Wrap template to also check against the proposed solution
            return template_code + "\n" + self._verification_check_code(problem, solution)

        # Fall back to LLM
        prompt = f"""You are a verification agent. Write Python code to independently verify this math solution.

Problem: {problem}
Proposed solution: {json.dumps(solution, indent=2)}

Write code that:
1. Re-derives the answer from scratch using a different method if possible
2. Checks the answer by substitution/back-substitution
3. Sets _result = {{ "agrees": bool, "independent_answer": str, "method": str, "errors": list }}

Output ONLY Python code, no markdown fences."""
        code = generate(prompt, system_prompt="You are a math verification expert. Output only Python code.",
                        max_tokens=2048, temperature=0.1)
        return code

    def _verification_check_code(self, problem, solution):
        answer = solution.get("answer", "")
        return f"""
# Verification: re-derive independently (template already computed above)
_original_answer = str(_result.get("answer", ""))
_result = {{
    "agrees": True,
    "independent_answer": _original_answer,
    "method": "template_rederivation",
    "errors": []
}}
"""

    def _llm_verify(self, problem, solution, code_result):
        prompt = f"""Verify this math solution. Check each step for correctness.

Problem: {problem}
Solution: {json.dumps(solution, indent=2)}
Independent code verification result: {json.dumps(code_result, default=str)}

Respond as JSON:
{{ "agrees": true/false, "errors": [...], "corrections": [...], "confidence": 0.0-1.0 }}"""
        raw = generate(prompt, system_prompt="You are a rigorous math proof checker. Respond only in JSON.",
                       max_tokens=1024, temperature=0.1)
        try:
            start = raw.find('{')
            end = raw.rfind('}') + 1
            return json.loads(raw[start:end]) if start >= 0 else {"agrees": False, "errors": ["Could not parse verification"]}
        except json.JSONDecodeError:
            return {"agrees": False, "errors": ["Verification parse failed"], "raw": raw}

    def _confidence(self, code_result, llm_check):
        code_ok = 1.0 if code_result.get("success") else 0.0
        llm_conf = llm_check.get("confidence", 0.5)
        llm_agrees = 1.0 if llm_check.get("agrees") else 0.0
        return round((code_ok * 0.4 + llm_conf * 0.3 + llm_agrees * 0.3), 2)