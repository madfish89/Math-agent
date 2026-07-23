import json
import os
import sys
import time

from config import GRAPH_DIR
from llm_engine import generate, generate_code, generate_stream
from code_solver import CodeSolver
from code_templates import detect_and_solve
from memory import Memory
from verifier import VerifierAgent
from multi_agent import MultiAgentSystem


class MathAgent:
    def __init__(self, use_multi_agent=False, n_agents=3):
        self.memory = Memory()
        self.solver = CodeSolver()
        self.verifier = VerifierAgent(self.memory)
        self.multi_agent = MultiAgentSystem(self.memory, max_agents=n_agents)
        self.use_multi_agent = use_multi_agent
        self.history = []
        self._stream = False

    def solve(self, problem, graph=False, multi_agent=None):
        if multi_agent is None:
            multi_agent = self.use_multi_agent

        print(f"\n{'='*60}")
        print(f"PROBLEM: {problem}")
        print(f"{'='*60}")

        # Check memory for similar problems (semantic + keyword)
        similar_problems = self.memory.search_similar_problems(problem, limit=3)
        if similar_problems:
            print(f"\n📚 Found {len(similar_problems)} similar problems in memory:")
            for sp in similar_problems:
                print(f"  [sim={sp['similarity']}] {sp['problem'][:80]}")
                if sp['verified']:
                    print(f"    → {sp['solution'][:100]}")

        similar = self.memory.search_facts(problem)
        if similar:
            print(f"\n Found {len(similar)} related facts in memory:")
            for s in similar[:3]:
                print(f"  - {s['key']}: {s['value'][:100]}")

        # Step 1: Try direct code template first, then LLM
        print("\n Generating solution code...")
        code = detect_and_solve(problem)
        if code:
            print("  (using direct code template)")
        else:
            print("  (using LLM code generation)")
            code = generate_code(problem)

        print(f"\n--- Generated Code ---\n{code}\n--- End Code ---")

        # Step 2: Execute code
        print("\n  Executing code...")
        result = self.solver.solve(code, problem_context=problem)

        if not result.get("success"):
            print(f"\n Code execution failed: {result.get('error', result.get('stderr', 'unknown'))}")
            print("\n Retrying with LLM reasoning...")
            return self._llm_fallback(problem, code, result)

        solution = result.get("result", {})
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")

        if stdout.strip():
            print(f"\n stdout:\n{stdout[:500]}")
        if stderr.strip():
            print(f"\n stderr:\n{stderr[:300]}")

        # Step 3: Verify
        print("\nVerifying solution...")
        verification = self.verifier.verify(problem, solution)
        confidence = verification.get("confidence", 0)
        print(f"   Confidence: {confidence}")
        print(f"   Code verified: {verification['code_verification'].get('success', False)}")
        print(f"   LLM verified: {verification['llm_verification'].get('agrees', False)}")

        # Step 4: Multi-agent discussion if requested
        agent_discussion = None
        if multi_agent:
            print("\n Starting multi-agent discussion...")
            agent_discussion = self.multi_agent.discuss(problem)
            print(f"   {len(agent_discussion.get('agent_responses', {}))} agents responded")
            print(f"   Verification confidence: {agent_discussion.get('verification', {}).get('confidence', 'N/A')}")

        # Step 5: Format output
        print("\n" + "="*60)
        print("SOLUTION")
        print("="*60)

        self._print_solution(problem, solution, verification, agent_discussion)

        # Step 6: Graph
        graph_path = None
        if graph and solution.get("graph"):
            graph_path = self._handle_graph(solution["graph"], problem)
            if graph_path:
                print(f"\n Graph saved: {graph_path}")

        # Step 7: Look up external sources for theorems used
        proofs = solution.get("proofs", [])
        if proofs:
            print("\n🔗 Looking up external sources for theorems...")
            try:
                from web_search import search_math_concept, format_sources
                first_theorem = proofs[0].get("theorem", "")
                if first_theorem:
                    sources = search_math_concept(first_theorem)
                    formatted = format_sources(sources)
                    if formatted.strip():
                        print(formatted)
            except Exception as e:
                print(f"  (web search unavailable: {e})")

        # Step 8: Store in memory
        self.memory.store_problem(
            problem=problem,
            solution=json.dumps(solution, default=str),
            proof=json.dumps(solution.get("proofs", []), default=str),
            verified=verification.get("verified", False),
            method=solution.get("method", "code_execution")
        )
        self.memory.remember("user", problem)
        self.memory.remember("assistant", json.dumps(solution, default=str))

        return {
            "problem": problem,
            "solution": solution,
            "verification": verification,
            "multi_agent": agent_discussion,
            "graph": graph_path,
            "code": code,
            "stdout": stdout,
            "stderr": stderr,
        }

    def _print_solution(self, problem, solution, verification, discussion=None):
        answer = solution.get("answer", "No answer found")
        print(f"\nAnswer: {answer}\n")

        steps = solution.get("steps", [])
        if steps:
            print("Step-by-step solution:")
            print("-" * 40)
            for i, step in enumerate(steps, 1):
                print(f"\nStep {i}: {step.get('step', '')}")
                print(f"  Explanation: {step.get('explanation', '')}")
                proof = step.get('proof', '')
                if proof:
                    print(f"  Proof: {proof}")

        proofs = solution.get("proofs", [])
        if proofs:
            print("\n" + "-" * 40)
            print("Proofs used:")
            print("-" * 40)
            for p in proofs:
                theorem = p.get("theorem", "")
                statement = p.get("statement", "")
                applied = p.get("applied_to", p.get("applied_to", ""))
                print(f"\n📐 {theorem}")
                print(f"   {statement}")
                if applied:
                    print(f"   Applied to: {applied}")

        print("\n" + "-" * 40)
        print(f"Verification confidence: {verification.get('confidence', 0)}")
        v = verification.get("llm_verification", {})
        if v.get("errors"):
            print(f"Errors found: {v['errors']}")
        if v.get("corrections"):
            print(f"Corrections: {v['corrections']}")

        if discussion:
            print("\n" + "-" * 40)
            print("Multi-agent discussion:")
            for agent, resp in discussion.get("agent_responses", {}).items():
                ans = resp.get("answer", resp.get("raw", ""))[:200] if isinstance(resp, dict) else str(resp)[:200]
                print(f"  {agent}: {ans}")

    def _handle_graph(self, graph_spec, problem):
        try:
            from graphing import make_graph
            if isinstance(graph_spec, str):
                return make_graph(graph_spec, title=problem[:50])
            elif isinstance(graph_spec, dict):
                return make_graph(**graph_spec, title=problem[:50])
        except Exception as e:
            print(f"Graph error: {e}")
            return None

    def _llm_fallback(self, problem, code, result):
        print("\n⚠️  Code execution failed. Using LLM reasoning fallback...")
        prompt = f"""The Python code failed to execute. Solve this problem using pure reasoning.

Problem: {problem}

Failed code:
{code}

Error: {result.get('error', result.get('stderr', 'unknown'))}

Provide a detailed solution with:
- Step-by-step explanation
- Justification for each step (proofs/theorems)
- Final answer

Format as JSON:
{{ "answer": str, "steps": [{{ "step": str, "explanation": str, "proof": str }}], "proofs": [{{ "theorem": str, "statement": str }}], "method": "llm_reasoning" }}"""

        if self._stream:
            import sys
            print("\n--- Streaming LLM response ---")
            def _print_token(tok):
                print(tok, end="", flush=True)
            raw = generate_stream(prompt, system_prompt="You are a math expert. Provide rigorous solutions with proofs.",
                           max_tokens=2048, temperature=0.3, on_token=_print_token)
            print("\n--- End stream ---")
        else:
            raw = generate(prompt, system_prompt="You are a math expert. Provide rigorous solutions with proofs.",
                           max_tokens=2048, temperature=0.3)
        try:
            start = raw.find('{')
            end = raw.rfind('}') + 1
            solution = json.loads(raw[start:end]) if start >= 0 else {"answer": raw, "method": "llm_reasoning"}
        except json.JSONDecodeError:
            solution = {"answer": raw, "method": "llm_reasoning"}

        verification = self.verifier.verify(problem, solution)
        self._print_solution(problem, solution, verification, None)

        self.memory.store_problem(
            problem=problem,
            solution=json.dumps(solution, default=str),
            proof=json.dumps(solution.get("proofs", []), default=str),
            verified=verification.get("verified", False),
            method="llm_reasoning"
        )
        self.memory.remember("user", problem)
        self.memory.remember("assistant", json.dumps(solution, default=str))

        return {
            "problem": problem,
            "solution": solution,
            "verification": verification,
            "multi_agent": None,
            "graph": None,
            "code": code,
            "fallback": True,
        }

    def chat(self, message):
        """Interactive chat mode - route requests."""
        msg_lower = message.lower().strip()

        if any(w in msg_lower for w in ["solve", "calculate", "find", "derive", "prove",
                                        "compute", "evaluate", "integrate", "differentiate",
                                        "solve", "what is", "determine"]):
            wants_graph = "graph" in msg_lower or "plot" in msg_lower
            wants_multi = "agents" in msg_lower or "discuss" in msg_lower
            return self.solve(message, graph=wants_graph, multi_agent=wants_multi)
        else:
            # General math conversation
            history = self.memory.recall(10)
            ctx = "\n".join(f"{m['role']}: {m['content'][:200]}" for m in history[-6:])
            prompt = f"""Previous context:
{ctx}

User: {message}

Respond as a math expert. Use proper notation. If the user is asking a math problem, suggest they prefix with 'solve'."""
            response = generate(prompt, system_prompt="You are a helpful math expert agent.",
                                max_tokens=1024, temperature=0.5)
            self.memory.remember("user", message)
            self.memory.remember("assistant", response)
            return response


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Math AI Agent")
    parser.add_argument("problem", nargs="?", help="Math problem to solve")
    parser.add_argument("--graph", action="store_true", help="Generate graph")
    parser.add_argument("--multi-agent", action="store_true", help="Use multi-agent discussion")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--memory", action="store_true", help="Show memory")
    parser.add_argument("--stream", action="store_true", help="Stream LLM output token-by-token")
    parser.add_argument("--model-size", choices=["1.5B", "7B"], default="1.5B", help="Model size to use")
    args = parser.parse_args()

    if args.model_size != "1.5B":
        os.environ["MATH_AGENT_MODEL"] = args.model_size
        # Re-import config with new env
        import importlib
        import config as _cfg
        importlib.reload(_cfg)
        from config import GGUF_PATH as _new_gguf
        # Need to also reload llm_engine
        import llm_engine as _le
        importlib.reload(_le)
        from llm_engine import generate as _gen, generate_code as _gcode, generate_stream as _gstream
        # Re-instantiate
        agent = MathAgent(use_multi_agent=args.multi_agent)
        agent._stream = args.stream
    else:
        agent = MathAgent(use_multi_agent=args.multi_agent)
        agent._stream = args.stream

    if args.memory:
        facts = agent.memory.search_facts("")
        problems = agent.memory.get_problems()
        print(f"Stored facts: {len(facts)}")
        for f in facts:
            print(f"  {f['key']}: {f['value'][:100]}")
        print(f"\nSolved problems: {len(problems)}")
        for p in problems:
            print(f"  [{'✓' if p['verified'] else '✗'}] {p['problem'][:80]}")
        return

    if args.interactive or not args.problem:
        print("Math AI Agent - Interactive Mode")
        print("Type 'quit' to exit, 'memory' to see stored knowledge")
        print("="*60)
        while True:
            try:
                user_input = input("\n🧮 > ")
                if user_input.strip().lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break
                if user_input.strip().lower() == "memory":
                    facts = agent.memory.search_facts("")
                    problems = agent.memory.get_problems()
                    print(f"\nFacts: {len(facts)}, Problems: {len(problems)}")
                    for p in problems:
                        print(f"  [{'✓' if p['verified'] else '✗'}] {p['problem'][:80]}")
                    continue
                result = agent.chat(user_input)
                if isinstance(result, dict):
                    pass  # solve already printed
                else:
                    print(result)
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break
    else:
        agent.solve(args.problem, graph=args.graph, multi_agent=args.multi_agent)


if __name__ == "__main__":
    main()