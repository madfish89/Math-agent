import json
from llm_engine import generate
from memory import Memory
from verifier import VerifierAgent


class MultiAgentSystem:
    """Spawn sub-agents for discussion and cross-verification."""

    def __init__(self, memory=None, max_agents=3):
        self.memory = memory or Memory()
        self.max_agents = max_agents
        self.verifier = VerifierAgent(self.memory)

    def discuss(self, problem, context=""):
        agents = [f"Agent_{i+1}" for i in range(self.max_agents)]
        responses = {}

        for agent in agents:
            persona = self._get_persona(agent)
            prompt = f"""You are {agent}, a math expert with expertise in {persona}.
A problem has been posed. Provide your solution approach and answer.

Problem: {problem}
Context from other agents: {context if context else "None yet"}

Give your solution as JSON:
{{ "approach": str, "answer": str, "confidence": float, "concerns": list, "steps": list }}

Respond ONLY with JSON."""
            raw = generate(prompt, system_prompt=f"You are {agent}, expert in {persona}.",
                           max_tokens=1024, temperature=0.5 + 0.1 * agents.index(agent))
            try:
                start = raw.find('{')
                end = raw.rfind('}') + 1
                responses[agent] = json.loads(raw[start:end]) if start >= 0 else {"raw": raw}
            except json.JSONDecodeError:
                responses[agent] = {"raw": raw, "parse_error": True}

        # Synthesize
        synthesis = self._synthesize(problem, responses)

        # Verify
        verification = self.verifier.verify(problem, synthesis)

        return {
            "agent_responses": responses,
            "synthesis": synthesis,
            "verification": verification
        }

    def _get_persona(self, agent_name):
        personas = {
            "Agent_1": "algebra, calculus, and symbolic computation",
            "Agent_2": "number theory, combinatorics, and proof writing",
            "Agent_3": "geometry, topology, and numerical methods",
        }
        return personas.get(agent_name, "general mathematics")

    def _synthesize(self, problem, responses):
        prompt = f"""Synthesize multiple expert solutions into one unified answer.

Problem: {problem}
Agent responses: {json.dumps(responses, indent=2, default=str)}

Create a unified solution as JSON:
{{ "answer": str, "method": str, "steps": [{{ "step": str, "explanation": str, "proof": str }}], "proofs": [{{ "theorem": str, "statement": str }}], "consensus": bool, "graph": str or null }}

Respond ONLY with JSON."""
        raw = generate(prompt, system_prompt="You are a synthesis agent. Combine expert opinions into one rigorous solution.",
                       max_tokens=2048, temperature=0.3)
        try:
            start = raw.find('{')
            end = raw.rfind('}') + 1
            return json.loads(raw[start:end]) if start >= 0 else {"raw": raw}
        except json.JSONDecodeError:
            return {"raw": raw, "parse_error": True}

    def cross_check(self, problem, solution_a, solution_b):
        """Have two agents compare solutions."""
        prompt = f"""Two solutions were proposed for the same problem. Compare them.

Problem: {problem}
Solution A: {json.dumps(solution_a, default=str)}
Solution B: {json.dumps(solution_b, default=str)}

Respond as JSON:
{{ "same_answer": bool, "correct": "A"|"B"|"both"|"neither", "differences": list, "explanation": str }}"""
        raw = generate(prompt, system_prompt="You are a cross-checking agent. Compare solutions rigorously.",
                       max_tokens=1024, temperature=0.1)
        try:
            start = raw.find('{')
            end = raw.rfind('}') + 1
            return json.loads(raw[start:end]) if start >= 0 else {"raw": raw}
        except json.JSONDecodeError:
            return {"raw": raw}