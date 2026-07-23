import json
import skills

MODEL = "gpt-4o-mini"

def handle(problem, llm_response=None):
    """
    Always uses the LLM for the answer and steps.
    If the problem is a graph request, Python generates the graph data.
    Returns JSON string.
    """
    # Parse LLM response
    if llm_response:
        try:
            sol = json.loads(llm_response)
        except:
            sol = {"answer": llm_response, "steps": [], "confidence": 0.5}
    else:
        sol = {"answer": "", "steps": [], "confidence": 0.5}

    # Check if Python skills can generate a graph for this problem
    py_result = skills.solve(problem)
    if py_result and "graph_data" in py_result:
        sol["graph_data"] = py_result["graph_data"]
        # Use the Python answer as a fallback if LLM gave nothing
        if not sol.get("answer"):
            sol["answer"] = py_result.get("answer", "")
        if not sol.get("steps"):
            sol["steps"] = py_result.get("steps", [])
        if not sol.get("confidence"):
            sol["confidence"] = py_result.get("confidence", 0.9)

    return json.dumps(sol)

def llm_prompt(problem):
    """Build the prompt for GPT-4o-mini."""
    return [
        {"role": "system", "content": "You are a math expert. Solve with step-by-step reasoning and proofs for each step. Respond ONLY in JSON: {\"answer\": str, \"steps\": [{\"step\": str, \"explanation\": str, \"proof\": str}], \"confidence\": float}"},
        {"role": "user", "content": problem},
    ]