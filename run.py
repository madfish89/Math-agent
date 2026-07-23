#!/usr/bin/env python3
"""Quick start: solve a math problem with the agent."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import MathAgent

agent = MathAgent()

# Demo problem
problem = "Find all eigenvalues and eigenvectors of the matrix [[3,1],[1,3]]"
print("DEMO: " + problem)
print()

result = agent.solve(problem, graph=False, multi_agent=False)
print("\n\nDone. Run with --interactive for chat mode.")