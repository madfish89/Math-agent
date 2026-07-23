import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model", "Qwen2.5-Math-1.5B-Instruct")
GGUF_PATH = os.path.join(BASE_DIR, "model", "qwen2.5-math-1.5b-instruct-q5_k_m.gguf")
GRAPH_DIR = os.path.join(BASE_DIR, "graphs")
MEMORY_DB = os.path.join(BASE_DIR, "memory", "agent_memory.db")
TRAINING_DIR = os.path.join(BASE_DIR, "training")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Model selection: override via env MATH_AGENT_MODEL=1.5B or 7B
MODEL_SIZE = os.environ.get("MATH_AGENT_MODEL", "1.5B")

MODEL_CONFIGS = {
    "1.5B": {
        "model_id": "Qwen/Qwen2.5-Math-1.5B-Instruct",
        "gguf_repo": "bartowski/Qwen2.5-Math-1.5B-Instruct-GGUF",
        "gguf_file": "Qwen2.5-Math-1.5B-Instruct-Q5_K_M.gguf",
        "gguf_path": os.path.join(BASE_DIR, "model", "qwen2.5-math-1.5b-instruct-q5_k_m.gguf"),
        "n_ctx": 4096,
    },
    "7B": {
        "model_id": "Qwen/Qwen2.5-Math-7B-Instruct",
        "gguf_repo": "bartowski/Qwen2.5-Math-7B-Instruct-GGUF",
        "gguf_file": "Qwen2.5-Math-7B-Instruct-Q5_K_M.gguf",
        "gguf_path": os.path.join(BASE_DIR, "model", "qwen2.5-math-7b-instruct-q5_k_m.gguf"),
        "n_ctx": 4096,
    },
}

_active = MODEL_CONFIGS.get(MODEL_SIZE, MODEL_CONFIGS["1.5B"])

MODEL_ID = _active["model_id"]
GGUF_REPO = _active["gguf_repo"]
GGUF_FILE = _active["gguf_file"]
GGUF_PATH = _active["gguf_path"]
N_CTX = _active["n_ctx"]

MAX_TOKENS = 2048
TEMPERATURE = 0.7

os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(MEMORY_DIR := os.path.dirname(MEMORY_DB), exist_ok=True)
os.makedirs(TRAINING_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)