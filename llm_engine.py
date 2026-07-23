import os
import json
from config import GGUF_PATH, MODEL_DIR, MODEL_ID, GGUF_REPO, GGUF_FILE, MAX_TOKENS, TEMPERATURE, N_CTX

_LLM = None


def get_llm():
    global _LLM
    if _LLM is not None:
        return _LLM
    if os.path.exists(GGUF_PATH):
        from llama_cpp import Llama
        _LLM = Llama(
            model_path=GGUF_PATH,
            n_ctx=N_CTX,
            n_gpu_layers=-1,
            verbose=False,
        )
        return _LLM

    # Fall back to transformers (HF model)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=MODEL_DIR)

    # Check for trained LoRA adapter
    lora_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training", "math-lora")
    if os.path.exists(lora_path):
        from peft import PeftModel
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, cache_dir=MODEL_DIR,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(model, lora_path)
        print(f"Loaded LoRA adapter from {lora_path}")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, cache_dir=MODEL_DIR,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    _LLM = {"tokenizer": tok, "model": model}
    return _LLM


def generate(prompt, system_prompt=None, max_tokens=None, temperature=None, stream=False, on_token=None):
    llm = get_llm()
    mt = max_tokens or MAX_TOKENS
    temp = temperature or TEMPERATURE

    if stream and on_token:
        return _generate_stream(llm, prompt, system_prompt, mt, temp, on_token)

    if isinstance(llm, dict):
        tok = llm["tokenizer"]
        model = llm["model"]
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=mt, temperature=temp, do_sample=temp > 0)
        response = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response.strip()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    out = llm.create_chat_completion(messages, max_tokens=mt, temperature=temp)
    return out["choices"][0]["message"]["content"].strip()


def _generate_stream(llm, prompt, system_prompt, max_tokens, temperature, on_token):
    """Stream tokens one by one. on_token(token_str) is called for each token."""
    if isinstance(llm, dict):
        # transformers streaming via TextIteratorStreamer
        from transformers import TextIteratorStreamer
        import threading
        tok = llm["tokenizer"]
        model = llm["model"]
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = {**inputs, "max_new_tokens": max_tokens, "temperature": temperature,
                       "do_sample": temperature > 0, "streamer": streamer}
        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()
        full = ""
        for token_text in streamer:
            if token_text:
                full += token_text
                on_token(token_text)
        thread.join()
        return full.strip()

    # llama.cpp streaming
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    full = ""
    for chunk in llm.create_chat_completion(messages, max_tokens=max_tokens,
                                             temperature=temperature, stream=True):
        delta = chunk["choices"][0].get("delta", {})
        token_text = delta.get("content", "")
        if token_text:
            full += token_text
            on_token(token_text)
    return full.strip()


def generate_stream(prompt, system_prompt=None, max_tokens=None, temperature=None, on_token=None):
    """Convenience wrapper for streaming generation. Returns full text."""
    return generate(prompt, system_prompt=system_prompt, max_tokens=max_tokens,
                    temperature=temperature, stream=True, on_token=on_token)


def generate_code(problem):
    system = "You are a Python code generator. Output ONLY executable Python code. No markdown, no explanation, no comments. Start with 'from sympy import *' and end with '_result = {...}'."
    user = f"""Write Python code to solve this math problem. The code must:
1. Use sympy for symbolic computation
2. Set _result as a dict with keys: "answer", "steps", "proofs", "method"
3. "steps" = list of {{"step": str, "explanation": str, "proof": str}}
4. "proofs" = list of {{"theorem": str, "statement": str, "applied_to": str}}
5. Prefer exact symbolic answers
6. Do NOT use markdown fences or explanation text - ONLY Python code

Problem: {problem}

from sympy import *
"""
    raw = generate(user, system_prompt=system, max_tokens=2048, temperature=0.2)
    return _extract_code(raw)


def _extract_code(text):
    """Extract Python code from LLM output, handling markdown fences and explanations."""
    text = text.strip()

    # Try to find code between ```python and ```
    if "```python" in text:
        start = text.find("```python")
        end = text.find("```", start + 9)
        if end > start:
            return text[start + 9:end].strip()
    if "```" in text:
        parts = text.split("```")
        for i, part in enumerate(parts):
            if i % 2 == 1 and part.strip():
                # This is inside a code fence
                lines = part.strip().split("\n")
                # Skip language identifier line
                if lines[0].strip().lower() in ["python", "py", ""]:
                    lines = lines[1:]
                return "\n".join(lines).strip()

    # Try to find the first Python statement
    lines = text.split("\n")
    code_lines = []
    started = False
    for line in lines:
        stripped = line.strip()
        if not started:
            if stripped.startswith(("from ", "import ", "x ", "y ", "z ", "lambda",
                                     "A ", "M ", "n ", "f ", "g ", "expr", "result", "_result",
                                     "def ", "class ", "#")):
                started = True
                code_lines.append(line)
        else:
            code_lines.append(line)

    if code_lines:
        return "\n".join(code_lines).strip()

    return text