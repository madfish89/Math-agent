import os
import sys
import json
import threading
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

from config import GRAPH_DIR, MODEL_CONFIGS, MODEL_SIZE
from memory import Memory
from code_templates import detect_and_solve
from code_solver import CodeSolver
from verifier import VerifierAgent

app = Flask(__name__, static_folder='web')
CORS(app)

memory = Memory()
solver = CodeSolver()
verifier = VerifierAgent(memory)

# Track active model
current_model = os.environ.get("MATH_AGENT_MODEL", "1.5B")

# Thread-safe job storage
_jobs = {}
_jobs_lock = threading.Lock()


@app.route('/')
def index():
    return send_from_directory('web', 'index.html')


@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('web', path)


@app.route('/api/models')
def get_models():
    models = []
    for size, cfg in MODEL_CONFIGS.items():
        gguf_exists = os.path.exists(cfg['gguf_path'])
        models.append({
            'id': size,
            'name': f"Qwen2.5-Math-{size}-Instruct",
            'available': gguf_exists,
            'gguf_path': cfg['gguf_path'],
            'model_id': cfg['model_id'],
        })
    return jsonify({'models': models, 'current': current_model})


@app.route('/api/modes')
def get_modes():
    return jsonify({
        'modes': [
            {'id': 'solve', 'name': 'Solve', 'desc': 'Generate code + verify'},
            {'id': 'multi-agent', 'name': 'Multi-Agent', 'desc': '3 agents discuss & cross-check'},
            {'id': 'stream', 'name': 'Streaming', 'desc': 'Token-by-token streaming output'},
        ]
    })


@app.route('/api/graph/<filename>')
def get_graph(filename):
    return send_file(os.path.join(GRAPH_DIR, filename), mimetype='image/png')


@app.route('/api/memory')
def get_memory():
    problems = memory.get_problems(20)
    facts = memory.search_facts("")
    return jsonify({'problems': problems, 'facts': facts, 'count': len(problems)})


@app.route('/api/solve', methods=['POST'])
def solve():
    data = request.json
    problem = data.get('problem', '').strip()
    use_multi_agent = data.get('multi_agent', False)
    use_stream = data.get('stream', False)
    model_size = data.get('model', '1.5B')

    if not problem:
        return jsonify({'error': 'No problem provided'}), 400

    job_id = str(uuid.uuid4())[:8]
    with _jobs_lock:
        _jobs[job_id] = {'status': 'running', 'steps': [], 'result': None}

    def run_job():
        try:
            result = _do_solve(problem, use_multi_agent, use_stream, model_size, job_id)
            with _jobs_lock:
                _jobs[job_id]['status'] = 'done'
                _jobs[job_id]['result'] = result
        except Exception as e:
            with _jobs_lock:
                _jobs[job_id]['status'] = 'error'
                _jobs[job_id]['error'] = str(e)

    t = threading.Thread(target=run_job, daemon=True)
    t.start()

    return jsonify({'job_id': job_id})


@app.route('/api/job/<job_id>')
def get_job(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        return jsonify(job)


@app.route('/api/job/<job_id>/stream')
def stream_job(job_id):
    from flask import Response
    def generate():
        while True:
            with _jobs_lock:
                job = _jobs.get(job_id)
                if not job:
                    yield f"data: {json.dumps({'type': 'error', 'msg': 'not found'})}\n\n"
                    return
                steps_sent = len(job.get('steps_sent', []))
                steps = job.get('steps', [])
                for i in range(steps_sent, len(steps)):
                    yield f"data: {json.dumps(steps[i])}\n\n"
                    job.setdefault('steps_sent', []).append(i)
                if job['status'] in ('done', 'error'):
                    if job['status'] == 'done':
                        yield f"data: {json.dumps({'type': 'done', 'result': job.get('result')})}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'error', 'msg': job.get('error', 'unknown')})}\n\n"
                    return
            time.sleep(0.3)
    return Response(generate(), mimetype='text/event-stream')


def _add_step(job_id, step_type, content, extra=None):
    step = {'type': step_type, 'content': content, 'ts': time.time()}
    if extra:
        step.update(extra)
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]['steps'].append(step)


def _do_solve(problem, use_multi_agent, use_stream, model_size, job_id):
    global current_model
    if model_size != current_model:
        os.environ["MATH_AGENT_MODEL"] = model_size
        import importlib
        import config as _cfg
        importlib.reload(_cfg)
        import llm_engine as _le
        importlib.reload(_le)
        current_model = model_size
        _add_step(job_id, 'info', f'Switched to model: {model_size}')

    from llm_engine import generate, generate_code, generate_stream

    # Step 1: Memory search
    _add_step(job_id, 'memory', 'Searching memory for similar problems...')
    similar = memory.search_similar_problems(problem, limit=3)
    if similar:
        _add_step(job_id, 'memory_found', f'Found {len(similar)} similar problems', {'problems': similar})

    # Step 2: Generate code
    _add_step(job_id, 'thinking', 'Generating solution code...')
    code = detect_and_solve(problem)
    if code:
        _add_step(job_id, 'code_gen', 'Using direct code template', {'code': code, 'source': 'template'})
    else:
        _add_step(job_id, 'code_gen', 'Using LLM code generation', {'source': 'llm'})
        code = generate_code(problem)
        _add_step(job_id, 'code_gen', 'LLM generated code', {'code': code, 'source': 'llm'})

    # Step 3: Execute
    _add_step(job_id, 'executing', 'Executing Python code...')
    result = solver.solve(code, problem_context=problem)

    if not result.get('success'):
        _add_step(job_id, 'error', f'Code execution failed: {result.get("error", result.get("stderr", ""))[:200]}')
        # LLM fallback
        _add_step(job_id, 'fallback', 'Falling back to LLM reasoning...')
        prompt = f"Solve this problem with step-by-step reasoning and proofs:\n{problem}\n\nFormat as JSON: {{\"answer\": str, \"steps\": [...], \"proofs\": [...]}}"
        if use_stream:
            streamed = ""
            def on_token(tok):
                nonlocal streamed
                streamed += tok
                _add_step(job_id, 'stream_token', tok)
            raw = generate_stream(prompt, system_prompt="You are a math expert.", max_tokens=2048, temperature=0.3, on_token=on_token)
        else:
            raw = generate(prompt, system_prompt="You are a math expert.", max_tokens=2048, temperature=0.3)
        try:
            s, e = raw.find('{'), raw.rfind('}') + 1
            solution = json.loads(raw[s:e]) if s >= 0 else {"answer": raw, "method": "llm_reasoning"}
        except json.JSONDecodeError:
            solution = {"answer": raw, "method": "llm_reasoning"}
    else:
        solution = result.get("result", {})

    _add_step(job_id, 'solution_found', 'Solution computed', {'solution': solution})

    # Step 4: Verify
    _add_step(job_id, 'verifying', 'Verifying solution...')
    verification = verifier.verify(problem, solution)
    _add_step(job_id, 'verified', f'Confidence: {verification.get("confidence", 0)}', {'verification': verification})

    # Step 5: Multi-agent
    agent_discussion = None
    if use_multi_agent:
        _add_step(job_id, 'multi_agent_start', 'Starting multi-agent discussion (3 agents)...')
        from multi_agent import MultiAgentSystem
        ma = MultiAgentSystem(memory, max_agents=3)
        agent_discussion = ma.discuss(problem)
        _add_step(job_id, 'multi_agent_done', 'Multi-agent discussion complete', {'discussion': agent_discussion})

    # Step 6: Web search
    _add_step(job_id, 'web_search', 'Looking up theorems in external sources...')
    sources = {}
    try:
        from web_search import search_math_concept, format_sources
        proofs = solution.get("proofs", [])
        if proofs:
            theorem = proofs[0].get("theorem", "")
            if theorem:
                sources = search_math_concept(theorem)
                _add_step(job_id, 'web_sources', f'Found sources for: {theorem}', {'sources': sources})
    except Exception as e:
        _add_step(job_id, 'web_search_skip', f'Web search unavailable: {e}')

    # Step 7: Store in memory
    memory.store_problem(
        problem=problem,
        solution=json.dumps(solution, default=str),
        proof=json.dumps(solution.get("proofs", []), default=str),
        verified=verification.get("verified", False),
        method=solution.get("method", "code_execution")
    )
    memory.remember("user", problem)
    memory.remember("assistant", json.dumps(solution, default=str))

    # Step 8: Graph
    graph_path = None
    if solution.get("graph"):
        try:
            from graphing import make_graph
            gs = solution["graph"]
            if isinstance(gs, str):
                graph_path = make_graph(gs, title=problem[:50])
            elif isinstance(gs, dict):
                graph_path = make_graph(**gs, title=problem[:50])
            if graph_path:
                _add_step(job_id, 'graph', 'Graph generated', {'path': os.path.basename(graph_path)})
        except Exception as e:
            _add_step(job_id, 'graph_error', f'Graph failed: {e}')

    return {
        'problem': problem,
        'solution': solution,
        'verification': verification,
        'multi_agent': agent_discussion,
        'graph': os.path.basename(graph_path) if graph_path else None,
        'sources': sources,
    }


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True, threaded=True)