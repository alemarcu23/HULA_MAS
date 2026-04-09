# How to Run the Simulation

## 1. Install Dependencies

```bash
pip install -r requirements.txt
pip install requests pyyaml
```

## 2. Pull Ollama Models

The EnginePlanner uses a larger model (`gemma3:27b`) for task planning and summarization.
Each RescueAgent uses `llama3:8b` for per-tick reasoning.

```bash
ollama pull llama3:8b       # Used by RescueAgents
ollama pull gemma3:27b      # Used by EnginePlanner
ollama pull qwen3:8b
```

You can change which model the planner uses by editing `planner_model` in `main.py`.

## 3. Start Ollama Instances

Each agent needs its own Ollama instance. The EnginePlanner shares Terminal 1 with Agent 0.

```bash
# Terminal 1 (default port 11434 — used by EnginePlanner + Agent 0)
ollama serve

BASE_port = 11434 

# Terminal 2 (port 11435 — used by Agent 1)
OLLAMA_HOST=0.0.0.0:11435 ollama serve

# Terminal 3 (port 11436 — used by Agent 2, only if num_rescue_agents=3)
OLLAMA_HOST=0.0.0.0:11436 ollama serve
```

Add one terminal per additional agent: Agent N uses port `11434 + N`.

## 4. Run the Simulation

```bash
python main.py
```

Then open http://localhost:3000 (Ctrl+F5 to clear cache).
Use the "God" view to start, or the human agent view for keyboard control.

## 5. Configuration Reference (main.py)

| Variable              | Default        | Description                                      |
|-----------------------|----------------|--------------------------------------------------|
| `condition`           | `"normal"`     | Human condition: `"normal"`, `"strong"`, `"weak"` |
| `name`                | `"humanagent"` | Human agent name                                 |
| `agent_type`          | `'llm'`        | `'llm'` (RescueAgent) or `'baseline'` (OfficialAgent) |
| `ticks_per_iteration` | `1200`         | MATRX ticks per planning iteration               |
| `num_rescue_agents`   | `2`            | Number of AI RescueAgent instances (1–5)         |
| `include_human`       | `False`        | Add a keyboard-controlled human agent            |
| `ollama_base_port`    | `11434`        | Agent N uses port `base + N`                     |
| `planner_model`       | `'gemma3:27b'` | Ollama model for the EnginePlanner               |
