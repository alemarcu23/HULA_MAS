import os, requests
import sys
import json
import signal
import pathlib
import time
import threading
from worlds1.WorldBuilder import create_builder
from loggers.OutputLogger import output_logger
from agents1.async_model_prompting import init_marble_pool, shutdown_marble_pool
from metrics.simulation_metrics import SimulationMetrics

if __name__ == "__main__":
    fld = os.getcwd()

    # Register signal handlers for SLURM preemption / job timeout

    # Configuration
    condition = "normal"
    name = "humanagent"
    agent_type='marble'              # 'llm' | 'langgraph' | 'baseline'
    ticks_per_iteration = 1200  # 1200 ticks * 0.1s/tick = 120 seconds = 2 minutes
    num_rescue_agents = 2       # Number of LLM-based RescueAgents (1-5)
    include_human = False        # Whether to add a keyboard-controlled human agent

    # ---- Deployment mode ----
    # False = local development (Ollama + GUI)
    # True  = HPC / headless (in-process transformers, no server needed)
    hpc_mode = True

    enable_gui = not hpc_mode
    llm_backend = 'transformers' if hpc_mode else 'ollama_sdk'
    api_base = None if hpc_mode else "http://localhost:11434"
    _hpc_model = os.environ.get('SAR_MODEL_PATH', 'Qwen/Qwen3-8B')
    planner_model = _hpc_model if hpc_mode else 'qwen3:8b'
    agent_model = _hpc_model if hpc_mode else 'qwen3:8b'

    # Server ports (change to avoid conflicts when running multiple jobs on one node)
    api_port = 3001         # MATRX API server port
    vis_port = 3000         # Visualization server port

    planning_mode = 'dag'        # 'simple' (flat list) or 'dag' (task graph with conditionals)

    # Agent capability presets — one per agent (cycles if fewer than num_rescue_agents)
    # Available: 'scout', 'medic', 'heavy_lifter', 'generalist', or custom dicts
    agent_presets = ['generalist', 'generalist']
    # Whether agents know their capabilities upfront or discover through failure
    capability_knowledge = 'informed'  # 'informed' | 'discovery'

    # Communication strategies — one per agent (cycles if fewer than num_rescue_agents)
    # Available: 'always_respond', 'busy_aware'
    comm_strategies = ['always_respond', 'always_respond']

    # LLM backend is set by hpc_mode above. Override here if needed:
    # llm_backend = 'ollama_sdk'  # 'ollama_sdk' | 'requests' | 'transformers'

    # World preset: 'static' (current world), 'preset2' (2 houses), 'preset3' (2 big houses), 'random'
    world_preset = 'static'
    world_seed = None            # int for reproducibility, None for random each run

    # Set to a YAML file path to override LLM task/plan generation with manual inputs.
    # See manual_plans.yaml for the expected format. Set to None to use LLM mode.
    manual_plans_file = None  # e.g. "manual_plans.yaml"

    # When True, an EnginePlanner agent coordinates task assignments via messages.
    # When False, agents self-assign tasks using their own planning module.
    use_planner = False

    # Log directory (override with SAR_LOG_DIR env var for HPC)
    log_dir = os.environ.get('SAR_LOG_DIR', os.path.join(fld, 'logs'))

    # Scale LLM thread pool for the number of agents
    init_marble_pool(
        num_rescue_agents, backend=llm_backend,
        preload_model=agent_model if hpc_mode else None,
    )

    builder = None
    vis_thread = None
    planner_brain = None
    agents = []
    iteration_history = []

    # Initialize score.json with defaults early (planner needs path at init)
    os.makedirs(log_dir, exist_ok=True)
    score_file = os.path.join(log_dir, 'score.json')

    start_time = time.time()

    try:
        # Planner config (passed to WorldBuilder which creates the brain)
        planner_config = {
            'llm_model': planner_model,
            'ticks_per_iteration': ticks_per_iteration,
            'max_iterations': 50,
            'score_file': score_file,
            'include_human': include_human,
            'api_base': api_base,
            'manual_plans_file': manual_plans_file,
            'planning_mode': planning_mode,
        } if use_planner else None

        builder, agents, total_victims, planner_brain = create_builder(
            condition=condition, name=name, agent_type=agent_type, folder=fld,
            num_rescue_agents=num_rescue_agents, include_human=include_human,
            api_base=api_base, agent_model=agent_model,
            planning_mode=planning_mode,
            agent_presets=agent_presets, capability_knowledge=capability_knowledge,
            comm_strategies=comm_strategies,
            world_preset=world_preset, world_seed=world_seed,
            enable_gui=enable_gui,
            planner_config=planner_config, use_planner=use_planner,
        )

        # Configure MATRX API port before startup
        from matrx.api import api as matrx_api
        matrx_api.set_api_port(api_port)

        # Start overarching MATRX scripts and threads
        media_folder = pathlib.Path().resolve()
        builder.startup(media_folder=media_folder)
        if enable_gui:
            from SaR_gui import visualization_server
            print("Starting custom visualizer")
            vis_thread = visualization_server.run_matrx_visualizer(
                verbose=False, media_folder=media_folder, vis_port=vis_port
            )
        world = builder.get_world()
        print("Started world...")

        # Write initial score.json
        with open(score_file, 'w') as f:
            json.dump({
                'score': 0,
                'block_hit_rate': 0.0,
                'victims_rescued': 0,
                'total_victims': total_victims
            }, f, indent=2)

        # Run with normal MATRX loop (planner is a registered agent)
        if not include_human:
            builder.api_info['matrx_paused'] = False
        world.run(builder.api_info)

        print("DONE!")

    except Exception as e:
        print(f"[main] Simulation error: {e}", file=sys.stderr)

    finally:
        # Save final iteration history (planner also checkpoints every 1000 ticks)
        if planner_brain is not None and hasattr(planner_brain, '_save_checkpoint'):
            try:
                planner_brain._save_checkpoint()
                print(f"Saved final iteration history checkpoint")
            except Exception as e:
                print(f"Failed to save final checkpoint: {e}", file=sys.stderr)

        # Aggregate and save comprehensive metrics
        try:
            sim_metrics = SimulationMetrics()
            it_history = list(planner_brain.iteration_history) if planner_brain and hasattr(planner_brain, 'iteration_history') else []
            report = sim_metrics.aggregate(
                agents=agents if agents else [],
                planner=planner_brain,
                score_file=score_file,
                start_time=start_time,
                config={
                    'agent_type': agent_type,
                    'num_rescue_agents': num_rescue_agents,
                    'planning_mode': planning_mode,
                    'planner_model': planner_model,
                    'agent_model': agent_model,
                    'world_preset': world_preset,
                    'world_seed': world_seed,
                    'agent_presets': agent_presets,
                    'capability_knowledge': capability_knowledge,
                    'comm_strategies': comm_strategies,
                    'ticks_per_iteration': ticks_per_iteration,
                    'use_planner': use_planner,
                },
                iteration_history=it_history,
            )
            metrics_path = os.path.join(log_dir, 'simulation_metrics.json')
            sim_metrics.save(metrics_path, report)
            print(f"Saved simulation metrics to {metrics_path}")
        except Exception as e:
            print(f"[main] Failed to save simulation metrics: {e}", file=sys.stderr)

        # Shut down visualization
        if enable_gui and vis_thread is not None:
            try:
                print("Shutting down custom visualizer")
                requests.get(
                    f"http://localhost:{vis_port}/shutdown_visualizer",
                    timeout=5,
                )
                vis_thread.join(timeout=5)
            except Exception:
                pass  # daemon thread will exit with process

        # Shut down LLM thread pool
        shutdown_marble_pool()

        # Run output logger and stop builder
        try:
            output_logger(fld)
        except Exception as e:
            print(f"[main] Output logger error: {e}", file=sys.stderr)

        if builder is not None:
            try:
                builder.stop()
            except Exception as e:
                print(f"[main] Builder stop error: {e}", file=sys.stderr)

        print("[main] Cleanup complete.")
