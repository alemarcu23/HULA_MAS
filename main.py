import argparse
import os, requests
import signal
import sys
import json
import pathlib
import time
from worlds1.WorldBuilder import create_builder
from loggers.OutputLogger import output_logger
from agents1.async_model_prompting import init_agent_pool, shutdown_agent_pool
from metrics.agent_metrics import ActionFileLogger
from metrics.simulation_metrics import SimulationMetrics

if __name__ == "__main__":
    fld = os.getcwd()

    parser = argparse.ArgumentParser(description="HULA-MAS Search and Rescue simulation")
    parser.add_argument('--preset', type=str, default=None,
                        help="World preset name (e.g. 'static', 'mild_trees', 'random')")
    parser.add_argument('--num_agents', type=int, default=None,
                        help="Number of AI rescue agents (1–5)")
    parser.add_argument('--agent_presets', type=str, default=None,
                        help="Comma-separated capability presets, e.g. 'scout,medic,generalist'")
    parser.add_argument('--agent_roles', type=str, default=None,
                        help="Comma-separated role hints, e.g. 'scout,medic,generalist'")
    parser.add_argument('--comm_strategies', type=str, default=None,
                        help="Comma-separated comm strategies, e.g. 'always_respond,busy_aware'")
    parser.add_argument('--reasoning_strategies', type=str, default=None,
                        help="Comma-separated reasoning strategies, e.g. 'react,react'")
    parser.add_argument('--planning_strategies', type=str, default=None,
                        help="Comma-separated planning strategies, e.g. 'io,io'")
    parser.add_argument('--replanning_policies', type=str, default=None,
                        help="Comma-separated replanning policies, e.g. 'every_turn,critic_gated'")
    parser.add_argument('--capability_knowledge', type=str, default=None,
                        help="'informed' or 'discovery'")
    args = parser.parse_args()

    # ── Deployment ──────────────────────────────────────────────────────────────
    # hpc_mode=True:  headless HPC run (transformers backend, no GUI)
    # hpc_mode=False: local dev    (Ollama backend, browser GUI at localhost:3000)
    hpc_mode   = bool(os.environ.get('SAR_MODEL_PATH'))
    enable_gui = not hpc_mode

    # ── LLM / Model ─────────────────────────────────────────────────────────────
    # SAR_MODEL_PATH env var overrides the HPC model path (e.g. a local checkpoint).
    llm_backend   = 'transformers' if hpc_mode else 'ollama_sdk'  # 'transformers' | 'ollama_sdk' | 'requests'
    api_base      = None if hpc_mode else "http://localhost:11434"
    _hpc_model    = os.environ.get('SAR_MODEL_PATH', 'Qwen/Qwen3-8B')
    planner_model = _hpc_model if hpc_mode else 'qwen3:8b'  # model used by EnginePlanner
    agent_model   = _hpc_model if hpc_mode else 'qwen3:8b'  # model used by RescueAgents

    # ── Simulation & World ───────────────────────────────────────────────────────
    condition     = "normal"  # human capability level: 'normal' | 'strong' | 'weak'
    name          = "humanagent"
    agent_type    = 'baseline'  # agent architecture
    include_human = False     # True adds a keyboard-controlled human agent (Arrow keys / Q W A S D E)

    ticks_per_iteration = 1200  # ticks before replanning (1200 × 0.1 s = 2 min)

    world_preset = args.preset if args.preset is not None else 'mild_trees'
    world_seed   = None      # int for reproducibility; None = random each run

    # ── Agents ──────────────────────────────────────────────────────────────────
    num_rescue_agents = args.num_agents if args.num_agents is not None else 3

    # Capability preset per agent; cycles if list is shorter than num_rescue_agents.
    # Options: 'scout', 'medic', 'heavy_lifter', 'generalist', or a custom dict.
    agent_presets = ['generalist', 'generalist', 'generalist']

    # Role hint per agent; injected into the LLM system prompt at startup.
    # Options: 'scout' | 'medic' | 'heavy_lifter' | 'rescuer' | 'generalist'
    # Agents are told their role but may adapt if the situation requires.
    agent_roles = ['generalist', 'generalist', 'generalist']

    # 'informed'  = agents know their capabilities from the start
    # 'discovery' = agents learn their capabilities by failing actions
    capability_knowledge = 'informed'

    # Communication strategy per agent; cycles if shorter than num_rescue_agents.
    # 'always_respond' | 'busy_aware'
    comm_strategies = ['always_respond', 'always_respond', 'always_respond']

    # Reasoning strategy per agent; cycles if shorter than num_rescue_agents.
    # 'io' | 'cot' | 'react' | 'reflexion' | 'self_refine' | 'self_reflective_tot'
    reasoning_strategies = ['react', 'react', 'react']

    # Planning strategy per agent; cycles if shorter than num_rescue_agents.
    # 'io' | 'deps' | 'td' | 'voyager'
    planning_strategies = ['io', 'io', 'io']

    # Replanning policy per agent; cycles if shorter than num_rescue_agents.
    # 'every_turn'  = run planner on every tick (current behavior)
    # 'critic_gated' = advance DAG on critic success, skip replan on failure,
    #                  only re-decompose when the plan is fully drained
    replanning_policies = ['every_turn', 'every_turn', 'every_turn']

    # ── CLI overrides (applied after defaults so scripts can override cleanly) ──
    if args.agent_presets:
        agent_presets = [s.strip() for s in args.agent_presets.split(',')]
    if args.agent_roles:
        agent_roles = [s.strip() for s in args.agent_roles.split(',')]
    if args.comm_strategies:
        comm_strategies = [s.strip() for s in args.comm_strategies.split(',')]
    if args.reasoning_strategies:
        reasoning_strategies = [s.strip() for s in args.reasoning_strategies.split(',')]
    if args.planning_strategies:
        planning_strategies = [s.strip() for s in args.planning_strategies.split(',')]
    if args.replanning_policies:
        replanning_policies = [s.strip() for s in args.replanning_policies.split(',')]
    if args.capability_knowledge:
        capability_knowledge = args.capability_knowledge

    # ── Planning ─────────────────────────────────────────────────────────────────
    # 'simple' = flat task list; 'dag' = task graph with conditional branching
    planning_mode = 'dag'

    # Path to a YAML file to bypass LLM task generation with hand-written plans.
    # See manual_plans.yaml for the expected format. None = use LLM mode.
    manual_plans_file = None  # e.g. "manual_plans.yaml"

    # True  = EnginePlanner agent coordinates task assignments via messages
    # False = agents self-assign tasks using their own planning module
    use_planner = False

    # ── Server ports ─────────────────────────────────────────────────────────────
    # Change these to avoid conflicts when running multiple jobs on the same node.
    api_port = 3001  # MATRX REST API
    vis_port = 3000  # browser visualizer

    # ── Logging ──────────────────────────────────────────────────────────────────
    # Each run gets its own subdirectory so concurrent runs never overwrite each other.
    # SAR_LOG_DIR overrides the base logs folder; SAR_EXPERIMENT_NAME sets the job label.
    _log_base = os.environ.get('SAR_LOG_DIR', os.path.join(fld, 'logs'))
    exp_name  = os.environ.get('SAR_EXPERIMENT_NAME', f"{agent_type}_{condition}")
    _ts       = time.strftime('%Hh-%Mm-%Ss_date_%dd-%mm-%Yy')
    log_dir   = os.path.join(_log_base, f"run_{exp_name}_at_time_{_ts}")

    # Scale LLM thread pool for the number of agents
    init_agent_pool(
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

    # Write score.json immediately so it always exists even if the run crashes early.
    with open(score_file, 'w') as f:
        json.dump({'score': 0, 'block_hit_rate': 0.0, 'victims_rescued': 0, 'total_victims': 0}, f, indent=2)

    # Per-agent action CSV log (shared by all agents, written in real-time)
    ActionFileLogger.init(os.path.join(log_dir, 'agent_actions.csv'))

    start_time = time.time()

    # Register a SIGTERM handler so HPC scheduler kills (SLURM sends SIGTERM before
    # SIGKILL) still trigger the finally block and flush metrics to disk.
    def _sigterm_handler(signum, frame):
        print("[main] SIGTERM received — flushing metrics before exit.", file=sys.stderr)
        sys.exit(1)  # raises SystemExit → triggers finally

    signal.signal(signal.SIGTERM, _sigterm_handler)

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
            agent_presets=agent_presets, agent_roles=agent_roles,
            capability_knowledge=capability_knowledge,
            comm_strategies=comm_strategies,
            reasoning_strategies=reasoning_strategies,
            planning_strategies=planning_strategies,
            replanning_policies=replanning_policies,
            world_preset=world_preset, world_seed=world_seed,
            enable_gui=enable_gui,
            planner_config=planner_config, use_planner=use_planner,
            score_file=score_file,
            log_dir=log_dir,
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

        # Aggregate and save comprehensive metrics.
        # aggregate() and save() are separated so the file is always written to disk
        # even when aggregation only partially succeeds.
        metrics_path = os.path.join(log_dir, 'simulation_metrics.json')
        sim_metrics = SimulationMetrics()
        report: dict = {'partial': True, 'error': None}
        try:
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
                    'agent_roles': agent_roles,
                    'capability_knowledge': capability_knowledge,
                    'comm_strategies': comm_strategies,
                    'reasoning_strategies': reasoning_strategies,
                    'planning_strategies': planning_strategies,
                    'replanning_policies': replanning_policies,
                    'condition': condition,
                    'include_human': include_human,
                    'ticks_per_iteration': ticks_per_iteration,
                    'use_planner': use_planner,
                },
                iteration_history=it_history,
            )
        except Exception as e:
            print(f"[main] Metrics aggregation error (will still save partial): {e}", file=sys.stderr)
            report['error'] = str(e)
        try:
            sim_metrics.save(metrics_path, report)
            print(f"Saved simulation metrics to {metrics_path}")
        except Exception as e:
            print(f"[main] Failed to write simulation_metrics.json: {e}", file=sys.stderr)

        # Enrich score.json with agent-derived metrics (best-effort)
        try:
            perf = report.get('task_performance', {})
            if perf and os.path.exists(score_file):
                with open(score_file) as f:
                    score_data = json.load(f)
                score_data['victims_found'] = perf.get('victims_found', 0)
                score_data['obstacles_removed'] = perf.get('obstacles_removed', 0)
                score_data['cells_explored'] = perf.get('cells_explored', 0)
                with open(score_file, 'w') as f:
                    json.dump(score_data, f, indent=2)
        except Exception as e:
            print(f"[main] Failed to enrich score.json: {e}", file=sys.stderr)

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

        # Close per-agent action log
        action_logger = ActionFileLogger.get()
        if action_logger:
            action_logger.close()

        # Shut down LLM thread pool
        shutdown_agent_pool()

        # Run output logger and stop builder
        try:
            output_logger(fld, log_dir=log_dir)
        except Exception as e:
            print(f"[main] Output logger error: {e}", file=sys.stderr)

        if builder is not None:
            try:
                builder.stop()
            except Exception as e:
                print(f"[main] Builder stop error: {e}", file=sys.stderr)

        print("[main] Cleanup complete.")
