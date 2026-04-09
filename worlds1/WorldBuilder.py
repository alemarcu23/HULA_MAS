import os
import sys
import json
import numpy as np
from matrx import WorldBuilder
from matrx.actions import MoveNorth, OpenDoorAction, CloseDoorAction, GrabObject
from matrx.actions.move_actions import MoveEast, MoveSouth, MoveWest
from matrx.agents import AgentBrain, HumanAgentBrain, SenseCapability
from matrx.grid_world import GridWorld, AgentBody
from actions1.CustomActions import RemoveObjectTogether, DropObject, Idle, CarryObject, Drop, CarryObjectTogether, DropObjectTogether
from matrx.actions.object_actions import RemoveObject
from matrx.objects import EnvObject
from matrx.goals import WorldGoal
from agents1.agent_sar import SearchRescueAgent
from agents1.capabilities import resolve_capabilities, DEFAULT_PRESET
from memory.shared_memory import SharedMemory
from worlds1.environment_info import EnvironmentInformation
from worlds1.world_presets import (
    get_preset, to_areas_raw, generate_roof_tiles, generate_street_tiles,
    WorldPreset,
)
from brains1.HumanBrain import HumanBrain
from loggers.ActionLogger import ActionLogger
from datetime import datetime

random_seed = 1
verbose = False
# Tick duration determines the speed of the world. A tick duration of 0.1 means 10 ticks are executed in a second.
# You can speed up or slow down the world by changing this value without changing behavior. Leave this value at 0.1 during evaluations.
tick_duration = 0.1
# Define the keyboard controls for the human agent
key_action_map = {
        'ArrowUp': MoveNorth.__name__,
        'ArrowRight': MoveEast.__name__,
        'ArrowDown': MoveSouth.__name__,
        'ArrowLeft': MoveWest.__name__,
        'q': CarryObject.__name__,
        'w': Drop.__name__,
        'd': RemoveObjectTogether.__name__,
        'a': CarryObjectTogether.__name__,
        's': DropObjectTogether.__name__,
        'e': RemoveObject.__name__,
    }

# Some settings
wall_color = "#8a8a8a"
drop_off_color = "#1F262A"
object_size = 0.9
nr_teams = 1
agents_per_team = 2
human_agents_per_team = 1
agent_sense_range = 2  # the range with which agents detect other agents. Do not change this value.
object_sense_range = 1  # the range with which agents detect blocks. Do not change this value.
other_sense_range = np.inf  # the range with which agents detect other objects (walls, doors, etc.). Do not change this value.
fov_occlusion = True


def _compute_agent_starts(drop_zone_loc, grid_width):
    """Compute agent start positions near the drop zone."""
    dz_x, dz_y = drop_zone_loc
    x = max(0, dz_x - 1)
    return [(x, dz_y + i) for i in range(5)]


# Add the agents to the world
def add_agents(builder, condition, name, folder, agent_type='baseline',
               num_rescue_agents=1, include_human=True,
               api_base="http://localhost:11434", agent_model='qwen3:8b',
               planning_mode='simple', agent_presets=None,
               capability_knowledge='informed', comm_strategies=None,
               env_info=None, agent_starts=None, use_planner=True):
    """
    Add agents to the world.

    Args:
        builder: The world builder
        condition: Task difficulty condition
        name: Human agent name
        folder: Working folder path
        agent_type: Type of AI agent to use ('baseline', 'llm', or 'langgraph')
        num_rescue_agents: Number of LLM-based RescueAgents (1-5)
        include_human: Whether to add a keyboard-controlled human agent
        api_base: Base URL for the LLM inference server (shared by all agents)
        agent_model: Model name for rescue agents (e.g. 'qwen3:8b')
        planning_mode: Planning strategy for MARBLE agents ('simple' or 'dag')
        agent_presets: List of preset names or capability dicts, one per agent.
                       Defaults to all 'generalist'.
        capability_knowledge: 'informed' (agents know capabilities) or 'discovery'
                              (agents learn from failures).
        agent_starts: List of (x,y) start positions for agents.
    """
    if agent_presets is None:
        agent_presets = [DEFAULT_PRESET] * num_rescue_agents
    # Extend or truncate to match num_rescue_agents
    while len(agent_presets) < num_rescue_agents:
        agent_presets.append(agent_presets[-1] if agent_presets else DEFAULT_PRESET)

    if comm_strategies is None:
        comm_strategies = ['always_respond'] * num_rescue_agents
    while len(comm_strategies) < num_rescue_agents:
        comm_strategies.append(comm_strategies[-1] if comm_strategies else 'always_respond')

    if agent_starts is None:
        agent_starts = [(22, 11), (21, 11), (20, 11), (22, 10), (21, 10)]

    # Define the human's sense capabilities based on the selected condition
    sense_capability_human = SenseCapability({AgentBody: agent_sense_range, CollectableBlock: object_sense_range, None: other_sense_range, ObstacleObject: 1})

    agents = []
    # Shared memory for MARBLE agents (thread-safe, one instance per run)
    marble_shared_memory = SharedMemory()
    for team in range(nr_teams):
        team_name = f"Team {team}"
        # Add the artificial agents based on condition and agent_type
        for agent_nr in range(num_rescue_agents):
            agent_name = f"RescueBot{agent_nr}"
            caps = resolve_capabilities(agent_presets[agent_nr])

            # Per-agent SenseCapability with vision from capabilities
            vision_range = {'low': 1, 'medium': 2, 'high': 3}.get(caps['vision'], 2)
            sense_capability_agent = SenseCapability({
                AgentBody: agent_sense_range,
                CollectableBlock: vision_range,
                None: other_sense_range,
                ObstacleObject: vision_range,
            })

            if agent_type == 'marble':
                brain = SearchRescueAgent(
                    slowdown=8,
                    condition=condition,
                    name=name,
                    folder=folder,
                    llm_model=agent_model,
                    strategy='react',
                    include_human=include_human,
                    shared_memory=marble_shared_memory,
                    planning_mode=planning_mode,
                    api_base=api_base,
                    capabilities=caps,
                    capability_knowledge=capability_knowledge,
                    comm_strategy=comm_strategies[agent_nr],
                    env_info=env_info,
                    use_planner=use_planner,
                )
                agents.append(brain)
                print(f"[WorldBuilder] Using Agent '{agent_name}' (SearchRescueAgent, caps={caps})")

            loc = agent_starts[agent_nr % len(agent_starts)]
            builder.add_agent(loc, brain, team=team_name, name=agent_name,
                              customizable_properties=['score', 'thinking_state'],
                              score=0, thinking_state='idle',
                              capabilities=caps,
                              capability_knowledge=capability_knowledge,
                              sense_capability=sense_capability_agent,
                              is_traversable=True, img_name="/images/robot-final4.svg")

        # Add human agent (optional)
        if include_human:
            brain = HumanBrain(max_carry_objects=1, grab_range=1, drop_range=0, remove_range=1, fov_occlusion=fov_occlusion, strength=condition, name=name)
            human_loc = agent_starts[0] if agent_starts else (22, 12)
            human_loc = (human_loc[0], human_loc[1] + 1)
            builder.add_human_agent(human_loc, brain, team=team_name, name=name, key_action_map=key_action_map, sense_capability=sense_capability_human, is_traversable=True, img_name="/images/rescue-man-final3.svg", visualize_when_busy=True)
    return agents


def _apply_static_decorations(builder, overrides):
    """Apply exact decorative object coordinates from the static preset."""
    # Roof tiles
    for loc in overrides.get('roof_tiles', []):
        builder.add_object(loc, 'roof', EnvObject, is_traversable=True, is_movable=False,
                           visualize_shape='img', img_name="/images/roof-final5.svg")

    # Street tiles (main)
    for loc in overrides.get('street_tiles', []):
        builder.add_object(loc, 'street', EnvObject, is_traversable=True, is_movable=False,
                           visualize_shape='img', img_name="/images/paving-final20.svg", visualize_size=1)

    # Street tiles (alternate)
    for loc in overrides.get('street_tiles_alt', []):
        builder.add_object(loc, 'street', EnvObject, is_traversable=True, is_movable=False,
                           visualize_shape='img', img_name="/images/paving-final15.svg", visualize_size=1)

    # Plants
    for loc in overrides.get('plants', []):
        builder.add_object(loc, 'plant', EnvObject, is_traversable=True, is_movable=False,
                           visualize_shape='img', img_name="/images/tree.svg", visualize_size=1.25)

    # Decorative objects (helicopter, ambulance, etc.)
    for obj in overrides.get('decorative_objects', []):
        builder.add_object(obj['pos'], obj['name'], EnvObject,
                           is_traversable=obj.get('traversable', True), is_movable=False,
                           visualize_shape='img', img_name=obj['img'],
                           visualize_size=obj.get('size', 1))

    # Keyboard sign
    if 'keyboard_sign' in overrides:
        builder.add_object(location=list(overrides['keyboard_sign']), is_traversable=True,
                           name="keyboard sign", img_name="/images/keyboard-final.svg",
                           visualize_depth=110, visualize_size=20)

    # Area signs
    for loc, num_str, size in overrides.get('area_signs', []):
        builder.add_object(location=list(loc), is_traversable=True, is_movable=False,
                           name=f"area {num_str} sign",
                           img_name=f"/images/sign{num_str}.svg",
                           visualize_depth=110, visualize_size=size)


def _apply_auto_decorations(builder, preset):
    """Generate and apply decorative objects procedurally from room geometry."""
    # Roof tiles from wall cells
    for loc in generate_roof_tiles(preset.rooms):
        builder.add_object(loc, 'roof', EnvObject, is_traversable=True, is_movable=False,
                           visualize_shape='img', img_name="/images/roof-final5.svg")

    # Street tiles connecting rooms to drop zone
    for loc in generate_street_tiles(preset.rooms, preset.drop_zone,
                                     preset.grid_width, preset.grid_height):
        builder.add_object(loc, 'street', EnvObject, is_traversable=True, is_movable=False,
                           visualize_shape='img', img_name="/images/paving-final20.svg", visualize_size=1)


# Create the world
def create_builder(condition, name, folder, agent_type='baseline',
                   num_rescue_agents=1, include_human=True,
                   api_base="http://localhost:11434", agent_model='qwen3:8b',
                   planning_mode='simple', agent_presets=None,
                   capability_knowledge='informed', comm_strategies=None,
                   world_preset='static', world_seed=None, enable_gui=True,
                   planner_config=None, use_planner=True):
    # Set numpy's random generator
    np.random.seed(random_seed)

    # Resolve the world preset
    preset = get_preset(world_preset, seed=world_seed)
    print(f"[WorldBuilder] Using world preset '{preset.name}' "
          f"({preset.grid_width}x{preset.grid_height}, {len(preset.rooms)} rooms)")

    # Count total victims
    total_victims = sum(len(r.victims) for r in preset.rooms)

    # Create the collection goal with dynamic drop zone
    dz = preset.drop_zone
    score_file = planner_config.get('score_file') if planner_config else None
    goal = CollectionGoal(
        max_nr_ticks=np.inf,
        drop_zone_location=dz.location,
        drop_zone_height=dz.height,
        total_victims=total_victims,
        score_file=score_file,
    )

    # Create the world builder
    builder = WorldBuilder(
        shape=[preset.grid_width, preset.grid_height],
        tick_duration=tick_duration, run_matrx_api=enable_gui,
        run_matrx_visualizer=False, verbose=verbose,
        simulation_goal=goal, visualization_bg_clr='#9a9083',
    )

    # Create folders where the logs are stored during the official condition
    log_base = os.environ.get('SAR_LOG_DIR', os.path.join(os.getcwd(), 'logs'))
    current_exp_folder = datetime.now().strftime("exp_"+condition+"_at_time_%Hh-%Mm-%Ss_date_%dd-%mm-%Yy")
    logger_save_folder = os.path.join(log_base, current_exp_folder)
    builder.add_logger(ActionLogger, log_strategy=1, save_path=logger_save_folder, file_name_prefix="actions_")

    # World bounds
    builder.add_room(top_left_location=(0, 0), width=preset.grid_width,
                     height=preset.grid_height, name="world_bounds",
                     wall_visualize_colour="#1F262A")

    # Add rooms from preset
    for room in preset.rooms:
        builder.add_room(
            top_left_location=room.pos, width=room.width, height=room.height,
            name=f'area {room.id}', door_locations=[room.door], doors_open=True,
            wall_visualize_colour=wall_color, with_area_tiles=True,
            area_visualize_colour='#0008ff', area_visualize_opacity=0.0,
            door_open_colour='#9a9083',
            area_custom_properties={'doormat': room.doormat},
        )

    # Drop zone
    builder.add_area(
        dz.location, width=1, height=dz.height,
        name="Drop off 0", visualize_opacity=0.5,
        visualize_colour=drop_off_color, drop_zone_nr=0,
        is_drop_zone=True, is_goal_block=False, is_collectable=False,
    )

    # Ghost blocks (drop zone placeholders) — one per ghost victim
    for i, (gv_name, gv_img) in enumerate(preset.ghost_victims):
        builder.add_object(
            (dz.location[0], dz.location[1] + i),
            name="Collect Block", callable_class=GhostBlock,
            visualize_shape='img', img_name=gv_img, drop_zone_nr=0,
        )

    # Actual victims
    for room in preset.rooms:
        for victim in room.victims:
            builder.add_object(
                victim.location, victim.name,
                callable_class=CollectableBlock, visualize_shape='img',
                img_name=victim.img, area=victim.area,
            )

    # Obstacles
    for room in preset.rooms:
        for obs in room.obstacles:
            builder.add_object(
                obs.location, obs.name, ObstacleObject,
                visualize_shape='img', img_name=obs.img,
            )

    # Decorative objects
    if preset.decorative_overrides:
        _apply_static_decorations(builder, preset.decorative_overrides)
    else:
        _apply_auto_decorations(builder, preset)

    # Build environment info from preset
    areas_raw = to_areas_raw(preset)
    env_info = EnvironmentInformation.build(
        areas_raw=areas_raw,
        drop_zone=dz.location,
        drop_zone_height=dz.height,
        grid_size=(preset.grid_width, preset.grid_height),
        num_victims=total_victims,
    )

    # Agent start positions derived from drop zone
    agent_starts = _compute_agent_starts(dz.location, preset.grid_width)

    # Register planner agent FIRST so it acts before rescue agents each tick
    planner_brain = None
    if use_planner and planner_config is not None:
        from engine.engine_planner import EnginePlanner
        planner_brain = EnginePlanner(
            env_info=env_info,
            **planner_config,
        )
        planner_sense = SenseCapability({None: np.inf})
        builder.add_agent(
            (0, 0), planner_brain,
            team='planner', name='PlannerAgent',
            sense_capability=planner_sense,
            is_traversable=True,
            visualize_opacity=0.0,
            visualize_size=0.0,
        )

    agents = add_agents(builder, condition, name, folder, agent_type,
                        num_rescue_agents=num_rescue_agents, include_human=include_human,
                        api_base=api_base, agent_model=agent_model,
                        planning_mode=planning_mode,
                        agent_presets=agent_presets,
                        capability_knowledge=capability_knowledge,
                        comm_strategies=comm_strategies,
                        env_info=env_info,
                        agent_starts=agent_starts,
                        use_planner=use_planner)

    return builder, agents, total_victims, planner_brain


class CollectableBlock(EnvObject):
    '''
    Objects that can be collected by agents.
    '''
    def __init__(self, location, name, visualize_shape, img_name, **kwargs):
        super().__init__(location, name, is_traversable=True, is_movable=True,
                         visualize_shape=visualize_shape, img_name=img_name,
                         visualize_size=object_size, class_callable=CollectableBlock,
                         is_drop_zone=False, is_goal_block=False, is_collectable=True,
                         **kwargs)

class ObstacleObject(EnvObject):
    '''
    Obstacles that can be removed by agents
    '''
    def __init__(self, location, name, visualize_shape, img_name):
        super().__init__(location, name, is_traversable=False, is_movable=True,
                         visualize_shape=visualize_shape,img_name=img_name,
                         visualize_size=1.25, class_callable=ObstacleObject,
                         is_drop_zone=False, is_goal_block=False, is_collectable=False)

class GhostBlock(EnvObject):
    '''
    Objects on the drop zone that cannot be carried by agents.
    '''
    def __init__(self, location, drop_zone_nr, name, visualize_shape, img_name):
        super().__init__(location, name, is_traversable=True, is_movable=False,
                         visualize_shape=visualize_shape, img_name=img_name,
                         visualize_size=object_size, class_callable=GhostBlock,
                         visualize_depth=110, drop_zone_nr=drop_zone_nr, visualize_opacity=0.5,
                         is_drop_zone=False, is_goal_block=True, is_collectable=False)

class CollectionGoal(WorldGoal):
    '''
    The goal for world which determines when the simulator should stop.
    '''
    def __init__(self, max_nr_ticks, drop_zone_location=(23, 8),
                 drop_zone_height=8, total_victims=8, score_file=None):
        super().__init__()
        self.max_nr_ticks = max_nr_ticks
        self._dz_x = drop_zone_location[0]
        self._dz_y_min = drop_zone_location[1]
        self._dz_y_max = drop_zone_location[1] + drop_zone_height - 1
        self._total_victims = total_victims
        self._score_file = score_file or os.path.join('logs', 'score.json')
        self.__drop_off= {}
        self.__drop_off_zone = {}
        self.__progress = 0
        self.__score = 0

    def score(self, grid_world):
        return self.__score

    def goal_reached(self, grid_world):
        if grid_world.current_nr_ticks >= self.max_nr_ticks:
            return True
        return self.isVictimPlaced(grid_world)

    def isVictimPlaced(self, grid_world):
        '''
        @return true if all victims have been rescued
        '''
        # find all drop off locations, its tile ID's and goal victims
        if self.__drop_off =={}:
            self.__find_drop_off_locations(grid_world)
        # Go through each drop zone, and check if the victims are there on the right spot
        is_satisfied, progress = self.__check_completion(grid_world)
        # Progress in percentage
        self.__progress = progress / sum([len(goal_vics) for goal_vics in self.__drop_off.values()])
        self._write_score_json()

        return is_satisfied

    def progress(self, grid_world):
        # find all drop off locations, its tile ID's and goal blocks
        if self.__drop_off =={}:
            self.__find_drop_off_locations(grid_world)
        # Go through each drop zone, and check if the victims are there in the right spot
        is_satisfied, progress = self.__check_completion(grid_world)
        # Progress in percentage
        self.__progress = progress / sum([len(goal_vics) for goal_vics in self.__drop_off.values()])
        return self.__progress

    def _write_score_json(self):
        """Write current score data to score.json for the EnginePlanner."""
        score_data = {
            'score': self.__score,
            'block_hit_rate': self.__progress,
            'victims_rescued': len(getattr(self, '_scored_victims', set())),
            'total_victims': self._total_victims
        }
        os.makedirs(os.path.dirname(self._score_file), exist_ok=True)
        with open(self._score_file, 'w') as f:
            json.dump(score_data, f, indent=2)

    def __find_drop_off_locations(self, grid_world):
        goal_vics = {}
        all_objs = grid_world.environment_objects
        for obj_id, obj in all_objs.items():  # go through all objects
            if "drop_zone_nr" in obj.properties.keys():  # check if the object is part of a drop zone
                zone_nr = obj.properties["drop_zone_nr"]  # obtain the zone number
                if obj.properties["is_goal_block"]:  # check if the object is a ghostly goal victim
                    if zone_nr in goal_vics.keys():  # create or add to the list
                        goal_vics[zone_nr].append(obj)
                    else:
                        goal_vics[zone_nr] = [obj]

        self.__drop_off_zone = {}
        self.__drop_off = {}
        for zone_nr in goal_vics.keys():  # go through all drop of zones and fill the drop_off dict
            # Instantiate the zone's dict.
            self.__drop_off_zone[zone_nr] = {}
            self.__drop_off[zone_nr] = {}
            # Obtain the zone's goal victims.
            vics = goal_vics[zone_nr].copy()
            # The number of victims is the maximum number of victims to collect for this zone.
            max_rank = len(vics)
            # Find the 'bottom' location
            bottom_loc = (-np.inf, -np.inf)
            for vic in vics:
                if vic.location[1] > bottom_loc[1]:
                    bottom_loc = vic.location
            # Now loop through victim lists and add them to their appropriate ranks
            for rank in range(max_rank):
                loc = (bottom_loc[0], bottom_loc[1]-rank)
                # find the victim at that location
                for vic in vics:
                    if vic.location == loc:
                        # Add to self.drop_off
                        self.__drop_off_zone[zone_nr][rank] = [loc, vic.properties['img_name'][8:-4], None]
                        for i in self.__drop_off_zone.keys():
                            self.__drop_off[i] = {}
                            vals = list(self.__drop_off_zone[i].values())
                            vals.reverse()
                            for j in range(len(self.__drop_off_zone[i].keys())):
                                self.__drop_off[i][j] = vals[j]

    def __check_completion(self, grid_world):
        # Get the current tick number
        curr_tick = grid_world.current_nr_ticks

        # Track victims already scored to avoid double-counting
        if not hasattr(self, '_scored_victims'):
            self._scored_victims = set()

        # Get all objects in the world
        all_objs = grid_world.environment_objects

        # Use dynamic drop zone coordinates
        drop_zone_x = self._dz_x
        drop_zone_y_min = self._dz_y_min
        drop_zone_y_max = self._dz_y_max

        # Find all collectable victims in the drop zone
        for obj_id, obj in all_objs.items():
            if "is_collectable" not in obj.properties.keys():
                continue
            if not obj.properties["is_collectable"]:
                continue

            # Check if victim is in the drop zone
            obj_loc = obj.location
            if obj_loc[0] == drop_zone_x and drop_zone_y_min <= obj_loc[1] <= drop_zone_y_max:
                # Check if we've already scored this victim
                victim_key = f"{obj_id}_{obj.properties['img_name']}"
                if victim_key in self._scored_victims:
                    continue

                # Get victim type and award points
                img_name = obj.properties['img_name'][8:-4]

                # Only score injured victims (not healthy ones)
                if 'healthy' in img_name.lower():
                    continue

                if 'critical' in img_name.lower():
                    self.__score += 6
                    print(f"[CollectionGoal] Critical victim '{img_name}' rescued! +6 points (Total: {self.__score})")
                elif 'mild' in img_name.lower():
                    self.__score += 3
                    print(f"[CollectionGoal] Mild victim '{img_name}' rescued! +3 points (Total: {self.__score})")

                # Mark as scored
                self._scored_victims.add(victim_key)

                # Find matching goal block and mark as complete
                for zone_nr, goal_vics in self.__drop_off.items():
                    for rank, vic_data in goal_vics.items():
                        shape = vic_data[1]
                        tick = vic_data[2]
                        if shape == img_name and tick is None:
                            self.__drop_off[zone_nr][rank][2] = curr_tick
                            break

                # Remove the victim from the world (hide from UI)
                grid_world.remove_from_grid(obj_id, remove_from_carrier=True)

        # Now check if all victims are collected
        is_satisfied = True
        progress = 0
        for zone_nr, goal_vics in self.__drop_off.items():
            zone_satisfied = True
            ticks = [goal_vics[r][2] for r in range(len(goal_vics))]  # list of ticks in rank order
            for tick in ticks:
                if tick is not None:
                    progress += 1
            if None in ticks:
                zone_satisfied = False
            # update our satisfied boolean
            is_satisfied = is_satisfied and zone_satisfied

        for agent_id, agent_body in grid_world.registered_agents.items():
            if agent_id.startswith('rescuebot'):
                agent_body.change_property('score', self.__score)

        return is_satisfied, progress
