import time
import warnings
from collections import OrderedDict

import numpy as np

from matrx.actions.object_actions import *
from matrx.goals import WorldGoalV2
from matrx.agents.agent_utils.state import State
from matrx.api import api


class TickProcessor:
    """Handles state building, action validation/execution, and goal checking.

    Holds a reference to an EntityGrid for data access.
    Extracted from GridWorld so these helpers can be reused independently.
    """

    def __init__(self, entity_grid, shape, simulation_goal, world_id,
                 visualization_bg_clr, visualization_bg_img, verbose=False):
        self._entity_grid = entity_grid
        self._shape = shape
        self._simulation_goal = simulation_goal
        self._world_id = world_id
        self._visualization_bg_clr = visualization_bg_clr
        self._visualization_bg_img = visualization_bg_img
        self._verbose = verbose

    # ------------------------------------------------------------------
    # State building
    # ------------------------------------------------------------------

    def get_complete_state(self, current_nr_ticks, tick_duration):
        """Compile all objects and agents into a complete world state."""
        state_dict = {}
        for obj_id, obj in self._entity_grid.environment_objects.items():
            state_dict[obj.obj_id] = obj.properties
        for agent_id, agent in self._entity_grid.registered_agents.items():
            state_dict[agent.obj_id] = agent.properties

        state = State(own_id=None)
        state.state_update(state_dict)

        world_info = {
            "nr_ticks": current_nr_ticks,
            "curr_tick_timestamp": int(round(time.time() * 1000)),
            "grid_shape": self._shape,
            "tick_duration": tick_duration,
            "world_ID": self._world_id,
            "vis_settings": {
                "vis_bg_clr": self._visualization_bg_clr,
                "vis_bg_img": self._visualization_bg_img
            }
        }

        state._add_world_info(world_info)
        return state

    def get_agent_state(self, agent_obj, current_nr_ticks, tick_duration):
        """Build a filtered state for a specific agent based on its sense capabilities."""
        agent_loc = agent_obj.location
        sense_capabilities = agent_obj.sense_capability.get_capabilities()

        wildcard_objs = {}
        objs_in_range = OrderedDict()
        if "*" in sense_capabilities.keys():
            wildcard_objs = self._entity_grid.get_objects_in_range(
                agent_loc, "*", sense_capabilities["*"])
            sense_capabilities.pop("*")

        for obj_type, sense_range in sense_capabilities.items():
            env_objs = self._entity_grid.get_objects_in_range(
                agent_loc, obj_type, sense_range)
            objs_in_range.update(env_objs)

        for wildcard_obj_id, wildcard_obj in wildcard_objs.items():
            if type(wildcard_obj) not in sense_capabilities.keys():
                objs_in_range[wildcard_obj_id] = wildcard_obj

        state_dict = {}
        for env_obj in objs_in_range:
            state_dict[env_obj] = objs_in_range[env_obj].properties

        state = State(agent_obj.obj_id)
        state.state_update(state_dict)

        team_members = [agent_id for agent_id, other_agent
                        in self._entity_grid.registered_agents.items()
                        if agent_obj.team == other_agent.team]
        world_info = {
            "nr_ticks": current_nr_ticks,
            "curr_tick_timestamp": int(round(time.time() * 1000)),
            "grid_shape": self._shape,
            "tick_duration": tick_duration,
            "team_members": team_members,
            "world_ID": self._world_id,
            "vis_settings": {
                "vis_bg_clr": self._visualization_bg_clr,
                "vis_bg_img": self._visualization_bg_img
            }
        }

        state._add_world_info(world_info)
        return state

    def fetch_initial_states(self, registered_agents, message_manager,
                             current_nr_ticks, tick_duration, teams, matrx_info):
        """Prime the API with initial agent states (called when MATRX starts paused)."""
        for agent_id, agent_obj in registered_agents.items():
            state = self.get_agent_state(agent_obj, current_nr_ticks, tick_duration)
            filtered_agent_state = agent_obj.filter_observations(state)

            api._add_state(agent_id=agent_id, state=filtered_agent_state,
                           agent_inheritence_chain=agent_obj.class_inheritance,
                           world_settings=matrx_info)

        api._add_state(agent_id="god",
                       state=self.get_complete_state(current_nr_ticks, tick_duration),
                       agent_inheritence_chain="god",
                       world_settings=matrx_info)

        message_manager.agents = registered_agents.keys()
        message_manager.teams = teams

        api._next_tick()

    # ------------------------------------------------------------------
    # Action validation and execution
    # ------------------------------------------------------------------

    def check_action_is_possible(self, agent_id, action_name, action_kwargs, world_state):
        """Validate whether an action is possible (passed as callback to agents)."""
        if action_name is None:
            result = ActionResult(ActionResult.IDLE_ACTION, succeeded=True)
            return result

        registered_agents = self._entity_grid.registered_agents
        all_actions = self._entity_grid.all_actions

        if agent_id not in registered_agents.keys():
            result = ActionResult(
                ActionResult.AGENT_WAS_REMOVED.replace("{AGENT_ID}", agent_id), succeeded=False)
            return result

        elif action_name in all_actions.keys() and \
                action_name not in registered_agents[agent_id].action_set:
            result = ActionResult(ActionResult.AGENT_NOT_CAPABLE, succeeded=False)

        elif action_name in all_actions.keys():
            action_class = all_actions[action_name]
            action = action_class()
            # NOTE: actions expect the grid_world reference — we pass grid_world_ref
            # which is set by GridWorld before any actions execute
            result = action.is_possible(self._grid_world_ref, agent_id,
                                        world_state=world_state, **action_kwargs)

        else:
            warnings.warn(
                f"The action with name {action_name} was not found when checking whether "
                f"this action is possible to perform by agent {agent_id}.")
            result = ActionResult(ActionResult.UNKNOWN_ACTION, succeeded=False)

        return result

    def perform_action(self, agent_id, action_name, action_kwargs, world_state):
        """Execute an action if possible, apply mutations, return result to agent."""
        result = self.check_action_is_possible(agent_id, action_name, action_kwargs, world_state)

        if result.succeeded:
            if action_name is None:
                return result

            action_class = self._entity_grid.all_actions[action_name]
            action = action_class()
            # NOTE: actions expect the grid_world reference
            result = action.mutate(self._grid_world_ref, agent_id,
                                   world_state=world_state, **action_kwargs)

            self._entity_grid.update_agent_location(agent_id)

        set_action_result = self._entity_grid.registered_agents[agent_id].set_action_result_func
        set_action_result(result)

        return result

    def set_agent_busy(self, action_name, action_kwargs, agent_id, current_nr_ticks):
        """Mark an agent as busy with an action for its duration."""
        if action_name is None:
            duration_in_ticks = 0
        else:
            action_class = self._entity_grid.all_actions[action_name]
            action = action_class()

            duration_in_ticks = action.duration_in_ticks
            if "action_duration" in action_kwargs.keys():
                duration_in_ticks = action_kwargs["action_duration"]

            if "duration_in_ticks" in action_kwargs.keys():
                warnings.warn("'duration_in_ticks' is deprecated for setting an action's duration; "
                              "use 'action_duration'.", PendingDeprecationWarning)
                duration_in_ticks = action_kwargs["duration_in_ticks"]

        self._entity_grid.registered_agents[agent_id]._set_agent_busy(
            curr_tick=current_nr_ticks, action_duration=duration_in_ticks)

        self._entity_grid.registered_agents[agent_id]._set_current_action(
            action_name=action_name, action_args=action_kwargs)

    # ------------------------------------------------------------------
    # Goal checking
    # ------------------------------------------------------------------

    def check_simulation_goal(self, world_state):
        """Check if simulation goal(s) have been achieved.

        Note: Some goals call goal_reached(grid_world) — they need the
        grid_world_ref, which is set by GridWorld.
        """
        goal_status = {}
        if self._simulation_goal is not None:
            if isinstance(self._simulation_goal, (list, tuple)):
                for sim_goal in self._simulation_goal:
                    if isinstance(sim_goal, WorldGoalV2):
                        is_done = sim_goal.goal_reached(world_state, self._grid_world_ref)
                    else:
                        is_done = sim_goal.goal_reached(self._grid_world_ref)
                    goal_status[sim_goal] = is_done
            else:
                if isinstance(self._simulation_goal, WorldGoalV2):
                    is_done = self._simulation_goal.goal_reached(
                        world_state, self._grid_world_ref)
                else:
                    is_done = self._simulation_goal.goal_reached(self._grid_world_ref)
                goal_status[self._simulation_goal] = is_done

        is_done = np.array(list(goal_status.values())).all()
        return is_done, goal_status

    def set_grid_world_ref(self, grid_world):
        """Set the GridWorld back-reference needed by actions and goals.

        Actions and goals expect a GridWorld instance (they call
        grid_world.registered_agents, grid_world.get_env_object(), etc.).
        This must be called once after GridWorld creates the TickProcessor.
        """
        self._grid_world_ref = grid_world
