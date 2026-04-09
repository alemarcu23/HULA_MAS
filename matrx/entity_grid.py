import os
import random
import warnings
from collections import OrderedDict

import numpy as np

from matrx.actions.object_actions import *
from matrx.objects.env_object import EnvObject
from matrx.objects.standard_objects import AreaTile
from matrx.objects.agent_body import _get_all_classes
from matrx.logger.logger import GridWorldLogger
from matrx.utils import get_distance


class EntityGrid:
    """Owns the core data stores (agents, objects, grid) and all
    entity-registry / spatial-query / grid-manipulation operations.

    Extracted from GridWorld so these helpers can be reused independently.
    """

    def __init__(self, shape, rnd_seed=1, verbose=False):
        self._shape = shape
        self._verbose = verbose

        self._registered_agents = OrderedDict()
        self._environment_objects = OrderedDict()
        self._obj_indices = {}
        self._teams = {}
        self._loggers = []

        self._rnd_seed = random.randint(1, 1000000)
        self._rnd_gen = np.random.RandomState(seed=self._rnd_seed)

        self._all_actions = _get_all_classes(Action, omit_super_class=True)

        self._grid = np.array([[None for _ in range(shape[0])] for _ in range(shape[1])])

    # ------------------------------------------------------------------
    # Public query methods (called externally by CustomActions, goals, etc.)
    # ------------------------------------------------------------------

    def get_env_object(self, requested_id, obj_type=None):
        """Fetch an object or agent by ID, optionally filtering by type."""
        obj = None

        if requested_id in self._registered_agents:
            if obj_type is not None:
                if isinstance(self._registered_agents[requested_id], obj_type):
                    obj = self._registered_agents[requested_id]
            else:
                obj = self._registered_agents[requested_id]

        if requested_id in self._environment_objects:
            if obj_type is not None:
                if isinstance(self._environment_objects[requested_id], obj_type):
                    obj = self._environment_objects[requested_id]
            else:
                obj = self._environment_objects[requested_id]

        return obj

    def get_objects_in_range(self, agent_loc, object_type, sense_range):
        """Get all objects of a specific type within *sense_range* of *agent_loc*."""
        env_objs = OrderedDict()

        for obj_id, env_obj in self._environment_objects.items():
            coordinates = env_obj.location
            distance = get_distance(coordinates, agent_loc)
            if (object_type is None or object_type == "*" or isinstance(env_obj, object_type)) and \
                    distance <= sense_range:
                env_objs[obj_id] = env_obj

        for agent_id, agent_obj in self._registered_agents.items():
            coordinates = agent_obj.location
            distance = get_distance(coordinates, agent_loc)
            if (object_type is None or object_type == "*" or isinstance(agent_obj, object_type)) and \
                    distance <= sense_range:
                env_objs[agent_id] = agent_obj

        return env_objs

    def remove_from_grid(self, object_id, remove_from_carrier=True):
        """Remove an object/agent from the grid and registries."""
        grid_obj = self.get_env_object(object_id)
        loc = grid_obj.location

        self._grid[loc[1], loc[0]].remove(grid_obj.obj_id)
        if len(self._grid[loc[1], loc[0]]) == 0:
            self._grid[loc[1], loc[0]] = None

        if object_id in self._registered_agents:
            for obj_id in self._registered_agents[object_id].is_carrying:
                self._environment_objects[obj_id].carried_by.remove(object_id)
            success = self._registered_agents.pop(object_id, default=False)

        elif object_id in self._environment_objects:
            if remove_from_carrier:
                for agent_id in self._environment_objects[object_id].carried_by:
                    obj = self._environment_objects[object_id]
                    self._registered_agents[agent_id].is_carrying.remove(obj)
            success = self._environment_objects.pop(object_id, default=False)
        else:
            success = False

        if success is not False:
            success = True

        if self._verbose:
            if success:
                print(f"@{os.path.basename(__file__)}: Succeeded in removing object with ID {object_id}")
            else:
                print(f"@{os.path.basename(__file__)}: Failed to remove object with ID {object_id}.")

        return success

    # ------------------------------------------------------------------
    # Registration methods (called by WorldBuilder via GridWorld delegates)
    # ------------------------------------------------------------------

    def register_agent(self, agent, agent_body, check_action_possible_callback):
        """Register an agent (human or AI) to the grid."""
        agent_seed = self._rnd_gen.randint(1, 1000000)

        self._validate_obj_placement(agent_body)

        self._registered_agents[agent_body.obj_id] = agent_body

        if self._verbose:
            print(f"@{os.path.basename(__file__)}: Created agent with id {agent_body.obj_id}.")

        avatar_props = agent_body.properties

        if agent_body.is_human_agent is False:
            agent._factory_initialise(agent_name=agent_body.obj_name,
                                      agent_id=agent_body.obj_id,
                                      action_set=agent_body.action_set,
                                      sense_capability=agent_body.sense_capability,
                                      agent_properties=avatar_props,
                                      callback_is_action_possible=check_action_possible_callback,
                                      rnd_seed=agent_seed)
        else:
            agent._factory_initialise(agent_name=agent_body.obj_name,
                                      agent_id=agent_body.obj_id,
                                      action_set=agent_body.action_set,
                                      sense_capability=agent_body.sense_capability,
                                      agent_properties=avatar_props,
                                      callback_is_action_possible=check_action_possible_callback,
                                      rnd_seed=agent_seed,
                                      key_action_map=agent_body.properties["key_action_map"])

        return agent_body.obj_id

    def register_env_object(self, env_object, ensure_unique_id=True):
        """Register a non-agent environment object."""
        self._validate_obj_placement(env_object)

        if ensure_unique_id:
            env_object.obj_id = self._ensure_unique_obj_name(env_object.obj_id)

        self._environment_objects[env_object.obj_id] = env_object

        if self._verbose:
            print(f"@{__file__}: Created an environment object with id {env_object.obj_id}.")

        return env_object.obj_id

    def register_teams(self):
        """Register all teams from registered agents."""
        for agent_id, agent_body in self._registered_agents.items():
            team = agent_body.properties['team']
            if team not in self._teams:
                self._teams[team] = []
            self._teams[team].append(agent_id)

    def register_logger(self, logger):
        """Append a logger to the loggers list."""
        if self._loggers is None:
            self._loggers = [logger]
        else:
            self._loggers.append(logger)

    # ------------------------------------------------------------------
    # Grid manipulation (internal helpers, also called from TickProcessor)
    # ------------------------------------------------------------------

    def add_to_grid(self, grid_obj):
        """Place an EnvObject / AgentBody on the sparse grid."""
        if isinstance(grid_obj, EnvObject):
            loc = grid_obj.location
            if self._grid[loc[1], loc[0]] is not None:
                self._grid[loc[1], loc[0]].append(grid_obj.obj_id)
            else:
                self._grid[loc[1], loc[0]] = [grid_obj.obj_id]
        else:
            raise BaseException(
                f"Object is not of type {str(type(EnvObject))} but of {str(type(grid_obj))} "
                f"when adding to grid in EntityGrid.")

    def update_grid(self):
        """Rebuild the entire sparse grid from current positions."""
        self._grid = np.array([[None for _ in range(self._shape[0])] for _ in range(self._shape[1])])
        for obj_id, obj in self._environment_objects.items():
            self.add_to_grid(obj)
        for agent_id, agent in self._registered_agents.items():
            self.add_to_grid(agent)

    def update_agent_location(self, agent_id):
        """Update an agent's position on the sparse grid."""
        loc = self._registered_agents[agent_id].location
        if self._grid[loc[1], loc[0]] is not None:
            self._grid[loc[1], loc[0]].append(agent_id)
        else:
            self._grid[loc[1], loc[0]] = [agent_id]
        self._registered_agents[agent_id].location = loc

    def update_obj_location(self, obj_id):
        """Update an object's position on the sparse grid."""
        loc = self._environment_objects[obj_id].location
        if self._grid[loc[1], loc[0]] is not None:
            self._grid[loc[1], loc[0]].append(obj_id)
        else:
            self._grid[loc[1], loc[0]] = [obj_id]

    def _validate_obj_placement(self, env_object):
        """Check that an object can be placed at its target location."""
        obj_loc = env_object.location
        objs_at_loc = self.get_objects_in_range(obj_loc, "*", 0)

        for key in list(objs_at_loc.keys()):
            if AreaTile.__name__ in objs_at_loc[key].class_inheritance:
                objs_at_loc.pop(key)

        intraversable_objs = []
        for obj in objs_at_loc:
            if not objs_at_loc[obj].is_traversable:
                intraversable_objs.append(objs_at_loc[obj].obj_id)

        if not env_object.is_traversable and len(intraversable_objs) > 0:
            raise Exception(
                f"Invalid placement. Could not place object {env_object.obj_id} in grid, location already "
                f"occupied by intraversable object {intraversable_objs} at location {obj_loc}")

    def _ensure_unique_obj_name(self, obj_id):
        """Ensure every object ID is unique by appending incrementing counts."""
        if obj_id in self._obj_indices:
            n = self._obj_indices[obj_id]
            self._obj_indices[obj_id] += 1

            while f"{obj_id}_{n}" in self._obj_indices:
                n = self._obj_indices[obj_id]
                self._obj_indices[obj_id] += 1

            obj_id = f"{obj_id}_{n}"
        else:
            self._obj_indices[obj_id] = 1

        return obj_id

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def registered_agents(self):
        return self._registered_agents

    @property
    def environment_objects(self):
        return self._environment_objects

    @property
    def grid(self):
        return self._grid

    @property
    def shape(self):
        return self._shape

    @property
    def teams(self):
        return self._teams

    @property
    def loggers(self):
        return self._loggers

    @property
    def all_actions(self):
        return self._all_actions

    @property
    def rnd_gen(self):
        return self._rnd_gen
