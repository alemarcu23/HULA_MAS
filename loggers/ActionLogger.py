from matrx.logger.logger import GridWorldLogger
from matrx.grid_world import GridWorld

class ActionLogger(GridWorldLogger):
    '''
    Logger for saving the actions of all agents during each tick of the task.
    '''
    def __init__(self, save_path="", file_name_prefix="", file_extension=".csv", delimiter=";"):
        super().__init__(save_path=save_path, file_name=file_name_prefix, file_extension=file_extension, delimiter=delimiter, log_strategy=1)

    def log(self, grid_world, agent_data):
        log_data = {}
        log_data['score'] = grid_world.simulation_goal.score(grid_world)
        log_data['completeness'] = grid_world.simulation_goal.progress(grid_world)
        for agent_id, agent_body in grid_world.registered_agents.items():
            log_data[agent_id + '_action'] = agent_body.current_action
            log_data[agent_id + '_location'] = agent_body.location

        # Skip rows where every agent is idle — keeps the file small
        _idle = {None, '', 'Idle'}
        all_idle = all(
            log_data.get(aid + '_action') in _idle
            for aid in grid_world.registered_agents
        )
        if all_idle:
            return None

        return log_data