import copy
from threading import Thread
from crowd_sim_RL.envs import SingleAgentEnv
from utils.steerbench_parser import SimulationState
from visualization.visualize_training import VisualizationLive
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class MultiAgentEnvironment(MultiAgentEnv):

    def __init__(self, env_config):
        self.first_time = True

        self.visualizer: VisualizationLive
        self.sim_state: SimulationState = env_config["sim_state"]
        self.env_config = env_config

        self.load_agents()

        self.mode = env_config["mode"]
        self.max_step_count = env_config["timesteps_per_iteration"]

    def step(self, action_dict):
        if self.first_time and "vis" in self.mode:
            self.setup_visualization()

        obs, rew, done, info = {}, {}, {}, {}

        for i, action in action_dict.items():
            obs[i], rew[i], done[i], info[i] = self.agents[i].step(action)
            if done[i]:
                self.dones.add(i)

        done["__all__"] = len(self.dones) > 0
        # done["__all__"] = len(self.dones) == len(self.agents)

        if "vis" in self.mode:
            self.render()

        return obs, rew, done, info

    def reset(self):
        self.resetted = True
        self.dones = set()
        self.sim_state = copy.deepcopy(self.original_sim_state)

        for agent in self.agents:
            agent.load_params(self.sim_state)

        return {i: a.reset() for i, a in enumerate(self.agents)}

    def initial_visualization(self, visualization):
        visualization.run()

    def setup_visualization(self):
        zoom_factor = 10
        visualization = VisualizationLive(self.sim_state, zoom_factor)
        thread = Thread(target=self.initial_visualization, args=(visualization,))
        thread.start()
        self._set_visualizer(visualization)
        self.first_time = False

    def load_agents(self):
        self.original_sim_state = copy.deepcopy(self.sim_state)

        self.agents = []
        for i in range(0, len(self.sim_state.agents)):
            self.env_config["agent_id"] = i
            self.agents.append(SingleAgentEnv(self.env_config))

    def set_phase(self, new_sim_state):
        self.sim_state = new_sim_state
        self.env_config["sim_state"] = self.sim_state
        self.load_agents()

    def _set_visualizer(self, visualizer: VisualizationLive):
        self.visualizer = visualizer

    def render(self):
        self.visualizer.update_agents(self.sim_state.agents)

    def get_agents(self):
        return self.sim_state.agents
