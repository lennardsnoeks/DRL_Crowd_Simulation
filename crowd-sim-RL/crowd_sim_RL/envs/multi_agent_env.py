import copy
from crowd_sim_RL.envs import SingleAgentEnv
from utils.steerbench_parser import SimulationState
from visualization.visualize_steerbench import VisualizationLive
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class MultiAgentEnvironment(MultiAgentEnv):

    def __init__(self, env_config):
        self.sim_state: SimulationState = env_config["sim_state"]
        self.original_sim_state = copy.deepcopy(self.sim_state)

        self.agents = []
        for agent in self.sim_state.agents:
            env_config["agent_id"] = agent.id
            self.agents.append(SingleAgentEnv(env_config))

        self.mode = env_config["mode"]
        if self.mode == "multi_train":
            self.visualizer: VisualizationLive
            self.max_step_count = env_config["timesteps_per_iteration"]
            self._set_visualizer(env_config["visualization"])

    def step(self, action_dict):
        obs, rew, done, info = {}, {}, {}, {}

        for i, action in action_dict.items():
            obs[i], rew[i], done[i], info[i] = self.agents[i].step(action)
            if done[i]:
                self.dones.add(i)

        # set done is true if one of agents is done
        done["__all__"] = len(self.dones) > 0
        # done["__all__"] = len(self.dones) == len(self.agents)

        self.render()

        return obs, rew, done, info

    def reset(self):
        self.resetted = True
        self.dones = set()
        sim_state = copy.deepcopy(self.original_sim_state)

        for agent in self.agents:
            agent.load_params(sim_state)
            
        return {i: a.reset() for i, a in enumerate(self.agents)}

    def _set_visualizer(self, visualizer: VisualizationLive):
        self.visualizer = visualizer

    def render(self):
        self.visualizer.update_agents(self.sim_state.agents)
