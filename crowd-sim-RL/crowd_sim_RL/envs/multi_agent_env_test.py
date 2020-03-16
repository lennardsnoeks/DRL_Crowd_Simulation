import os
from crowd_sim_RL.envs import SingleAgentEnv
from simulations import ddpg_config
from utils.steerbench_parser import XMLSimulationState
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune import run


class MultiAgentEnvironmentTest(MultiAgentEnv):

    def __init__(self, _):
        self.sim_state = self.get_config()

        env_config = {
            "sim_state": self.sim_state,
            "mode": "train_test",
            "agent_id": 0
        }

        self.agent = SingleAgentEnv(env_config)
        self.observation_space = self.agent.get_observation_space()
        self.action_space = self.agent.get_action_space()

    def get_config(self):
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, "../../test_XML_files/hallway_test/hallway_single.xml")
        seed = 22222
        return XMLSimulationState(filename, seed).simulation_state

    def step(self, actions):
        a = actions["agent1"]
        obs, rew, done, info = self.agent.step(a)

        return {"agent1": obs}, {"agent1": rew}, {"__all__": done}, {}

    def reset(self):
        return {"agent1": self.agent.reset()}


def main():
    iterations = 100
    checkpoint_freq = 100

    config = ddpg_config.DDPG_CONFIG.copy()
    config["gamma"] = 0.95
    config["num_workers"] = 0
    config["num_gpus"] = 0
    config["eager"] = False
    config["exploration_should_anneal"] = False
    config["schedule_max_timesteps"] = 100000
    config["exploration_noise_type"] = "ou"
    config["observation_filter"] = "MeanStdFilter"
    config["clip_actions"] = True
    config["env"] = MultiAgentEnvironmentTest

    stop = {"training_iteration": iterations}
    run("DDPG", checkpoint_freq=checkpoint_freq, stop=stop, config=config)


if __name__ == "__main__":
    main()
