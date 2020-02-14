import os

from crowd_sim_RL.envs import SingleAgentEnv
from simulations import ddpg_config
from utils.steerbench_parser import XMLSimulationState


def main():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "../test_XML_files/obstacles2.xml")
    sim_state = XMLSimulationState(filename).simulation_state

    train(sim_state)


def train(sim_state):
    config = ddpg_config.DDPG_CONFIG.copy()

    config["env_config"] = {
        "sim_state": sim_state,
    }

    env_config = config["env_config"]

    single_env = SingleAgentEnv(env_config, 0)
    obs_space = single_env.get_observation_space()
    action_space = single_env.get_action_space()

    config["multiagent"] = {
        "policies": {
            "policy_0": (None, obs_space, action_space, {"gamma": 0.95})
        },
        "policy_mapping_fn": lambda agent_id: "policy_0"
    }









if __name__ == "__main__":
    main()