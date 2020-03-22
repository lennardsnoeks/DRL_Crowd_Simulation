import os
import ray
from ray.rllib.agents.ddpg.ddpg_policy import DDPGTFPolicy
from ray.tune import register_env, run
from crowd_sim_RL.envs import SingleAgentEnv
from crowd_sim_RL.envs.multi_agent_env import MultiAgentEnvironment
from utils.steerbench_parser import XMLSimulationState
from simulations.configs import ddpg_config


def main():
    filename = "hallway_2"
    sim_state = parse_sim_state(filename)

    checkpoint = ""

    train(sim_state, checkpoint)


def parse_sim_state(filename):
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "../test_XML_files/hallway_test/" + filename + ".xml")
    seed = 22222
    sim_state = XMLSimulationState(filename, seed).simulation_state

    return sim_state


def initial_visualization(visualization):
    visualization.run()


def make_multi_agent_config(sim_state, config):
    multi_agent_config = {}
    policy_dict = {}

    env_config = config["env_config"]
    env_config["agent_id"] = 0

    single_env = SingleAgentEnv(env_config)
    obs_space = single_env.get_observation_space()
    action_space = single_env.get_action_space()

    for agent in sim_state.agents:
        policy_id = "policy_" + str(agent.id)
        policy_dict[policy_id] = (DDPGTFPolicy, obs_space, action_space, {"gamma": 0.95})

    multi_agent_config["policies"] = policy_dict
    multi_agent_config["policy_mapping_fn"] = lambda agent_id: "policy_" + str(agent_id)

    return multi_agent_config


def train(sim_state, checkpoint):
    iterations = 100
    checkpoint_freq = 10

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
    config["env_config"] = {
        "sim_state": sim_state,
        "mode": "multi_train_vis",
        "timesteps_reset": config["timesteps_per_iteration"]
    }

    multi_agent_config = make_multi_agent_config(sim_state, config)
    config["multiagent"] = multi_agent_config

    register_env("multi_agent_env", lambda _: MultiAgentEnvironment(config["env_config"]))
    config["env"] = "multi_agent_env"

    ray.init()

    stop = {
        "episode_reward_mean": 260,
        # "training_iteration": iterations
    }

    name = "hallway_2"
    if checkpoint == "":
        run("DDPG", name=name, checkpoint_freq=checkpoint_freq, stop=stop, config=config)
    else:
        run("DDPG", name=name, checkpoint_freq=checkpoint_freq, stop=stop, config=config, restore=checkpoint)


if __name__ == "__main__":
    main()
