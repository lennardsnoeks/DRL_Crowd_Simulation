import os
import ray
from ray.tune import register_env, run
from crowd_sim_RL.envs.multi_agent_env_centralized import MultiAgentCentralized
from utils.steerbench_parser import XMLSimulationState
from simulations import ddpg_config


def main():
    filename = "hallway_single"
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
        "mode": "train_vis",
        "timesteps_reset": config["timesteps_per_iteration"]
    }

    register_env("multi_agent_centralized", lambda _: MultiAgentCentralized(config["env_config"]))
    config["env"] = "multi_agent_centralized"

    ray.init()

    stop = {
        "episode_reward_mean": 305,
        # "training_iteration": iterations
    }

    name = "hallway_2"
    if checkpoint == "":
        run("DDPG", name=name, checkpoint_freq=checkpoint_freq, stop=stop, config=config)
    else:
        run("DDPG", name=name, checkpoint_freq=checkpoint_freq, stop=stop, config=config, restore=checkpoint)


if __name__ == "__main__":
    main()
