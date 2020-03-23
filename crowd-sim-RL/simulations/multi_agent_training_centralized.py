import os
import ray
from ray.tune import register_env, run
from crowd_sim_RL.envs.multi_agent_env_centralized import MultiAgentCentralized
from utils.steerbench_parser import XMLSimulationState
from simulations.configs import ddpg_config

iterations_count = 0
iterations_max = 100
mean_max = 300


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


def on_train_result(info):
    global iterations_count, iterations_max, mean_max
    result = info["result"]
    trainer = info["trainer"]
    mean = result["episode_reward_mean"]

    # always checkpoint on last iteration or if mean reward > asked mean reward
    if iterations_count == iterations_max - 1 or mean > mean_max:
        trainer.save()
    iterations_count += 1


def train(sim_state, checkpoint):
    global iterations_max, mean_max
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
    config["callbacks"] = {
        "on_train_result": on_train_result,
    }

    register_env("multi_agent_centralized", lambda _: MultiAgentCentralized(config["env_config"]))
    config["env"] = "multi_agent_centralized"

    ray.init()

    stop = {
        "episode_reward_mean": mean_max,
        # "training_iteration": iterations_max
    }

    name = "hallway_2"
    if checkpoint == "":
        run("DDPG", name=name, checkpoint_freq=checkpoint_freq, stop=stop, config=config)
    else:
        run("DDPG", name=name, checkpoint_freq=checkpoint_freq, stop=stop, config=config, restore=checkpoint)


if __name__ == "__main__":
    main()
