import os
import ray
from ray.tune import register_env, run
from crowd_sim_RL.envs.multi_agent_env_centralized import MultiAgentCentralized
from utils.steerbench_parser import XMLSimulationState
from simulations.configs import ddpg_config, ppo_config

iterations_count = 0
iterations_max = 100
mean_max = 550


def main():
    filename = "4-hallway/4"
    sim_state = parse_sim_state(filename)

    checkpoint = ""

    train(sim_state, checkpoint)


def parse_sim_state(filename):
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "../test_XML_files/training/" + filename + ".xml")
    seed = 22222
    sim_state = XMLSimulationState(filename, seed).simulation_state

    return sim_state


def on_train_result(info):
    global iterations_count, iterations_max, mean_max
    result = info["result"]
    trainer = info["trainer"]
    mean = result["episode_reward_mean"]

    # always checkpoint on last iteration or if mean reward > asked mean reward
    """if iterations_count == iterations_max - 1 or mean > mean_max:
        trainer.save()"""

    if mean > mean_max:
        trainer.save()

    iterations_count += 1


def train(sim_state, checkpoint):
    global iterations_max, mean_max
    checkpoint_freq = 0

    #config = ddpg_config.DDPG_CONFIG.copy()
    config = ppo_config.PPO_CONFIG.copy()

    config["gamma"] = 0.99
    config["num_workers"] = 7
    config["num_gpus"] = 0
    config["eager"] = False
    config["observation_filter"] = "MeanStdFilter"
    config["metrics_smoothing_episodes"] = 20
    config["clip_actions"] = True
    config["env_config"] = {
        "sim_state": sim_state,
        "mode": "train",
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

    name = "hallway"
    algo = "PPO"    # Options: DDPG, PPO, TD3

    if checkpoint == "":
        run(algo, num_samples=1, name=name, checkpoint_freq=checkpoint_freq, stop=stop, config=config)
    else:
        run(algo, name=name, checkpoint_freq=checkpoint_freq, stop=stop, config=config, restore=checkpoint)


if __name__ == "__main__":
    main()
