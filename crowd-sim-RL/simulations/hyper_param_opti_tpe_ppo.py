import os
import ray
from ray.tune import run, register_env
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from crowd_sim_RL.envs import SingleAgentEnv
from utils.steerbench_parser import XMLSimulationState
from simulations.configs import ppo_config
from hyperopt import hp


def main():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "../test_XML_files/2-obstacles/1.xml")
    seed = 1
    sim_state = XMLSimulationState(filename, seed).simulation_state

    train(sim_state)


def train(sim_state):
    config = ppo_config.PPO_CONFIG.copy()
    config["num_workers"] = 0
    config["num_gpus"] = 0
    config["clip_actions"] = True

    config["gamma"] = 0.95
    config["num_sgd_iter"] = 3
    config["sgd_minibatch_size"] = 16
    config["train_batch_size"] = 256
    config["clip_param"] = 0.1
    config["entropy_coeff"] = 0

    config["env_config"] = {
        "sim_state": sim_state,
        "mode": "train_hyper",
        "agent_id": 0,
        "timesteps_reset": config["timesteps_per_iteration"]
    }

    register_env("single_agent_env", lambda _: SingleAgentEnv(config["env_config"]))
    config["env"] = "single_agent_env"

    ray.init()

    space = {
        "gamma": hp.choice("gamma", [0.95, 0.96, 0.97, 0.98, 0.99]),
        "num_sgd_iter": hp.choice("num_sgd_iter", [3, 5, 10]),
        "observation_filter": hp.choice("observation_filter", ["NoFilter", "MeanStdFilter"]),
        "sgd_minibatch_size": hp.choice("sgd_minibatch_size", [16, 32, 64]),
        "train_batch_size": hp.choice("train_batch_size", [256, 512, 1024, 2048]),
        "clip_param": hp.choice("clip_param", [0.1, 0.2, 0.3]),
        "entropy_coeff": hp.choice("entropy_coeff", [0, 0.05, 0.01])
    }

    current_best_params = [
        {
            "gamma": 0,
            "num_sgd_iter": 0,
            "observation_filter": 0,
            "sgd_minibatch_size": 0,
            "train_batch_size": 0,
            "clip_param": 0,
            "entropy_coeff": 0
        }
    ]

    search = HyperOptSearch(space, metric="episode_reward_mean", mode="max", points_to_evaluate=current_best_params)
    scheduler = AsyncHyperBandScheduler(metric="episode_reward_mean", mode="max")

    stop = {
        "episode_reward_mean": 137
    }

    analysis = run("PPO", name="hyper", num_samples=1, search_alg=search, scheduler=scheduler, stop=stop, config=config)

    print("Best config: ", analysis.get_best_config(metric="episode_reward_mean"))


if __name__ == "__main__":
    main()
