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
    filename = os.path.join(dirname, "../test_XML_files/obstacles/obstacles.xml")
    seed = 1
    sim_state = XMLSimulationState(filename, seed).simulation_state

    train(sim_state)


def train(sim_state):
    config = ppo_config.PPO_CONFIG.copy()
    config["num_workers"] = 4
    config["num_gpus"] = 0
    config["clip_actions"] = True

    config["gamma"] = 0.95
    config["num_sgd_iter"] = 10
    config["sgd_minibatch_size"] = 128
    config["train_batch_size"] = 2000
    config["lr"] = 0.003
    config["clip_param"] = 0.1
    config["kl_coeff"] = 0.3
    config["kl_target"] = 0.003
    config["lambda"] = 0.9
    config["entropy_coeff"] = 0

    config["env_config"] = {
        "sim_state": sim_state,
        "mode": "hyper_param_opt",
        "agent_id": 0,
        "timesteps_per_iteration": config["timesteps_per_iteration"]
    }

    register_env("single_agent_env", lambda _: SingleAgentEnv(config["env_config"]))
    config["env"] = "single_agent_env"

    ray.init()

    space = {
        "gamma": hp.choice([0.95, 0.99]),
        "num_sgd_iter": hp.choice([10, 20, 30]),
        "sgd_minibatch_size": hp.choice([128, 256, 1024]),
        "train_batch_size": hp.choice([2000, 4000]),
        "lr": hp.choice([0.003, 0.0001, 0.000005]),
        "clip_param": hp.choice([0.1, 0.2, 0.3]),
        "kl_coeff": hp.choice([0.3, 0.65, 1]),
        "kl_target": hp.choice([0.003, 0.01, 0.03]),
        "lambda": hp.choice([0.9, 0.95, 1.0]),
        "entropy_coeff": hp.choice([0, 0.01])
    }

    current_best_params = [
        {
            "gamma": 0,
            "num_sgd_iter": 0,
            "sgd_minibatch_size": 0,
            "train_batch_size": 0,
            "lr": 0,
            "clip_param": 0,
            "kl_coeff": 0,
            "kl_target": 0,
            "lambda": 0,
            "entropy_coeff": 0
        }
    ]

    search = HyperOptSearch(space, metric="episode_reward_mean", mode="max", points_to_evaluate=current_best_params)
    scheduler = AsyncHyperBandScheduler(metric="episode_reward_mean", mode="max")

    stop = {
        "episode_reward_mean": 137
    }

    analysis = run("DDPG", name="tpe", num_samples=4, search_alg=search, scheduler=scheduler, stop=stop, config=config)

    print("Best config: ", analysis.get_best_config(metric="episode_reward_mean"))


if __name__ == "__main__":
    main()
