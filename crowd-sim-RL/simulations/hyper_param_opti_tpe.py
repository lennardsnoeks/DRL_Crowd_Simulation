import os
import ray
from ray.tune import run, register_env
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from crowd_sim_RL.envs import SingleAgentEnv
from utils.steerbench_parser import XMLSimulationState
from simulations import ddpg_config
from hyperopt import hp


def main():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "../test_XML_files/hallway_test/hallway_single.xml")
    seed = 22222
    sim_state = XMLSimulationState(filename, seed).simulation_state

    train(sim_state)


def train(sim_state):
    config = ddpg_config.DDPG_CONFIG.copy()
    config["num_workers"] = 4
    config["num_gpus"] = 0
    config["clip_actions"] = True

    config["gamma"] = 0.95
    config["exploration_should_anneal"] = False
    config["exploration_noise_type"] = "ou"
    config["observation_filter"] = "NoFilter"
    config["train_batch_size"] = 16

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
        "gamma": hp.uniform("gamma", 0.95, 0.99),
        "actor_hiddens": hp.choice("actor_hiddens", [[64, 64], [400, 300]]),
        "critic_hiddens": hp.choice("critic_hiddens", [[64, 64], [400, 300]]),
        #"exploration_noise_type": hp.choice("exploration_noise_type", ["ou", "gaussian"]),
        #"exploration_should_anneal": hp.choice("exploration_should_anneal", [True, False]),
        "observation_filter": hp.choice("observation_filter", ["NoFilter", "MeanStdFilter"]),
        "train_batch_size": hp.choice("train_batch_size", [16, 32, 64])
    }

    current_best_params = [
        {
            "gamma": 0.95,
            "actor_hiddens": 0,
            "critic_hiddens": 0,
            #"exploration_noise_type": 0,
            #"exploration_should_anneal": 0,
            "observation_filter": 0,
            "train_batch_size": 0
        }
    ]

    search = HyperOptSearch(space, metric="episode_reward_mean", mode="max", points_to_evaluate=current_best_params)
    #search = HyperOptSearch(space, metric="episode_reward_mean", mode="max")

    scheduler = AsyncHyperBandScheduler(metric="episode_reward_mean", mode="max")

    stop = {
        "episode_reward_mean": 175
    }

    analysis = run("DDPG", name="tpe", num_samples=4, search_alg=search, scheduler=scheduler, stop=stop, config=config)

    print("Best config: ", analysis.get_best_config(metric="episode_reward_mean"))


if __name__ == "__main__":
    main()
