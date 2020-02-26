import os
import ray
from ray.tune import run, register_env
from ray.tune.suggest.hyperopt import HyperOptSearch
from crowd_sim_RL.envs import SingleAgentEnv
from utils.steerbench_parser import XMLSimulationState
from simulations import ddpg_config
from hyperopt import hp


def main():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "../test_XML_files/obstacles.xml")
    sim_state = XMLSimulationState(filename).simulation_state

    train(sim_state)


def train(sim_state):
    config = ddpg_config.DDPG_CONFIG.copy()
    config["num_workers"] = 0
    config["num_gpus"] = 1
    config["clip_actions"] = True

    config["gamma"] = 0.95
    config["exploration_should_anneal"] = True
    config["exploration_noise_type"] = "ou"
    config["observation_filter"] = "NoFilter"
    config["train_batch_size"] = 32

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
        "actor_hiddens": hp.choice("actor_hiddens", [[64, 64], [400,300]]),
        "critic_hiddens": hp.choice("critic_hiddens", [[64, 64], [400,300]]),
        "exploration_noise_type": hp.choice("exploration_noise_type", ["ou", "gaussian"]),
        "exploration_should_anneal": hp.choice("exploration_should_anneal", [True, False]),
        "observation_filter": hp.choice("observation_filter", ["NoFilter", "MeanStdFilter"]),
        "train_batch_size": hp.choice("train_batch_size", [32, 64])
    }

    current_best_params = [
        {
            "gamma": 0.95,
            "actor_hiddens": 0,
            "critic_hiddens": 0,
            "exploration_noise_type": 0,
            "exploration_should_anneal": 0,
            "observation_filter": 0,
            "train_batch_size": 0
        }
    ]

    search = HyperOptSearch(space, metric="episode_reward_mean", mode="max", points_to_evaluate=current_best_params)

    stop = {
        "training_iteration": 25
    }

    run("DDPG", search_alg=search, stop=stop, config=config)


if __name__ == "__main__":
    main()
