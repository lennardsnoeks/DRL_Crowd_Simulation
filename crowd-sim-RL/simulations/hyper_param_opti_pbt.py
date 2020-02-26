import os
import ray
from ray.tune import run, register_env
from ray.tune.schedulers import PopulationBasedTraining
from crowd_sim_RL.envs import SingleAgentEnv
from utils.steerbench_parser import XMLSimulationState
from simulations import ddpg_config


def main():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "../test_XML_files/obstacles.xml")
    sim_state = XMLSimulationState(filename).simulation_state

    train(sim_state)


def explore(config):
    # ensure we collect enough timesteps to do sgd
    if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
        config["train_batch_size"] = config["sgd_minibatch_size"] * 2
    # ensure we run at least one sgd iter
    if config["num_sgd_iter"] < 1:
        config["num_sgd_iter"] = 1
    return config


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

    pbt = PopulationBasedTraining(
        time_attr="time_total_s",
        metric="episode_reward_mean",
        mode="max",
        perturbation_interval=120,
        resample_probability=0.25,
        hyperparam_mutations={
            "gamma": [0.95, 0.99],
            "actor_hiddens": [[64, 64], [400, 300]],
            "critic_hiddens": [[64, 64], [400, 300]],
            "exploration_noise_type": ["ou", "gaussian"],
            "exploration_should_anneal": [True, False],
            "observation_filter": ["NoFilter", "MeanStdFilter"],
            "train_batch_size": [32, 64]
        },
        custom_explore_fn=explore)

    stop = {
        "training_iteration": 10
    }

    run("DDPG", scheduler=pbt, stop=stop, config=config)


if __name__ == "__main__":
    main()
