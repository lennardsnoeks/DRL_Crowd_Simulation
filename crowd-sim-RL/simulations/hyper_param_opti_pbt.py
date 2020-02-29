import os
import ray
import random
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


def train(sim_state):
    config = ddpg_config.DDPG_CONFIG.copy()
    config["num_workers"] = 5
    config["num_gpus"] = 0
    config["clip_actions"] = True

    config["gamma"] = 0.99
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
            "gamma": lambda: random.uniform(0.90, 0.99),
            "actor_hiddens": [[64, 64], [400, 300]],
            "critic_hiddens": [[64, 64], [400, 300]],
            "exploration_noise_type": ["ou", "gaussian"],
            "exploration_should_anneal": [True, False],
            "observation_filter": ["NoFilter", "MeanStdFilter"],
            "train_batch_size": [32, 64]
        })

    stop = {
        "training_iteration": 10
    }

    analysis = run("DDPG", name="pbt", num_samples=4, scheduler=pbt, stop=stop, config=config)

    print("Best config: ", analysis.get_best_config(metric="episode_reward_mean"))


if __name__ == "__main__":
    main()
