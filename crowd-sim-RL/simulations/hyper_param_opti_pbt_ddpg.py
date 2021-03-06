import os
import ray
from ray.tune import run, register_env
from ray.tune.schedulers import PopulationBasedTraining
from crowd_sim_RL.envs import SingleAgentEnv
from utils.steerbench_parser import XMLSimulationState
from simulations.configs import ddpg_config


def main():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "../test_XML_files/2-obstacles/1.xml")
    seed = 1
    sim_state = XMLSimulationState(filename, seed).simulation_state

    train(sim_state)


def train(sim_state):
    config = ddpg_config.DDPG_CONFIG.copy()
    config["num_workers"] = 0
    config["num_gpus"] = 0
    config["clip_actions"] = True

    config["gamma"] = 0.95
    config["actor_hiddens"] = [64, 64]
    config["critic_hiddens"] = [64, 64]
    config["observation_filter"] = "NoFilter"
    config["train_batch_size"] = 16
    config["sample_batch_size"] = 1

    config["env_config"] = {
        "sim_state": sim_state,
        "mode": "train_hyper",
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
            "gamma": [0.95, 0.96, 0.97, 0.98, 0.99],
            "actor_hiddens": [[64, 64], [400, 300]],
            "critic_hiddens": [[64, 64], [400, 300]],
            "train_batch_size": [16, 32, 64],
            "sample_batch_size": [1, 8, 16],
            "observation_filter": ["NoFilter", "MeanStdFilter"]
        })

    stop = {
        "episode_reward_mean": 137
    }

    analysis = run("DDPG", name="pbt", num_samples=4, scheduler=pbt, stop=stop, config=config)

    print("Best config: ", analysis.get_best_config(metric="episode_reward_mean"))


if __name__ == "__main__":
    main()
