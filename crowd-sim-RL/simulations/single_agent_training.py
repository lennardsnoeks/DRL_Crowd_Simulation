import os
import ray
import math
import ray.rllib.agents.ddpg as ddpg
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print
from crowd_sim_RL.envs import SingleAgentEnv
from utils.steerbench_parser import XMLSimulationState
from visualization.visualize_training import VisualizationLive
from threading import Thread
from simulations import ddpg_config, ppo_config
from ray.tune import run, register_env


def main():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "../test_XML_files/obstacles.xml")
    #filename = os.path.join(dirname, "../test_XML_files/hallway_test/hallway_single.xml")
    #seed = 16
    seed = 22222
    sim_state = XMLSimulationState(filename, seed).simulation_state

    train(sim_state)


def initial_visualization(visualization):
    visualization.run()


def train(sim_state):
    iterations = 100
    zoom_factor = 10
    visualization = VisualizationLive(sim_state, zoom_factor)

    config2 = ppo_config.PPO_CONFIG.copy()
    config = ddpg_config.DDPG_CONFIG.copy()
    config["gamma"] = 0.95
    config["num_workers"] = 0
    config["num_gpus"] = 0
    config["eager"] = False
    config["exploration_should_anneal"] = False
    config["pure_exploration_steps"] = 1000
    config["train_batch_size"] = 32
    config["exploration_noise_type"] = "ou"
    config["observation_filter"] = "MeanStdFilter"
    config["clip_actions"] = True
    config["env_config"] = {
        "sim_state": sim_state,
        "visualization": visualization,
        "mode": "train",
        "agent_id": 0,
        "timesteps_reset": 1000
    }

    """register_env("single_agent_env", lambda _: SingleAgentEnv(config["env_config"]))
    config["env"] = "single_agent_env"""

    ray.init()
    trainer = ddpg.DDPGTrainer(env=SingleAgentEnv, config=config)
    #trainer2 = ppo.PPOTrainer(env=SingleAgentEnv, config=config2)

    thread = Thread(target=initial_visualization, args=(visualization,))
    thread.start()

    reward_mean = 0
    i = 0
    while math.isnan(reward_mean) or reward_mean < 165:
        result = trainer.train()
        reward_mean = result["episode_reward_mean"]

        print(pretty_print(result))
        if i == iterations - 1:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)

        i += 1

    """stop = {
        #"training_iteration": iterations
        "episode_reward_mean": 165
    }

    run("DDPG", checkpoint_freq=iterations, stop=stop, config=config)"""

    visualization.stop()
    thread.join()


if __name__ == "__main__":
    main()
