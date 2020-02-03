import os
import ray
import ray.rllib.agents.ddpg as ddpg
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print
from crowd_sim_RL.envs import SingleAgentEnv
from utils.steerbench_parser import XMLSimulationState
from visualization.visualize_simulation import VisualizationSim
from visualization.visualize_steerbench import VisualizationLive
from threading import Thread
from simulations import ddpg_config


def main():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "../test_XML_files/obstacles.xml")
    sim_state = XMLSimulationState(filename).simulation_state

    train(sim_state)


def initial_visualization(visualization):
    visualization.run()


def train(sim_state):
    visualization = VisualizationLive(sim_state)

    #config = ddpg.DEFAULT_CONFIG.copy()
    config = ddpg_config.DDPG_CONFIG.copy()
    #config = ppo.DEFAULT_CONFIG.copy()
    config["gamma"] = 0.95
    config["num_workers"] = 0
    config["num_gpus"] = 1
    config["eager"] = False
    config["observation_filter"] = "NoFilter"
    #config["batch_mode"] = "complete_episodes"
    config["clip_actions"] = True
    config["env_config"] = {
        "sim_state": sim_state,
        "visualization": visualization,
        "mode": "train",
        "timesteps_per_iteration": config["timesteps_per_iteration"]
    }

    ray.init()
    trainer = ddpg.DDPGTrainer(env=SingleAgentEnv, config=config)
    #trainer = ppo.PPOTrainer(env=SingleAgentEnv, config=config)

    thread = Thread(target=initial_visualization, args=(visualization,))
    thread.start()

    for i in range(1):
        result = trainer.train()
        print(pretty_print(result))
        if i == 9:
            checkpoint = trainer.save()

    """trainer = ddpg.DDPGTrainer(env=SingleAgentEnv, config=config)
    trainer.restore("/home/lennard/ray_results/DDPG_SingleAgentEnv_2020-01-29_23-59-26rqx4q006/checkpoint_10/checkpoint-10")"""

    visualization_sim = VisualizationSim(sim_state, trainer)
    thread2 = Thread(target=initial_visualization, args=(visualization_sim,))
    thread2.start()

    visualization.stop()
    thread.join()

    visualization_sim.stop()
    thread2.join()


if __name__ == "__main__":
    main()
