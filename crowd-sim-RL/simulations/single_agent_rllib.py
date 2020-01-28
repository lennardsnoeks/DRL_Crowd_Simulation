import os
import ray
import ray.rllib.agents.ddpg as ddpg
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print
from crowd_sim_RL.envs import SingleAgentEnv
from utils.steerbench_parser import XMLSimulationState
from visualization.visualize_steerbench import Visualization
from threading import Thread
from simulations import DDPG_CONFIG


def main():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "../test_XML_files/obstacles.xml")
    sim_state = XMLSimulationState(filename).simulation_state

    train(sim_state)


def initial_visualization(visualization):
    visualization.run()


def train(sim_state):
    visualization = Visualization(sim_state)

    #config = ddpg.DEFAULT_CONFIG.copy()
    config = DDPG_CONFIG.DDPG_CONFIG.copy()
    #config = ppo.DEFAULT_CONFIG.copy()
    config["num_workers"] = 0

    config["num_gpus"] = 1
    config["eager"] = False
    config["observation_filter"] = "NoFilter"
    #config["batch_mode"] = "complete_episodes"
    config["clip_actions"] = True
    config["env_config"] = {
        "sim_state": sim_state,
        "visualization": visualization
    }

    ray.init()
    trainer = ddpg.DDPGTrainer(env=SingleAgentEnv, config=config)
    #trainer = ppo.PPOTrainer(env=SingleAgentEnv, config=config)

    thread = Thread(target=initial_visualization, args=(visualization,))
    thread.start()

    for i in range(25):
        result = trainer.train()
        print(pretty_print(result))

    visualization.stop()
    thread.join()


if __name__ == "__main__":
    main()
