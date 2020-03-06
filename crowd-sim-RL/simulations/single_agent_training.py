import os
import ray
import ray.rllib.agents.ddpg as ddpg
from ray.tune.logger import pretty_print
from crowd_sim_RL.envs import SingleAgentEnv
from utils.steerbench_parser import XMLSimulationState
from visualization.visualize_training import VisualizationLive
from threading import Thread
from simulations import ddpg_config
from ray.tune import run, register_env


def main():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "../test_XML_files/hallway_squeeze_1.xml")
    seed = 22222
    sim_state = XMLSimulationState(filename, seed).simulation_state

    train(sim_state)


def initial_visualization(visualization):
    visualization.run()


def train(sim_state):
    iterations = 50
    visualization = VisualizationLive(sim_state)

    config = ddpg_config.DDPG_CONFIG.copy()
    config["gamma"] = 0.95
    config["num_workers"] = 0
    config["num_gpus"] = 0
    config["eager"] = False
    config["exploration_should_anneal"] = True
    config["exploration_noise_type"] = "ou"
    config["observation_filter"] = "MeanStdFilter"
    config["clip_actions"] = True
    config["env_config"] = {
        "sim_state": sim_state,
        "visualization": visualization,
        "mode": "x",
        "agent_id": 0,
        "timesteps_per_iteration": config["timesteps_per_iteration"]
    }

    register_env("single_agent_env", lambda _: SingleAgentEnv(config["env_config"]))
    config["env"] = "single_agent_env"

    ray.init()
    """trainer = ddpg.DDPGTrainer(env=SingleAgentEnv, config=config)

    thread = Thread(target=initial_visualization, args=(visualization,))
    thread.start()

    for i in range(iterations):
        result = trainer.train()
        print(pretty_print(result))

        if i == iterations - 1:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)"""

    stop = {
        "training_iteration": iterations
    }

    run("DDPG", checkpoint_freq=iterations, stop=stop, config=config)

    """visualization.stop()
    thread.join()"""


if __name__ == "__main__":
    main()
