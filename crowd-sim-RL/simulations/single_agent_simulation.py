import os
import ray
import ray.rllib.agents.ddpg as ddpg
from crowd_sim_RL.envs import SingleAgentEnv
from utils.steerbench_parser import XMLSimulationState
from visualization.visualize_simulation import VisualizationSim
from simulations import ddpg_config


def main():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "../test_XML_files/obstacles2.xml")
    sim_state = XMLSimulationState(filename).simulation_state

    checkpoint_path = "/home/lennard/ray_results/DDPG_SingleAgentEnv_2020-02-05_21-41-50gmsa3arh/checkpoint_1/checkpoint-1"
    simulate(sim_state, checkpoint_path)


def simulate(sim_state, checkpoint_path):
    config = ddpg_config.DDPG_CONFIG.copy()
    config["gamma"] = 0.95
    config["num_workers"] = 0
    config["num_gpus"] = 1
    config["eager"] = False
    config["observation_filter"] = "NoFilter"
    #config["batch_mode"] = "complete_episodes"
    config["clip_actions"] = True
    config["env_config"] = {
        "sim_state": sim_state,
        "mode": "sim",
        "timesteps_per_iteration": config["timesteps_per_iteration"]
    }

    ray.init()
    trainer = ddpg.DDPGTrainer(env=SingleAgentEnv, config=config)
    trainer.restore(checkpoint_path)

    visualization_sim = VisualizationSim(sim_state, trainer)
    visualization_sim.run()


if __name__ == "__main__":
    main()
