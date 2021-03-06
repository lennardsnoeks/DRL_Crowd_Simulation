import os
import ray
import ray.rllib.agents.ddpg as ddpg
import ray.rllib.agents.ppo as ppo
from crowd_sim_RL.envs.multi_agent_env_centralized import MultiAgentCentralized
from utils.steerbench_parser import XMLSimulationState
from visualization.visualize_simulation_multi_centralized import VisualizationSimMultiCentralized
from simulations.configs import ddpg_config, ppo_config


def main():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "../test_XML_files/training/")
    seed = 1
    sim_state = XMLSimulationState(filename, seed).simulation_state

    checkpoint_path = ""

    simulate(sim_state, checkpoint_path)


def simulate(sim_state, checkpoint_path):
    #config = ddpg_config.DDPG_CONFIG.copy()
    config = ppo_config.PPO_CONFIG.copy()

    config["gamma"] = 0.99
    config["num_workers"] = 0
    config["num_gpus"] = 0
    config["eager"] = False
    config["explore"] = False
    config["observation_filter"] = "MeanStdFilter"
    config["clip_actions"] = True
    config["env_config"] = {
        "sim_state": sim_state,
        "mode": "sim",
        "agent_id": 0,
        "timesteps_reset": config["timesteps_per_iteration"]
    }

    ray.init()
    #trainer = ddpg.DDPGTrainer(env=MultiAgentCentralized, config=config)
    trainer = ppo.PPOTrainer(env=MultiAgentCentralized, config=config)
    trainer.restore(checkpoint_path)

    zoom_factor = 10
    visualization_sim = VisualizationSimMultiCentralized(sim_state, trainer, zoom_factor)
    visualization_sim.run()


if __name__ == "__main__":
    main()
