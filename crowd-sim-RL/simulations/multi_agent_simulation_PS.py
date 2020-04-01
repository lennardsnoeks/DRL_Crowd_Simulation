import os
import ray
import ray.rllib.agents.ddpg as ddpg
import ray.rllib.agents.ppo as ppo
from crowd_sim_RL.envs import SingleAgentEnv, SingleAgentEnv2, SingleAgentEnv3
from crowd_sim_RL.envs.multi_agent_env import MultiAgentEnvironment
from utils.steerbench_parser import XMLSimulationState
from simulations.configs import ddpg_config, ppo_config
from visualization.visualize_simulation_multi import VisualizationSimMulti


def main():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "../test_XML_files/hallway_slimmer/hallway_4.xml")
    seed = 1
    sim_state = XMLSimulationState(filename, seed).simulation_state

    #checkpoint_path = "/home/lennard/ray_results/DDPG/DDPG_multi_agent_env_6960a2d6_0_2020-03-15_14-53-55_w30woiy/checkpoint_308_2/checkpoint-308"
    checkpoint_path = "/home/lennard/ray_results/hyper/DDPG_multi_agent_env_8b6d8c1c_0_2020-04-01_14-51-469w2zmrvb/checkpoint_127/checkpoint-127"

    simulate(sim_state, checkpoint_path)


def simulate(sim_state, checkpoint_path):
    #config = ppo_config.PPO_CONFIG.copy()
    config = ddpg_config.DDPG_CONFIG.copy()
    config["gamma"] = 0.95
    #config["gamma"] = 0.99
    gamma = config["gamma"]
    config["num_workers"] = 0
    config["num_gpus"] = 0
    config["eager"] = False
    config["explore"] = False
    config["observation_filter"] = "MeanStdFilter"
    config["clip_actions"] = True
    config["env_config"] = {
        "sim_state": sim_state,
        "mode": "multi_sim",
        "agent_id": 0,
        "timesteps_per_iteration": config["timesteps_per_iteration"]
    }

    env_config = config["env_config"]
    single_env = SingleAgentEnv2(env_config)
    obs_space = single_env.get_observation_space()
    action_space = single_env.get_action_space()

    config["multiagent"] = {
        "policies": {
            "policy_0": (None, obs_space, action_space, {"gamma": gamma})
        },
        "policy_mapping_fn": lambda agent_id: "policy_0"
    }

    ray.init()
    trainer = ddpg.DDPGTrainer(env=MultiAgentEnvironment, config=config)
    #trainer = ppo.PPOTrainer(env=MultiAgentEnvironment, config=config)
    trainer.restore(checkpoint_path)

    visualization_sim = VisualizationSimMulti(sim_state, trainer)
    visualization_sim.run()


if __name__ == "__main__":
    main()