import os
import ray
import ray.rllib.agents.ddpg as ddpg
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.a3c as a3c
import ray.rllib.agents.sac as sac
from crowd_sim_RL.envs import SingleAgentEnv, SingleAgentEnv2, SingleAgentEnv3
from crowd_sim_RL.envs.multi_agent_env import MultiAgentEnvironment
from utils.steerbench_parser import XMLSimulationState
from simulations.configs import ddpg_config, ppo_config, a2c_config, td3_config, sac_config, apex_config
from visualization.visualize_simulation_multi import VisualizationSimMulti


def main():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "../test_XML_files/training/4-hallway/group.xml")
    seed = 1
    sim_state = XMLSimulationState(filename, seed).simulation_state

    #checkpoint_path = "/home/lennard/ray_results/DDPG/DDPG_multi_agent_env_6960a2d6_0_2020-03-15_14-53-55_w30woiy/checkpoint_308_2/checkpoint-308"
    checkpoint_path = "/home/lennard/ray_results/test_ppo/PPO_multi_agent_env_661e2280_0_2020-04-14_19-14-28hpv1b6ds/checkpoint_89/checkpoint-89"

    simulate(sim_state, checkpoint_path)


def simulate(sim_state, checkpoint_path):
    #config = td3_config.TD3_CONFIG.copy()
    #config = a2c_config.A2C_CONFIG.copy()
    config = ppo_config.PPO_CONFIG.copy()
    #config = ddpg_config.DDPG_CONFIG.copy()
    #config = sac_config.SAC_CONFIG.copy()
    #config = apex_config.APEX_DDPG_CONFIG.copy()

    config["gamma"] = 0.99
    #config["gamma"] = 0.95
    gamma = config["gamma"]
    config["num_workers"] = 0
    config["num_gpus"] = 0
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
    single_env = SingleAgentEnv3(env_config)
    obs_space = single_env.get_observation_space()
    action_space = single_env.get_action_space()

    config["multiagent"] = {
        "policies": {
            "policy_0": (None, obs_space, action_space, {"gamma": gamma})
        },
        "policy_mapping_fn": lambda agent_id: "policy_0"
    }

    ray.init()

    #trainer = ddpg.DDPGTrainer(env=MultiAgentEnvironment, config=config)
    trainer = ppo.PPOTrainer(env=MultiAgentEnvironment, config=config)
    #trainer = ddpg.TD3Trainer(env=MultiAgentEnvironment, config=config)
    #trainer = a3c.A2CTrainer(env=MultiAgentEnvironment, config=config)
    #trainer = sac.SACTrainer(env=MultiAgentEnvironment, config=config)
    #trainer = ddpg.ApexDDPGTrainer(env=MultiAgentEnvironment, config=config)

    trainer.restore(checkpoint_path)

    visualization_sim = VisualizationSimMulti(sim_state, trainer)
    visualization_sim.run()


if __name__ == "__main__":
    main()
