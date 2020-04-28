import os
import ray
import ray.rllib.agents.ddpg as ddpg
from ray.rllib.agents.ddpg.ddpg_policy import DDPGTFPolicy
from crowd_sim_RL.envs import SingleAgentEnv
from crowd_sim_RL.envs.multi_agent_env import MultiAgentEnvironment
from simulations.ppo_centralized_critic import CCTrainer, CCPPO
from utils.steerbench_parser import XMLSimulationState
from simulations.configs import ddpg_config, ppo_config, td3_config
from visualization.visualize_simulation_multi_concurrent import VisualizationSimMultiConcurrent


def main():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "../test_XML_files/4-hallway/4.xml")
    seed = 22222
    sim_state = XMLSimulationState(filename, seed).simulation_state

    checkpoint_path = "/home/lennard/ray_results/central_critic/CCPPOTrainer_multi_agent_env_17f4268c_0_2020-04-27_23-50-21_kx46i91/checkpoint_73"

    simulate(sim_state, checkpoint_path)


def make_multi_agent_config(sim_state, config, centralized):
    multi_agent_config = {}
    policy_dict = {}

    env_config = config["env_config"]
    env_config["agent_id"] = 0

    gamma = config["gamma"]

    single_env = SingleAgentEnv(env_config)
    obs_space = single_env.get_observation_space()
    action_space = single_env.get_action_space()

    for agent in sim_state.agents:
        policy_id = "policy_" + str(agent.id)
        if centralized:
            policy_dict[policy_id] = (CCPPO, obs_space, action_space, {"gamma": gamma})
        else:
            policy_dict[policy_id] = (None, obs_space, action_space, {"gamma": gamma})

    multi_agent_config["policies"] = policy_dict
    multi_agent_config["policy_mapping_fn"] = lambda agent_id: "policy_" + str(agent_id)

    return multi_agent_config


def simulate(sim_state, checkpoint_path):
    #config = ddpg_config.DDPG_CONFIG.copy()
    config = ppo_config.PPO_CONFIG.copy()
    #config = td3_config.TD3_CONFIG.copy()

    config["gamma"] = 0.99
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

    centralized = True

    multi_agent_config = make_multi_agent_config(sim_state, config, centralized)
    config["multiagent"] = multi_agent_config

    ray.init()
    #trainer = ddpg.PPOTrainer(env=MultiAgentEnvironment, config=config)
    #trainer = ddpg.DDPGTrainer(env=MultiAgentEnvironment, config=config)
    #trainer = ddpg.TD3Trainer(env=MultiAgentEnvironment, config=config)
    trainer = CCTrainer(env=MultiAgentEnvironment, config=config)

    trainer.restore(checkpoint_path)

    visualization_sim = VisualizationSimMultiConcurrent(sim_state, trainer)
    visualization_sim.run()


if __name__ == "__main__":
    main()
