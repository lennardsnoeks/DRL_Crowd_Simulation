import os
import ray
import ray.rllib.agents.ddpg as ddpg
import ray.rllib.agents.ppo as ppo
from ray.rllib.agents.ddpg.ddpg_policy import DDPGTFPolicy
from ray.rllib.models import ModelCatalog

from crowd_sim_RL.envs import SingleAgentEnv
from crowd_sim_RL.envs.multi_agent_env import MultiAgentEnvironment
from simulations.ppo_centralized_critic import CCTrainer, CCPPO, CentralizedCriticModel
from utils.steerbench_parser import XMLSimulationState
from simulations.configs import ddpg_config, ppo_config
from visualization.visualize_simulation_multi_concurrent import VisualizationSimMultiConcurrent


def main():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "../test_XML_files/training/5-crossway_2_groups/group.xml")
    seed = 23
    sim_state = XMLSimulationState(filename, seed).simulation_state

    checkpoint_path = "/home/lennard/ray_results/crossway/centralq_ppo_1/checkpoint_103/checkpoint-103"
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

    if centralized:
        ModelCatalog.register_custom_model("cc_model", CentralizedCriticModel)
        config["model"] = {
            "custom_model": "cc_model"
        }

    ray.init()
    #trainer = ppo.PPOTrainer(env=MultiAgentEnvironment, config=config)
    #trainer = ddpg.DDPGTrainer(env=MultiAgentEnvironment, config=config)
    trainer = CCTrainer(env=MultiAgentEnvironment, config=config)

    trainer.restore(checkpoint_path)

    visualization_sim = VisualizationSimMultiConcurrent(sim_state, trainer)
    visualization_sim.run()


if __name__ == "__main__":
    main()
