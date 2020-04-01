import os
import ray
import ray.rllib.agents.ddpg as ddpg
from ray.rllib.agents.ddpg.ddpg_policy import DDPGTFPolicy
from crowd_sim_RL.envs import SingleAgentEnv
from crowd_sim_RL.envs.multi_agent_env import MultiAgentEnvironment
from utils.steerbench_parser import XMLSimulationState
from simulations.configs import ddpg_config
from visualization.visualize_simulation_multi import VisualizationSimMulti


def main():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "../test_XML_files/hallway_test/hallway_4.xml")
    seed = 22222
    sim_state = XMLSimulationState(filename, seed).simulation_state

    checkpoint_path = "/home/lennard/ray_results/DDPG/DDPG_multi_agent_env_6960a2d6_0_2020-03-15_14-53-55_w30woiy/checkpoint_308_2/checkpoint-308"

    simulate(sim_state, checkpoint_path)


def make_multi_agent_config(sim_state, config):
    multi_agent_config = {}
    policy_dict = {}

    env_config = config["env_config"]
    env_config["agent_id"] = 0

    single_env = SingleAgentEnv(env_config)
    obs_space = single_env.get_observation_space()
    action_space = single_env.get_action_space()

    for agent in sim_state.agents:
        policy_id = "policy_" + str(agent.id)
        policy_dict[policy_id] = (DDPGTFPolicy, obs_space, action_space, {"gamma": 0.95})

    multi_agent_config["policies"] = policy_dict
    multi_agent_config["policy_mapping_fn"] = lambda agent_id: "policy_" + str(agent_id)

    return multi_agent_config


def simulate(sim_state, checkpoint_path):
    config = ddpg_config.DDPG_CONFIG.copy()
    config["gamma"] = 0.95
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

    multi_agent_config = make_multi_agent_config(sim_state, config)
    config["multiagent"] = multi_agent_config

    ray.init()
    trainer = ddpg.DDPGTrainer(env=MultiAgentEnvironment, config=config)
    trainer.restore(checkpoint_path)

    visualization_sim = VisualizationSimMulti(sim_state, trainer)
    visualization_sim.run()


if __name__ == "__main__":
    main()