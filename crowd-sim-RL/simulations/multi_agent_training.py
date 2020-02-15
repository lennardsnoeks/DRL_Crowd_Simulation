import os
import ray
import ray.rllib.agents.ddpg as ddpg
from ray.tune.logger import pretty_print
from crowd_sim_RL.envs import SingleAgentEnv
from crowd_sim_RL.envs.multi_agent_env import MultiAgentEnvironment
from utils.steerbench_parser import XMLSimulationState
from visualization.visualize_steerbench import VisualizationLive
from threading import Thread
from simulations import ddpg_config


def main():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "../test_XML_files/obstacles2.xml")
    sim_state = XMLSimulationState(filename).simulation_state

    train(sim_state)


def initial_visualization(visualization):
    visualization.run()


def train(sim_state):
    iterations = 20
    visualization = VisualizationLive(sim_state)

    config = ddpg_config.DDPG_CONFIG.copy()
    config["gamma"] = 0.95
    config["num_workers"] = 0
    config["num_gpus"] = 1
    config["eager"] = False
    config["exploration_should_anneal"] = True
    config["exploration_noise_type"] = "ou"
    config["observation_filter"] = "MeanStdFilter"
    config["clip_actions"] = True
    config["env_config"] = {
        "sim_state": sim_state,
        "mode": "multi_train",
        "timesteps_per_iteration": config["timesteps_per_iteration"]
    }

    env_config = config["env_config"]
    single_env = SingleAgentEnv(env_config, 0)
    obs_space = single_env.get_observation_space()
    action_space = single_env.get_action_space()

    config["multiagent"] = {
        "policies": {
            "policy_0": (None, obs_space, action_space, {"gamma": 0.95})
        },
        "policy_mapping_fn": lambda agent_id: "policy_0"
    }

    ray.init()
    trainer = ddpg.DDPGTrainer(env=MultiAgentEnvironment, config=config)

    thread = Thread(target=initial_visualization, args=(visualization,))
    thread.start()

    for i in range(iterations):
        result = trainer.train()
        print(pretty_print(result))

        # Save checkpoint on last iteration
        if i == iterations - 1:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)

    visualization.stop()
    thread.join()


if __name__ == "__main__":
    main()