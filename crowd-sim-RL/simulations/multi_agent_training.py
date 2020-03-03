import os
import ray
import ray.rllib.agents.ddpg as ddpg
from ray.tune import register_env, run
from ray.tune.logger import pretty_print
from crowd_sim_RL.envs import SingleAgentEnv
from crowd_sim_RL.envs.multi_agent_env import MultiAgentEnvironment
from utils.steerbench_parser import XMLSimulationState
from visualization.visualize_steerbench import VisualizationLive
from threading import Thread
from simulations import ddpg_config


phase_set = False


def main():
    filename = "obstacles"
    sim_state = parse_sim_state(filename)

    train(sim_state)


def parse_sim_state(filename):
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "../test_XML_files/" + filename + ".xml")
    seed = 22222
    sim_state = XMLSimulationState(filename, seed).simulation_state

    return sim_state


def initial_visualization(visualization):
    visualization.run()


def on_train_result(info):
    global phase_set
    result = info["result"]
    if not phase_set and result["episode_reward_mean"] > 177:
        print("#### PHASE 2 ####")
        phase = 1

        sim_state = parse_sim_state("obstacles2")

        trainer = info["trainer"]
        trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.set_phase(phase, sim_state)))
        phase_set = True


def train(sim_state):
    total_iterations = 50
    checkpoint_freq = 10

    config = ddpg_config.DDPG_CONFIG.copy()
    config["gamma"] = 0.95
    config["num_workers"] = 5
    config["num_gpus"] = 0
    config["eager"] = False
    config["exploration_should_anneal"] = True
    config["exploration_noise_type"] = "ou"
    config["observation_filter"] = "MeanStdFilter"
    config["clip_actions"] = True
    config["env_config"] = {
        "sim_state": sim_state,
        "mode": "multi_train",
        "agent_id": 0,
        "timesteps_per_iteration": config["timesteps_per_iteration"]
    }
    config["callbacks"] = {
        "on_train_result": on_train_result
    }

    env_config = config["env_config"]
    single_env = SingleAgentEnv(env_config)
    obs_space = single_env.get_observation_space()
    action_space = single_env.get_action_space()

    config["multiagent"] = {
        "policies": {
            "policy_0": (None, obs_space, action_space, {"gamma": 0.95})
        },
        "policy_mapping_fn": lambda agent_id: "policy_0"
    }

    register_env("multi_agent_env", lambda _: MultiAgentEnvironment(config["env_config"]))
    config["env"] = "multi_agent_env"

    ray.init()

    stop = {
        "training_iteration": total_iterations
    }

    run("DDPG", checkpoint_freq=checkpoint_freq, stop=stop, config=config)


if __name__ == "__main__":
    main()
