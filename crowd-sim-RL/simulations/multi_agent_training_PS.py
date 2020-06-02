import os
import ray
from ray.tune import register_env, run
from crowd_sim_RL.envs import SingleAgentEnv
from crowd_sim_RL.envs.multi_agent_env import MultiAgentEnvironment
from utils.steerbench_parser import XMLSimulationState
from simulations.configs import ddpg_config, ppo_config

phase2_set = False
phase3_set = False
iterations_count = 0
iterations_max = 100
mean_max = 540
mean_save = 540
count_over_max = 5
count_over = 0


def main():
    filename = ""
    seed = 1
    sim_state = parse_sim_state(filename, seed)

    checkpoint = ""

    train(sim_state, checkpoint)


def parse_sim_state(filename, seed):
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "../test_XML_files/training/" + filename + ".xml")
    sim_state = XMLSimulationState(filename, seed).simulation_state

    return sim_state


def on_train_result(info):
    global iterations_count, count_over, count_over_max, iterations_max, mean_max, phase2_set, phase3_set, test_set
    result = info["result"]
    trainer = info["trainer"]
    mean = result["episode_reward_mean"]

    if mean > mean_save or count_over > count_over_max:
        trainer.save()

    # always checkpoint on last iteration or if mean reward > asked mean reward
    """if iterations_count == iterations_max - 1 or mean > mean_max or count_over > count_over_max:
        trainer.save()
    iterations_count += 1

    # curriculum learning
    if not phase2_set and mean > 157:
        print("#### PHASE 2 ####")

        sim_state = parse_sim_state("crossway_2")

        trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.set_phase(sim_state)))

        phase2_set = True

        trainer.save()

    if not phase3_set and mean > 305:
        print("#### PHASE 3 ####")

        sim_state = parse_sim_state("crossway_more")

        trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.set_phase(sim_state)))

        phase3_set = True

        trainer.save()

    if result["episode_reward_max"] >= 750:
        trainer.save()"""


def on_episode_end(info):
    global count_over
    episode = info["episode"]

    if episode.total_reward > mean_max:
        count_over += 1
    else:
        count_over = 0


def train(sim_state, checkpoint):
    global iterations_max, mean_max
    checkpoint_freq = 0

    #config = ddpg_config.DDPG_CONFIG.copy()
    config = ppo_config.PPO_CONFIG.copy()

    config["gamma"] = 0.99
    config["metrics_smoothing_episodes"] = 20
    config["num_workers"] = 0
    config["num_gpus"] = 0
    config["observation_filter"] = "MeanStdFilter"
    config["clip_actions"] = True
    config["timesteps_per_iteration"] = 1000
    config["env_config"] = {
        "sim_state": sim_state,
        "mode": "multi_train",
        "agent_id": 0,
        "timesteps_reset": config["timesteps_per_iteration"]
    }
    config["callbacks"] = {
        "on_train_result": on_train_result,
        "on_episode_end": on_episode_end
    }

    env_config = config["env_config"]

    single_env = SingleAgentEnv(env_config)
    obs_space = single_env.get_observation_space()
    action_space = single_env.get_action_space()

    gamma = config["gamma"]
    config["multiagent"] = {
        "policies": {
            "policy_0": (None, obs_space, action_space, {"gamma": gamma}),
        },
        "policy_mapping_fn": lambda agent_id: "policy_0"
    }

    register_env("multi_agent_env", lambda _: MultiAgentEnvironment(config["env_config"]))
    config["env"] = "multi_agent_env"

    ray.init()

    stop = {
        "episode_reward_mean": mean_max,
        #"training_iteration": iterations_max
    }

    name = ""
    algo = "PPO"    # Options: DDPG, PPO

    if checkpoint == "":
        run(algo, name=name, checkpoint_freq=checkpoint_freq, stop=stop, config=config)
    else:
        run(algo, name=name, checkpoint_freq=checkpoint_freq, stop=stop, config=config, restore=checkpoint)


if __name__ == "__main__":
    main()
