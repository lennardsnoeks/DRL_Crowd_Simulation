import os
import ray
from ray.rllib.agents.ddpg.ddpg_policy import DDPGTFPolicy
from ray.tune import register_env, run
from crowd_sim_RL.envs import SingleAgentEnv
from crowd_sim_RL.envs.multi_agent_env import MultiAgentEnvironment
from utils.steerbench_parser import XMLSimulationState
from simulations.configs import ddpg_config

phase2_set = False
phase3_set = False
test_set = False
iterations_count = 0
iterations_max = 100
mean_max = 300


def main():
    filename = "crossway"
    sim_state = parse_sim_state(filename)

    checkpoint = ""

    train(sim_state, checkpoint)


def parse_sim_state(filename):
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "../test_XML_files/crossway/" + filename + ".xml")
    seed = 22222
    sim_state = XMLSimulationState(filename, seed).simulation_state

    return sim_state


def on_train_result(info):
    global iterations_count, iterations_max, mean_max, phase2_set, phase3_set, test_set
    result = info["result"]
    trainer = info["trainer"]
    mean = result["episode_reward_mean"]

    # always checkpoint on last iteration or if mean reward > asked mean reward
    if iterations_count == iterations_max - 1 or mean > mean_max:
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

    """if result["episode_reward_max"] >= 750:
        trainer.save()"""

    if not test_set and mean >= 286:
        test_set = True
        trainer.save()


def on_episode_end(info):
    pass


def train(sim_state, checkpoint):
    global iterations_max, mean_max
    checkpoint_freq = 10

    config = ddpg_config.DDPG_CONFIG.copy()
    config["gamma"] = 0.95
    config["num_workers"] = 0
    config["num_gpus"] = 0
    config["eager"] = False
    config["exploration_should_anneal"] = False
    config["schedule_max_timesteps"] = 100000
    config["exploration_noise_type"] = "ou"
    config["observation_filter"] = "MeanStdFilter"
    config["clip_actions"] = True
    config["env_config"] = {
        "sim_state": sim_state,
        "mode": "multi_train_vis",
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

    config["multiagent"] = {
        "policies": {
            "policy_0": (DDPGTFPolicy, obs_space, action_space, {"gamma": 0.95})
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

    name = "hallway_2"
    if checkpoint == "":
        run("DDPG", name=name, checkpoint_freq=checkpoint_freq, stop=stop, config=config)
    else:
        run("DDPG", name=name, checkpoint_freq=checkpoint_freq, stop=stop, config=config, restore=checkpoint)


if __name__ == "__main__":
    main()
