import os
import ray
from ray.rllib.agents.ddpg.ddpg_policy import DDPGTFPolicy
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.agents.ddpg import ddpg
from ray.tune import register_env, run
from crowd_sim_RL.envs import SingleAgentEnv, SingleAgentEnv2, SingleAgentEnv3
from crowd_sim_RL.envs.multi_agent_env import MultiAgentEnvironment
from utils.steerbench_parser import XMLSimulationState
from simulations.configs import ddpg_config, ddpg_config2, ppo_config

phase2_set = False
phase3_set = False
test_set = False
iterations_count = 0
iterations_max = 100
mean_max = 500
count_over_max = 10
count_over = 0


def main():
    filename = "2/3-confusion"
    seed = 1
    sim_state = parse_sim_state(filename, seed)

    checkpoint = ""

    train(sim_state, checkpoint)


def parse_sim_state(filename, seed):
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "../test_XML_files/training/test_case_" + filename + ".xml")
    sim_state = XMLSimulationState(filename, seed).simulation_state

    return sim_state


def on_train_result(info):
    global iterations_count, count_over, count_over_max, iterations_max, mean_max, phase2_set, phase3_set, test_set
    result = info["result"]
    trainer = info["trainer"]
    mean = result["episode_reward_mean"]

    # always checkpoint on last iteration or if mean reward > asked mean reward
    if iterations_count == iterations_max - 1 or mean > mean_max or count_over > count_over_max:
        trainer.save()
    iterations_count += 1

    # curriculum learning
    """if iterations_count % 20:
        sim_state = parse_sim_state("2-obstacles", iterations_count)
        trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.set_phase(sim_state)))"""

    """if not phase2_set and mean > 157:
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
        trainer.save()

    if not test_set and mean >= 286:
        test_set = True
        trainer.save()"""


def on_episode_end(info):
    global count_over
    episode = info["episode"]

    if episode.total_reward > mean_max:
        count_over += 1


def train(sim_state, checkpoint):
    global iterations_max, mean_max
    checkpoint_freq = 5

    # DDPG
    config = ddpg_config.DDPG_CONFIG.copy()
    config["exploration_should_anneal"] = False
    config["exploration_noise_type"] = "ou"

    # PPO
    #config = ppo_config.PPO_CONFIG.copy()
    """config["gamma"] = 0.99
    config["num_sgd_iter"] = 5
    config["sgd_minibatch_size"] = 32
    config["train_batch_size"] = 2048
    config["lr"] = 0.0003
    config["clip_param"] = 0.2
    config["kl_coeff"] = 1
    config["kl_target"] = 0.01
    config["lambda"] = 0.95
    config["entropy_coeff"] = 0.01"""

    config["num_workers"] = 0
    config["num_gpus"] = 0
    config["observation_filter"] = "MeanStdFilter"
    config["clip_actions"] = True
    config["timesteps_per_iteration"] = 1000
    config["env_config"] = {
        "sim_state": sim_state,
        "mode": "multi_train_vis",
        "agent_id": 0,
        "timesteps_reset": config["timesteps_per_iteration"]
    }
    config["callbacks"] = {
        "on_train_result": on_train_result,
        "on_episode_end": on_episode_end
    }

    env_config = config["env_config"]

    single_env = SingleAgentEnv3(env_config)
    obs_space = single_env.get_observation_space()
    action_space = single_env.get_action_space()

    config["multiagent"] = {
        "policies": {
            "policy_0": (DDPGTFPolicy, obs_space, action_space, {"gamma": 0.95})
            #"policy_0": (PPOTFPolicy, obs_space, action_space, {"gamma": 0.99})
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

    name = "training_case_1"
    if checkpoint == "":
        run("DDPG", name=name, checkpoint_freq=checkpoint_freq, stop=stop, config=config)
    else:
        run("DDPG", name=name, checkpoint_freq=checkpoint_freq, stop=stop, config=config, restore=checkpoint)

    """if checkpoint == "":
        run("PPO", name=name, checkpoint_freq=checkpoint_freq, stop=stop, config=config)
    else:
        run("PPO", name=name, checkpoint_freq=checkpoint_freq, stop=stop, config=config, restore=checkpoint)"""


if __name__ == "__main__":
    main()
