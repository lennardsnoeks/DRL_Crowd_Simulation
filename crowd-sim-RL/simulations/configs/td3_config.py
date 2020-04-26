from ray.tune.utils import merge_dicts
#from simulations.configs.ddpg_config import DDPG_CONFIG
import ray.rllib.agents.ddpg as ddpg

DDPG_CONFIG = ddpg.DEFAULT_CONFIG.copy()

TD3_CONFIG = merge_dicts(
    DDPG_CONFIG,
    {
        # largest changes: twin Q functions, delayed policy updates, and target
        # smoothing
        "twin_q": True,
        "policy_delay": 2,
        "smooth_target_policy": True,
        "target_noise": 0.2,
        "target_noise_clip": 0.5,

        # other changes & things we want to keep fixed: IID Gaussian
        # exploration noise, larger actor learning rate, no l2 regularisation,
        # no Huber loss, etc.
        "exploration_should_anneal": False,
        "exploration_noise_type": "gaussian",
        "exploration_gaussian_sigma": 0.1,
        "learning_starts": 1500,
        "pure_exploration_steps": 1500,
        "actor_hiddens": [64, 64],
        "critic_hiddens": [64, 64],
        "n_step": 1,
        "gamma": 0.95,
        "actor_lr": 1e-3,
        "critic_lr": 1e-3,
        "l2_reg": 0.0,
        "tau": 5e-3,
        "train_batch_size": 16,
        "sample_batch_size": 1,
        "use_huber": False,
        "target_network_update_freq": 0,
        "num_workers": 0,
        "num_gpus_per_worker": 0,
        "per_worker_exploration": False,
        "worker_side_prioritization": False,
        "buffer_size": 100000,
        "prioritized_replay": False,
        "clip_rewards": False,
        "use_state_preprocessor": True
    },
)
