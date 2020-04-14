from ray.tune.utils import merge_dicts
from simulations.configs.ddpg_config import DDPG_CONFIG

APEX_DDPG_CONFIG = merge_dicts(
    DDPG_CONFIG,
    {
        "optimizer": merge_dicts(
            DDPG_CONFIG["optimizer"], {
                "max_weight_sync_delay": 400,
                "num_replay_buffer_shards": 4,
                "debug": False
            }),
        "n_step": 3,
        "num_gpus": 0,
        "num_workers": 7,
        "buffer_size": 2000000,
        "learning_starts": 50000,
        "train_batch_size": 32,
        "sample_batch_size": 20,
        "target_network_update_freq": 50000,
        "timesteps_per_iteration": 25000,
        "per_worker_exploration": True,
        "worker_side_prioritization": True,
        "min_iter_time_s": 30,
        "evaluation_interval": 0,
        "evaluation_num_episodes": 10
    },
)