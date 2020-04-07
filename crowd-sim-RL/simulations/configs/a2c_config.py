from ray.tune.utils import merge_dicts
from simulations.configs.a3c_config import A3C_CONFIG

A2C_CONFIG = merge_dicts(
    A3C_CONFIG,
    {
        "sample_batch_size": 10,
        "min_iter_time_s": 10,
        "sample_async": False,

        # A2C supports microbatching, in which we accumulate gradients over
        # batch of this size until the train batch size is reached. This allows
        # training with batch sizes much larger than can fit in GPU memory.
        # To enable, set this to a value less than the train batch size.
        "microbatch_size": None,
    },
)