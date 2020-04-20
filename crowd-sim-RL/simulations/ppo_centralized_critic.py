import os
import numpy as np
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.impala.vtrace_policy import BEHAVIOUR_LOGITS
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy, KLCoeffMixin, PPOLoss
from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import LearningRateSchedule, EntropyCoeffSchedule, ACTION_LOGP
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
from ray.rllib.utils.explained_variance import explained_variance
from ray.rllib.utils.tf_ops import make_tf_callable
from ray.rllib.utils import try_import_tf
from ray.tune import register_env, run
from crowd_sim_RL.envs import SingleAgentEnv3
from crowd_sim_RL.envs.multi_agent_env import MultiAgentEnvironment
from simulations.configs import ppo_config, ppo_config2
from utils.steerbench_parser import XMLSimulationState

tf = try_import_tf()

OPPONENT_OBS = "opponent_obs"
OPPONENT_ACTION = "opponent_action"

num_agents = 0


class CentralizedCriticModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CentralizedCriticModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        # Base of the model
        self.model = FullyConnectedNetwork(obs_space, action_space, num_outputs, model_config, name)
        self.register_variables(self.model.variables())

        # Obs space = Tuple(Box(2,), Box(3,21), Box(3,21)) => total length is 2 + 63 + 63 = 128 for one agent
        # Act space = Box(2,) = 2 for one agent
        acts_opp_num = (num_agents - 1) * 2
        obs_opp_num = (num_agents - 1) * 128

        obs = tf.keras.layers.Input(shape=obs_space.shape, name="obs")
        opp_obs = tf.keras.layers.Input(shape=(obs_opp_num, ), name="opp_obs")
        opp_act = tf.keras.layers.Input(shape=(acts_opp_num, ), name="opp_act")

        # Central VF maps (obs, opp_obs, opp_act) -> vf_pred
        concat_obs = tf.keras.layers.Concatenate(axis=1)(
            [obs, opp_obs, opp_act])
        central_vf_dense = tf.keras.layers.Dense(
            256, activation=tf.nn.tanh, name="c_vf_dense")(concat_obs)
        central_vf_out = tf.keras.layers.Dense(
            1, activation=None, name="c_vf_out")(central_vf_dense)
        self.central_vf = tf.keras.Model(
            inputs=[obs, opp_obs, opp_act], outputs=central_vf_out)
        self.register_variables(self.central_vf.variables)

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def central_value_function(self, obs, opponent_obs, opponent_actions):
        return tf.reshape(
            self.central_vf(
                [obs, opponent_obs, opponent_actions]), [-1])

    def value_function(self):
        return self.model.value_function()


class CentralizedValueMixin:
    def __init__(self):
        self.compute_central_vf = make_tf_callable(self.get_session())(
            self.model.central_value_function)


# Grabs the opponent obs/act and includes it in the experience train_batch,
# and computes GAE using the central vf predictions.
def centralized_critic_postprocessing(policy,
                                      sample_batch,
                                      other_agent_batches=None,
                                      episode=None):
    if policy.loss_initialized():
        assert other_agent_batches is not None

        [(_, opponents_batch)] = list(other_agent_batches.values())

        # also record the opponent obs and actions in the trajectory
        sample_batch[OPPONENT_OBS] = opponents_batch[SampleBatch.CUR_OBS]
        sample_batch[OPPONENT_ACTION] = opponents_batch[SampleBatch.ACTIONS]

        # overwrite default VF prediction with the central VF
        sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_vf(
            sample_batch[SampleBatch.CUR_OBS], sample_batch[OPPONENT_OBS],
            sample_batch[OPPONENT_ACTION])
    else:
        # policy hasn't initialized yet, use zeros
        sample_batch[OPPONENT_OBS] = np.zeros_like(
            sample_batch[SampleBatch.CUR_OBS])
        sample_batch[OPPONENT_ACTION] = np.zeros_like(
            sample_batch[SampleBatch.ACTIONS])
        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(
            sample_batch[SampleBatch.REWARDS], dtype=np.float32)

    completed = sample_batch["dones"][-1]
    if completed:
        last_r = 0.0
    else:
        last_r = sample_batch[SampleBatch.VF_PREDS][-1]

    train_batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"])
    return train_batch


# Copied from PPO but optimizing the central value function
def loss_with_central_critic(policy, model, dist_class, train_batch):
    CentralizedValueMixin.__init__(policy)

    logits, state = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)
    policy.central_value_out = policy.model.central_value_function(
        train_batch[SampleBatch.CUR_OBS], train_batch[OPPONENT_OBS],
        train_batch[OPPONENT_ACTION])

    policy.loss_obj = PPOLoss(
        dist_class,
        model,
        train_batch[Postprocessing.VALUE_TARGETS],
        train_batch[Postprocessing.ADVANTAGES],
        train_batch[SampleBatch.ACTIONS],
        train_batch[BEHAVIOUR_LOGITS],
        train_batch[ACTION_LOGP],
        train_batch[SampleBatch.VF_PREDS],
        action_dist,
        policy.central_value_out,
        policy.kl_coeff,
        tf.ones_like(train_batch[Postprocessing.ADVANTAGES], dtype=tf.bool),
        entropy_coeff=policy.entropy_coeff,
        clip_param=policy.config["clip_param"],
        vf_clip_param=policy.config["vf_clip_param"],
        vf_loss_coeff=policy.config["vf_loss_coeff"],
        use_gae=policy.config["use_gae"]
    )

    return policy.loss_obj.loss


def setup_mixins(policy, obs_space, action_space, config):
    # copied from PPO
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])


def central_vf_stats(policy, train_batch, grads):
    # Report the explained variance of the central value function.
    return {
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS],
            policy.central_value_out),
    }


CCPPO = PPOTFPolicy.with_updates(
    name="CCPPO",
    postprocess_fn=centralized_critic_postprocessing,
    loss_fn=loss_with_central_critic,
    before_loss_init=setup_mixins,
    grad_stats_fn=central_vf_stats,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        CentralizedValueMixin
    ])

CCTrainer = PPOTrainer.with_updates(name="CCPPOTrainer", get_policy_class=lambda config: CCPPO)


##### Below is code to run, above is code that implements centralized critic #####
def main():
    filename = "4-hallway/2"
    sim_state = parse_sim_state(filename)

    checkpoint = ""

    train(sim_state, checkpoint)


def parse_sim_state(filename):
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "../test_XML_files/training/" + filename + ".xml")
    seed = 22222
    sim_state = XMLSimulationState(filename, seed).simulation_state

    return sim_state


def make_multi_agent_config(sim_state, config):
    multi_agent_config = {}
    policy_dict = {}

    env_config = config["env_config"]
    env_config["agent_id"] = 0

    gamma = config["gamma"]

    single_env = SingleAgentEnv3(env_config)
    obs_space = single_env.get_observation_space()
    action_space = single_env.get_action_space()

    for agent in sim_state.agents:
        policy_id = "policy_" + str(agent.id)
        policy_dict[policy_id] = (None, obs_space, action_space, {"gamma": gamma})

    multi_agent_config["policies"] = policy_dict
    multi_agent_config["policy_mapping_fn"] = lambda agent_id: "policy_" + str(agent_id)

    return multi_agent_config


def train(sim_state, checkpoint):
    global num_agents
    num_agents = len(sim_state.agents)

    checkpoint_freq = 5

    config = ppo_config.PPO_CONFIG.copy()
    config["gamma"] = 0.99
    config["observation_filter"] = "MeanStdFilter"
    config["clip_actions"] = True
    config["env_config"] = {
        "sim_state": sim_state,
        "mode": "multi_train_vis",
        "timesteps_reset": config["timesteps_per_iteration"]
    }
    multi_agent_config = make_multi_agent_config(sim_state, config)
    config["multiagent"] = multi_agent_config
    register_env("multi_agent_env", lambda _: MultiAgentEnvironment(config["env_config"]))
    ModelCatalog.register_custom_model("cc_model", CentralizedCriticModel)
    config["env"] = "multi_agent_env"
    config["batch_mode"] = "truncate_episodes"
    config["eager"] = False
    config["num_workers"] = 0
    config["model"] = {
        "custom_model": "cc_model"
    }

    """config = ppo_config.PPO_CONFIG.copy()
    #config = ppo_config2.PPO_CONFIG.copy()
    config["gamma"] = 0.99
    config["num_workers"] = 0
    config["observation_filter"] = "MeanStdFilter"
    config["clip_actions"] = True
    config["env_config"] = {
        "sim_state": sim_state,
        "mode": "multi_train_vis",
        "timesteps_reset": config["timesteps_per_iteration"]
    }

    register_env("multi_agent_env", lambda _: MultiAgentEnvironment(config["env_config"]))
    config["env"] = "multi_agent_env"

    ModelCatalog.register_custom_model("cc_model", CentralizedCriticModel)

    config["model"] = {"custom_model": "cc_model"}
    config["batch_mode"] = "truncate_episodes" """

    stop = {}

    name = "central_critic"
    if checkpoint == "":
        run(CCTrainer, name=name, checkpoint_freq=checkpoint_freq, stop=stop, config=config)
    else:
        run(CCTrainer, name=name, checkpoint_freq=checkpoint_freq, stop=stop, config=config, restore=checkpoint)


if __name__ == "__main__":
    main()
