from gym.envs.registration import register

register(
    id='single-agent',
    entry_point='crowd_sim_RL.crowd_sim_RL:SingleAgentEnv',
)

register(
    id='single-agent-norm',
    entry_point='crowd_sim_RL.crowd_sim_RL:SingleAgentEnvNorm',
)


"""register(
    id='multi-agent',
    entry_point='crowd_sim_RL.crowd_sim_RL:MultiAgentEnv',
)"""