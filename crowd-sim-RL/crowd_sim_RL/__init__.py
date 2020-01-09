from gym.envs.registration import register

register(
    id='singleagent-v0',
    entry_point='crowd_sim_RL.envs:SingleAgentEnv'
)

register(
    id='singleagentnorm-v0',
    entry_point='crowd_sim_RL.envs:SingleAgentEnvNorm'
)


"""register(
    id='multi-agent',
    entry_point='crowd_sim_RL.envs:MultiAgentEnv'
)"""