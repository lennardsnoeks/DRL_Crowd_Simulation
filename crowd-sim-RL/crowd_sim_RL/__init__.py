from gym.envs.registration import register

register(
    id='single-agent',
    entry_point='crowd_sim_RL.crowd_sim_RL:SingleAgentEnv',
)

register(
    id='multi-agent',
    entry_point='crowd_sim_RL.crowd_sim_RL:MultiAgentEnv',
)