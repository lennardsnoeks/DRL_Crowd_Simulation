import os
import gym
import crowd_sim_RL
import numpy as np

from crowd_sim_RL.envs import SingleAgentEnv
from utils.steerbench_parser import XMLSimulationState
from visualization.visualize_steerbench import Visualization
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from threading import Thread


def main():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "../test_XML_files/office_single_agent.xml")
    sim_state = XMLSimulationState(filename).simulation_state

    #thread = Thread(target=initial_visualization, args=(sim_state, ))
    #thread.start()

    #train(sim_state)

    env: SingleAgentEnv = gym.make('singleagent-v0')
    env.load_params(sim_state)
    env.step()


def initial_visualization(sim_state):
    visualization = Visualization(sim_state)
    visualization.run()


def train(sim_state):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    env: SingleAgentEnv = gym.make('singleagent-v0')
    env.load_params(sim_state)

    assert len(env.action_space.shape) == 1
    nb_actions = env.action_space.shape[0]

    # build network
    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    actor.add(Dense(32))
    actor.add(Activation('relu'))
    actor.add(Dense(64))
    actor.add(Activation('relu'))
    actor.add(Dense(32))
    actor.add(Activation('relu'))
    actor.add(Dense(nb_actions))
    actor.add(Activation('linear'))
    print(actor.summary())

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = Dense(32)(flattened_observation)
    x = Activation('relu')(x)
    x = Dense(64)(x)
    x = Activation('relu')(x)
    x = Concatenate()([x, action_input])
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    print(critic.summary())

    # configure the learning agent
    memory = SequentialMemory(limit=100000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.2)
    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                      random_process=None, gamma=.95, target_model_update=1e-3)
    agent.compile([Adam(lr=.001, clipnorm=1.), Adam(lr=.0001, clipnorm=1.)], metrics=['mae'])

    agent.fit(env, nb_steps=10000, visualize=False, verbose=1)


if __name__ == "__main__":
    main()
