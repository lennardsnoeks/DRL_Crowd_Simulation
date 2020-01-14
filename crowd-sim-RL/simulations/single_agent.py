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
from rl.random import OrnsteinUhlenbeckProcess, GaussianWhiteNoiseProcess
from threading import Thread


def main():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "../test_XML_files/obstacles.xml")
    sim_state = XMLSimulationState(filename).simulation_state

    train(sim_state)


def initial_visualization(visualization):
    visualization.run()


def train(sim_state):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    visualization = Visualization(sim_state)

    env: SingleAgentEnv = gym.make('singleagent-v0')
    env.load_params(sim_state)
    env.set_visualizer(visualization)

    thread = Thread(target=initial_visualization, args=(visualization,))
    thread.start()

    assert len(env.action_space.shape) == 1
    nb_actions = env.action_space.shape[0]

    # build network
    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    actor.add(Dense(32))
    actor.add(Activation('elu'))
    actor.add(Dense(64))
    actor.add(Activation('elu'))
    actor.add(Dense(32))
    actor.add(Activation('linear'))
    actor.add(Dense(nb_actions))
    actor.add(Activation('linear'))
    print(actor.summary())

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = Dense(32)(flattened_observation)
    x = Activation('elu')(x)
    x = Dense(64)(x)
    x = Activation('elu')(x)
    x = Concatenate()([x, action_input])
    x = Dense(32)(x)
    x = Activation('linear')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    print(critic.summary())

    """actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    actor.add(Dense(32))
    actor.add(Activation('elu'))
    actor.add(Dense(32))
    actor.add(Activation('elu'))
    actor.add(Dense(nb_actions))
    actor.add(Activation('linear'))
    print(actor.summary())

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(32)(x)
    x = Activation('elu')(x)
    x = Dense(32)(x)
    x = Activation('elu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    print(critic.summary())"""

    # configure the learning agent
    memory = SequentialMemory(limit=100000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.1, mu=0., sigma=.2)
    gaussian_random = GaussianWhiteNoiseProcess(size=nb_actions, mu=0, sigma=.2)
    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                      random_process=random_process, gamma=.95, batch_size=32, target_model_update=1e-3)
    agent.compile([Adam(lr=.0001, clipnorm=1.), Adam(lr=.001, clipnorm=1.)], metrics=['mae'])

    agent.fit(env, nb_steps=100000, visualize=True, verbose=1, nb_max_episode_steps=1000)

    """memory = SequentialMemory(limit=100000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.5, mu=0., sigma=.5)
    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                      random_process=random_process, gamma=.99, target_model_update=1e-3)
    agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])"""

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    #agent.fit(env, nb_steps=50000, visualize=True, verbose=1, nb_max_episode_steps=200)


if __name__ == "__main__":
    main()
