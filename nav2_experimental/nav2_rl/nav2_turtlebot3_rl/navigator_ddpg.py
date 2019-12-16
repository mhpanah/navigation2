# Copyright (c) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from navigation_task_env import NavigationTaskEnv
#from navigation_task_env_checkpoint import NavigationTaskEnv
import pickle
import rclpy
from rclpy.node import Node
from time import sleep

import sys
import numpy as np
import gym
from gym import spaces

from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.processors import WhiteningNormalizerProcessor
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from rl_coach.coach import CoachInterface


class TB3Processor(WhiteningNormalizerProcessor):
    def process_action(self, action):
        return action*0.26


class TB3NavigationEnvironmentDDPG(NavigationTaskEnv):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=-0.26,
                                       high=0.26,
                                       shape=(2,),
                                       dtype=np.float32)

    def get_actions(self):
        return [0.0, 0.0]

    def get_velocity_cmd(self, action):
        self.act = action
        x_vel = float(action[0])
        y_vel = 0.0
        z_vel = float(action[1])
        return x_vel, y_vel, z_vel


class NavigatorDDPG():
    def __init__(self, env):
        self.state = env.reset()
        self.observation_space_size = env.observation_space_size()
        self.build_model(env)

    def build_model(self, env):

        nb_actions = env.action_space_size()
        self.actor = Sequential()
        # Actor Model
        self.actor.add(Flatten(input_shape=(1,) + (self.observation_space_size,)))
        self.actor.add(Dense(400))
        self.actor.add(Activation('relu'))
        self.actor.add(Dense(300))
        self.actor.add(Activation('relu'))
        self.actor.add(Dense(nb_actions))
        self.actor.add(Activation('tanh'))
        print(self.actor.summary())

        # Critic Model
        action_input = Input(shape=(nb_actions,), name='action_input')
        observation_input = Input(shape=(1,) + (self.observation_space_size,),
                                  name='observation_input')
        flattened_observation = Flatten()(observation_input)
        x = Dense(400)(flattened_observation)
        x = Activation('relu')(x)
        x = Concatenate()([x, action_input])
        x = Dense(300)(x)
        x = Activation('relu')(x)
        x = Dense(1)(x)
        x = Activation('linear')(x)
        self.critic = Model(inputs=[action_input, observation_input], outputs=x)
        print(self.critic.summary())

        memory = SequentialMemory(limit=200000, window_length=1) #400000

        #random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.0075, mu=0., sigma=.01)  # 0.03/4   0.04/4
        random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.03, mu=0., sigma=.1, sigma_min = 0.001, n_steps_annealing=5000000) #dt=0.01, 

        self.agent = DDPGAgent(nb_actions=nb_actions,
                               actor=self.actor, critic=self.critic,
                               critic_action_input=action_input,
                               batch_size=16,
                               memory=memory,
                               nb_steps_warmup_critic=25000,
                               nb_steps_warmup_actor=25000,
                               random_process=random_process,
                               gamma=.99,
                               target_model_update=14,
                               train_interval = 1,
                               processor=TB3Processor())

        self.agent.compile([Adam(lr=1e-5, epsilon=1e-8, decay=0.01), Adam(lr=1e-4, epsilon=1e-8, decay = 0.01)], metrics=['mae'])
        #self.agent.load_weights('ddpg_weights.h5f')
        #pickle_in = open("memory.pickle", "rb")
        #self.agent.memory = pickle.load(pickle_in)

    def train_model(self, env, action_size):
        #self.agent.load_weights('ddpg_weights.h5f')
        for it in range(0, 100000, 1):
            self.agent.fit(env, nb_steps=30000, visualize=False, verbose=1, nb_max_episode_steps=500)
            self.agent.nb_steps_warmup_critic = 1
            self.agent.nb_steps_warmup_actor = 1
            self.agent.save_weights('ddpg_weights{}.h5f'.format(it), overwrite=True)
            pickle_out = open("memory.pickle", "wb")
            pickle.dump(self.agent.memory, pickle_out)
            pickle_out.close()
            self.agent.test(env, nb_episodes=10, visualize=True)
            sleep(1)
            self.agent.load_weights('ddpg_weights{}.h5f'.format(it))

    def load_model(self, env, action_size):
        self.agent.load_weights('ddpg_weights.h5f')
        sleep(5)
        # self.agent.test(env, nb_episodes=500, visualize=True)
        observation = env.reset()
        for _ in range(5000):
            action = self.agent.forward(observation) * 0.26
            observation, r, done, info = env.step(action)
            if done:
                observation = env.reset()



def main(args=None):
    rclpy.init(args=args)
    env = TB3NavigationEnvironmentDDPG()
    action_size = env.action_space_size()
    ddpg_agent = NavigatorDDPG(env)

    # Ctrl-C doesn't make rclpy.ok() to return false. Thus, we catch the exception with
    # `finally` to shutdown ros and terminate the background thread cleanly.
    try:
        if len(sys.argv) > 1:
            ddpg_agent.train_model(env, action_size)
        else:
            ddpg_agent.load_model(env, action_size)
    except KeyboardInterrupt:
        pass
    finally:
        env.stop_action()
        rclpy.shutdown()
        env.cleanup()
    return

if __name__ == "__main__":
    main()
