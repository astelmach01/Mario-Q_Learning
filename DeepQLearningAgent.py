import gym
import gym_super_mario_bros
import random
from collections import deque


from gym.spaces import Box
from gym.wrappers import *
import numpy as np
import os
from os.path import exists
import matplotlib.pyplot as plt
from nes_py.wrappers import JoypadSpace
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow import keras

print(tf.config.list_physical_devices('GPU'))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(
            low=0, high=255, shape=self.observation_space.shape[:2], dtype=np.uint8)

    def observation(self, observation):
        return tf.image.rgb_to_grayscale(observation)


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = (shape, shape)
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        observation = np.array(observation)
        observation = observation / 255
        im = tf.image.resize(observation, self.shape)
        return im


class TransposeObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        im = np.transpose(observation, [3, 1, 2, 0])
        im = np.squeeze(im)
        return im


env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, [["right"], ["right", "A"]])
env = TransposeObservation(FrameStack(ResizeObservation(GrayScaleObservation(
    SkipFrame(env, skip=4)), shape=84), num_stack=4))
env.seed(42)
env.action_space.seed(42)
np.random.seed(42)
state = env.reset()

if len(state.shape) == 3:
    state = np.expand_dims(state, axis=0)


class DoubleDeepQNN(tf.keras.Model):

    def __init__(self, input_shape, output_shape):
        super(DoubleDeepQNN, self).__init__()

        self.model = models.Sequential()
        self.model.add(layers.Conv2D(
            32, (8, 8), input_shape=input_shape[1:], activation='relu', strides=4))
        self.model.add(layers.Conv2D(64, (4, 4), activation='relu', strides=2))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu', strides=1))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(output_shape))

        self.target = keras.models.clone_model(self.model)
        self.target.trainable = False

        self.model.compiled_metrics = None
        self.target.compiled_metrics = None

    def call(self, inputs):
        return self.model(inputs)

    def target(self, inputs):
        return self.target(inputs)


def batch(tensor, index):
    result = []
    for x in range(len(tensor)):
        result.append(tensor[x][index[x]])

    return tf.convert_to_tensor(result)


class DQNNAgent:
    def __init__(self, save_directory, action_space):
        self.save_directory = save_directory
        self.action_space = action_space
        self.net = DoubleDeepQNN(state.shape, action_space)
        self.exploration_rate = 1.0
        self.exploration_rate_decay = 0.999
        self.exploration_rate_min = 0.01
        self.gamma = .95
        self.max_len = 80000
        self.memory = deque(maxlen=self.max_len)
        self.batch_size = 64
        self.episode_rewards = []
        self.moving_average_episode_rewards = []
        self.current_episode_reward = 0.0
        self.current_step = 0
        self.sync_period = 1e4

        self.optimizer = keras.optimizers.Adam()

    def log_episode(self):
        self.episode_rewards.append(self.current_episode_reward)
        self.current_episode_reward = 0.0

    def log_period(self, episode, epsilon, step, checkpoint_period):
        self.moving_average_episode_rewards.append(np.round(
            np.mean(self.episode_rewards[-checkpoint_period:]), 3))
        print(f"Episode {episode} - Step {step} - Epsilon {epsilon} "
              f"- Mean Reward {self.moving_average_episode_rewards[-1]}")
        plt.plot(self.moving_average_episode_rewards)
        filename = os.path.join(self.save_directory,
                                "episode_rewards_plot.png")
        if exists(filename):
            plt.savefig(filename, format="png")
        with open(filename, "w"):
            plt.savefig(filename, format="png")
        plt.clf()

    def load_checkpoint(self, model_path):
        self.net.model = tf.keras.models.load_model(model_path)
        self.net.target.set_weights(self.net.model.get_weights())
        self.net.model.compile(keras.optimizers.Adam(), loss="mse")
        self.net.target.compile(keras.optimizers.Adam(), loss="mse")

        #with open(self.save_directory + "memory.json", "rb") as f:
        #self.memory = json.load(f)

    def save_model(self):

        #with open(self.save_directory + "memory.json", "wb") as f:
        # json.dump(self.memory, f)

        name = os.path.join(self.save_directory, 'tf_model.h5')
        self.net.model.compile(keras.optimizers.Adam(), loss="mse")
        self.net.model.save(name)
        print('Checkpoint saved to \'{}\''.format(self.save_directory))
        tf.keras.backend.clear_session()
        self.load_checkpoint(name)

    def remember(self, state, next_state, action, reward, done):
        self.memory.append((state, next_state, action, reward, done))

    def recall(self):
        state, next_state, action, reward, done = map(np.stack,
                                                      zip(*random.sample(self.memory, self.batch_size)))

        return state, next_state, np.squeeze(action), np.squeeze(reward), np.squeeze(done)

    def gradient_descent(self, step_reward):
        self.current_episode_reward += step_reward

        if (self.current_step % self.sync_period) == 0:
            self.net.target.set_weights(self.net.model.get_weights())

        if self.batch_size > len(self.memory) or len(self.memory) < 10000:
            return

        state, next_state, action, reward, done = self.recall()

        with tf.GradientTape() as tape:
            q_estimate = self.net(state)

            index = np.dstack((np.arange(self.batch_size), action))
            q_estimate = tf.gather_nd(q_estimate, index)

            best_action = np.argmax(self.net(next_state), axis=1)

            index = np.dstack((np.arange(self.batch_size), best_action))
            next_q = self.net.target(next_state)
            next_q = tf.gather_nd(next_q, index)

            q_target = (reward + (1 - done) * self.gamma * next_q)

            loss_value = keras.losses.MSE(q_estimate, q_target)

            grads = tape.gradient(loss_value, self.net.trainable_weights)
            self.optimizer.apply_gradients(
                zip(grads, self.net.trainable_weights))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            action = random.randrange(0, self.action_space)

        else:
            state = np.array(state)
            if len(state.shape) == 3:
                state = np.expand_dims(state, axis=0)
            predicted = self.net(state)
            action = np.argmax(predicted)

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(
            self.exploration_rate_min, self.exploration_rate)
        self.current_step += 1

        return action


save_directory = "mario_dqn"


def train():

    agent = DQNNAgent(save_directory, env.action_space.n)

    if exists(os.path.join(save_directory, 'tf_model.h5')):
        agent.load_checkpoint(os.path.join(save_directory, 'tf_model.h5'))
    episode = 0
    checkpoint_period = 20
    save_period = checkpoint_period
    while True:
        state = env.reset()
        while True:
            action = agent.act(state)
            
            # env.render()

            next_state, reward, done, info = env.step(action)
            agent.remember(state, next_state, action, reward, done)

            # perform gradient descent with minibatch
            agent.gradient_descent(reward)
            state = next_state
            if done:
                episode += 1
                agent.log_episode()
                if episode % checkpoint_period == 0:
                    if episode % save_period == 0:
                      agent.save_model()
                    agent.log_period(
                        episode=episode,
                        epsilon=agent.exploration_rate,
                        step=agent.current_step,
                        checkpoint_period=checkpoint_period
                    )
                break


def play():

    agent = DQNNAgent(save_directory, env.action_space.n)
    
    if exists(os.path.join(save_directory, 'tf_model.h5')):
        agent.load_checkpoint(os.path.join(save_directory, 'tf_model.h5'))
        
    while True:
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            env.render()
            next_state, reward, done, info = env.step(action)
            state = next_state


if "__name__" == "__main__":
    train()