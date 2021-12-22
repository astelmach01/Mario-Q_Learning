import numpy as np
import json
import random
import numpy as np
import matplotlib.pyplot as plt

import os
from os.path import exists

from util import Counter
from DeepQLearningAgent import SkipFrame

'''
Things you tried
-removed time space
- changed state
-custom reward function
-changed bounds of reward
-special rewards
-changed action space
-declining exploration rate/epsilon
'''

file_name = 'q_table.txt'
checkpoint_period = 20

class QLearningAgent:

    def __init__(self, env, alpha=.1, gamma=.9, exploration_rate=1, exploration_rate_min=.1,
                 exploration_rate_decay=0.999972, iterations=20000):

        self.env: SkipFrame = SkipFrame(env, skip=5)

        self.alpha = alpha
        self.gamma = gamma

        self.exploration_rate = exploration_rate
        self.exploration_rate_min = exploration_rate_min
        self.exploration_rate_decay = exploration_rate_decay

        self.iterations = iterations
        self.prev_score = 0

        self.episode_rewards = []
        self.moving_average_episode_rewards = []
        self.current_episode_reward = 0.0
        self.current_step = 0
        self.save_directory = file_name

        # try to load in q table from previously written text file
        self.q_values = Counter(self.env.action_space.n)
        self.valueIteration()

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

    def make_state(self, info):
        return str(info["x_pos"]) + "," + str(info["y_pos"])

    def epsilon_greedy_action(self, state):
        self.current_step += 1
        if random.uniform(0, 1) < self.exploration_rate:
            action = random.randint(0, self.env.action_space.n - 1)
        else:
            action = self.get_action(state)

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(
            self.exploration_rate_min, self.exploration_rate)

        return action

    def updateQValue(self, reward, state, action, next_max):
        self.q_values[state][action] += self.alpha * \
            (reward + self.gamma * next_max - self.q_values[state][action])

    def valueIteration(self):
        # For plotting metrics
        num_done_well = 0

        x_s = set()

        for i in range(1, self.iterations):
            state = self.env.reset()
            next_state, reward, done, info = self.env.step(0)

            done = False  # if you died and have 0 lives left

            while not done:

                # choose action
                action = self.epsilon_greedy_action(state)

                next_state, reward, done, info = self.env.step(action)
                next_state = self.make_state(info)

                next_max = self.get_max_value(next_state)
                self.updateQValue(reward, state, action, next_max)

                state = next_state

                x_s.add(info["x_pos"])
                self.current_episode_reward += reward

            self.log_episode()
            if i % checkpoint_period == 0:
                self.log_period(i, self.exploration_rate,
                                self.current_step, checkpoint_period)

        print("Training finished.\n")
        print("Largest x_pos: " + str(max(x_s)))
        print("Num done well: " + str(num_done_well))

        # write q table to file
        with open(file_name + 'q_table.txt', 'w') as convert_file:
            convert_file.write(json.dumps(self.q_values))

        with open(file_name + "x_s.txt", 'w') as f:
            for item in x_s:
                f.write("%s\n" % item)

    def get_action(self, state):
        return np.argmax(self.q_values[state])

    def get_max_value(self, state):
        return np.max(self.q_values[state])
