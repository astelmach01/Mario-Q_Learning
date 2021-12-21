import json
import random

import numpy as np
from nes_py.wrappers import JoypadSpace

from DeepQLearningAgent import SkipFrame
from util import Counter

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

file_name = 'q_tables\\regular_reward_low_alpha_high_gamma.txt'

class QLearningAgent:

    def __init__(self, env, actions, alpha=.1, gamma=.9, exploration_rate=1, exploration_rate_min=.1,
                 exploration_rate_decay=0.999999972, iterations=10000):

        self.env: SkipFrame = SkipFrame(env, skip=5)
        self.actions = actions

        self.alpha = alpha
        self.gamma = gamma

        self.exploration_rate = exploration_rate
        self.exploration_rate_min = exploration_rate_min
        self.exploration_rate_decay = exploration_rate_decay

        self.iterations = iterations
        self.prev_score = 0

        self.starting_state = None

        # try to load in q table from previously written text file
        # try:
        #     values = json.load(open(file_name))
        #     self.q_values = Counter()
        #     for key, value in values.items():
        #         self.q_values[key] = float(value)
        # except:
        #     self.q_values = Counter()

        self.q_values = Counter(self.env.action_space.n)

    def make_state(self, info):
        return str(info["x_pos"]) + "," + str(info["y_pos"])

    def epsilon_greedy_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            action = random.randint(0, self.env.action_space.n - 1)
        else:
            action = self.get_action(state)

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        return action

    def updateQValue(self, reward, state, action, next_max):
        self.q_values[state][action] += self.alpha * (reward + self.gamma * next_max - self.q_values[state][action])

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

                self.env.render()

                x_s.add(info["x_pos"])
                
            print("Iteration " + str(i) + ": x_pos = " + str(info["x_pos"]) + ". Reward: " + str(
                reward) + ". Q-value: " + str(self.q_values[state][action]) + ". Epsilon: " + str(
                self.exploration_rate))

        print("Training finished.\n")
        print("Largest x_pos: " + str(max(x_s)))
        print("Num done well: " + str(num_done_well))

        # write q table to file
        self.q_values = dict((''.join(str(k)), str(v)) for k, v in self.q_values.items())

        with open(file_name, 'w') as convert_file:
                convert_file.write(json.dumps(self.q_values)) 

        with open(file_name + "x_s.txt", 'w') as f:
            for item in x_s:
                f.write("%s\n" % item)

    def get_action(self, state):
        return np.argmax(self.q_values[state])

    def get_max_value(self, state): 
        return np.max(self.q_values[state])
