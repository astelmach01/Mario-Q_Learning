from Q_Agent.QLearningAgent import ValueIterationAgent
import random
from Q_Agent.DeepQLearningAgent import SkipFrame
import numpy as np
import json
from collections import Counter
import matplotlib.pyplot as plt
from os.path import exists
file_name = "q_tables\\double_q"



class DoubleQLearningAgent(ValueIterationAgent):

    def __init__(self, env, actions, alpha=.1, gamma=.9, exploration_rate=0, exploration_rate_min=0,
                 exploration_rate_decay=0.9997, iterations=5000):
        self.env = SkipFrame(env, skip=5)
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.exploration_rate = exploration_rate
        self.exploration_rate_min = exploration_rate_min
        self.exploration_rate_decay = exploration_rate_decay
        self.iterations = iterations
        self.episode_rewards = []
        self.moving_average_episode_rewards = []
        self.current_episode_reward = 0.0
        self.period = 20
        self.agent1 = ValueIterationAgent(env, actions)
        self.agent2 = ValueIterationAgent(env, actions)
       # try to load in q table from previously written text file.
       #  try:
       #       values = json.load(open("q_tables\\double_q1st_q_table_new_try.txt"))
       #       self.agent1.q_values = Counter()
       #       for key, value in values.items():
       #           format_value = value.strip('][').split(" ")
       #           format_value = list(map(float, [num for num in format_value if num != ""]))
       #           self.agent1.q_values[key] = format_value
       #       values2 = json.load(open("q_tables\\double_q2nd_q_table_new_try.txt"))
       #       self.agent2.q_values = Counter()
       #       for key, value in values2.items():
       #           format_value = value.strip('][').split(" ")
       #           format_value = list(map(float, [num for num in format_value if num != ""]))
       #           self.agent2.q_values[key] = format_value
       #  except BaseException as e:
       #       print("failed to load: {0}".format(e))
        self.valueIteration()

    def make_state(self, info):
        return str(info['x_pos'])+","+str(info['y_pos'])

    def log_episode(self):
        self.episode_rewards.append(self.current_episode_reward)
        self.current_episode_reward = 0.0

    def log_period(self, episode, epsilon, step, checkpoint_period):
        self.moving_average_episode_rewards.append(np.round(
            np.mean(self.episode_rewards[-checkpoint_period:]), 3))
        print(f"Episode {episode} - Step {step} - Epsilon {epsilon} "
              f"- Mean Reward {self.moving_average_episode_rewards[-1]}")
        plt.plot(self.moving_average_episode_rewards)
        # filename = os.path.join(self.save_directory, "episode_rewards_plot.png")
        filename = 'q_tables\\dq_episode_rewards_plot.png'
        if exists(filename):
            plt.savefig(filename, format="png")
        with open(filename, "w"):
            plt.savefig(filename, format="png")
        plt.clf()

    def make_state_pixeldata(self, state, info):
        processed_state = state
        object = 0
        enemy = 0
        pit = 0
        #probably mario low center
        mario_x = info['x_pixel']+7
        #29
        mario_y = info['y_pixel']+25
        #205, 50
        #205, 47
        # 3 pixels from right hitbox
        pixel_right = processed_state[min(mario_y, 239)][min(mario_x+8, 255)]
        # print(pixel_right)
        pixel_pit_detector = processed_state[239][mario_x+5]
        # if np.array_equal(pixel_pit_detector, [104, 136, 252]):
        #     #print('pit')
        #     pit = 1
        # if np.array_equal(pixel_right, [184, 248, 24]):
        #     pass #bush
        # if np.array_equal(pixel_right, [228, 92, 16]):
        #     #print('goomba')
        #     enemy = 1
        # if np.array_equal(pixel_right, [0, 0, 0]):
        #    # print('pipe')
        #     object = 1
        mario_surroundings_result = 0
        return str(pixel_right)+str(pixel_pit_detector)+str(info['status'])

    def epsilon_greedy_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            action = random.randint(0, self.env.action_space.n - 1)
        else:
            action = self.get_action(state)



        return action

    def get_action(self, state):
        # get action from sum of Q1 and Q2
        if isinstance(state, np.ndarray):
            return random.randint(0, self.env.action_space.n - 1)
        summed_q_values = np.asarray(self.agent1.q_values[state]) + np.asarray(self.agent2.q_values[state])
        max = -99999999
        for val in summed_q_values:
            if val > max:
                max = val
        max_vals = []
        for i in range(len(summed_q_values)):
            if summed_q_values[i] == max:
                max_vals.append(i)
        return random.choice(max_vals)



    def valueIteration(self):
        print(self.env.get_keys_to_action())
        print("number of actions: " + str(self.env.action_space.n))
        # print(self.q_values)
        #   help(self.env.unwrapped)

        # For plotting metrics
        num_done_well = 0
        num_flags_get = 0
        average_x_pos = 0

        # keeping track of the x values we've hit
        x_s = set()
        # changed reward range to -100, 100
        for i in range(1, self.iterations):
            state = self.env.reset()
            next_state, reward, done, info = self.env.step(0)

            done = False  # if you died and have 0 lives left

            # used to end game early
            iteration = 1

            while not done:

                # choose action
                action = self.epsilon_greedy_action(state)

                next_state, reward, done, info = self.env.step(action)
                next_state = self.make_state(info)
                self.current_episode_reward += reward
                # check if you've been in same x position for a while
                # and if so, end game early
                # if iteration % 50 == 0:
                #     if detect == info["x_pos"]:
                #         # reward *= -2
                #         done = True
                #     detect = info["x_pos"]

                # update one agent randomly
                if random.uniform(0, 1) < 0.5:
                    next_max = self.agent2.get_max_value(next_state)
                    self.agent1.updateQValue(reward, state, action, next_max)
                else:
                    next_max = self.agent1.get_max_value(next_state)
                    self.agent2.updateQValue(reward, state, action, next_max)

                state = next_state
                iteration += 1


                self.env.render()

                # amount of times we've gotten past 3rd pipe
                if info["x_pos"] > 1400:
                    num_done_well += 1
                    
                if info["flag_get"]:
                    print("MARIO DID IT")
                    num_flags_get += 1
                x_s.add(info["x_pos"])

            self.log_episode()
            if i % self.period == 0:
                self.log_period(i, self.exploration_rate, iteration, self.period)
            print("Iteration " + str(i) + ": x_pos = " + str(info["x_pos"]) + ". Reward: " + str(
                reward) + ". Q-value 1: " + str(self.agent1.q_values[state][action]) +". Q-value 2: "
                  + str(self.agent2.q_values[state][action]) +
                  ". Epsilon: " + str(
                self.exploration_rate))
            average_x_pos += info['x_pos']
            self.exploration_rate *= self.exploration_rate_decay
            self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        print("Training finished.\n")
        print("Largest x_pos: " + str(max(x_s)))
        print("Num done well: " + str(num_done_well))
        print("Num times Mario won the game: " + str(num_flags_get))
        print("Average reward: " + str(average_x_pos / self.iterations))

        # write q table to file
        writable_values_agent1 = dict((''.join(str(k)), str(v)) for k, v in self.agent1.q_values.items())
        writable_values_agent2 = dict((''.join(str(k)), str(v)) for k, v in self.agent2.q_values.items())


        # try:
        #     with open(file_name + "1st_q_table_new_try.txt", 'w') as convert_file:
        #         convert_file.write(json.dumps(writable_values_agent1))
        # except:
        #    q = 2
        #
        # try:
        #     with open(file_name + "2nd_q_table_new_try.txt", 'w') as convert_file:
        #         convert_file.write(json.dumps(writable_values_agent2))
        # except:
        #     q = 2

        with open(file_name + "x_s.txt", 'w') as f:
            for item in x_s:
                f.write("%s\n" % item)

