"""Methods for playing the game from the value iteration agent."""

from Q_Agent.DeepQLearningAgent import *
from Q_Agent.QLearningAgent import *
from Q_Agent.DoubleQLearningAgent import *
episodes = 100


def play_q(env: JoypadSpace, args, actions):
    """Play the game using the Q-learning agent."""
    agent: ValueIterationAgent = ValueIterationAgent(env, actions)
    agent.valueIteration()
    for _ in range(episodes):

        environment = None
        if actions is None:
            actions = env.action_space.n
        else:
            environment = JoypadSpace(gym.make(args.env), actions)
            environment.reset()

        done = False
        _, _, _, info, = environment.step(0)
        state = make_state(info)
        while not done:
            if done:
                _ = environment.reset()

            action = agent.get_action(state)
            next_state, _, done, info = environment.step(action)
            state = make_state(info)
            environment.render()

        # close the environment
        env.close()


def play_double_q(env: JoypadSpace, args, actions):
    """Play the game using the Q-learning agent."""
    agent = DoubleQLearningAgent(env, actions)
    agent.valueIteration()
    for _ in range(episodes):

        environment = None
        if actions is None:
            actions = env.action_space.n
        else:
            environment = JoypadSpace(gym.make(args.env), actions)
            environment.reset()

        done = False
        _, _, _, info, = environment.step(0)
        state = make_state(info)
        while not done:
            if done:
                _ = environment.reset()

            action = agent.get_action(state)
            next_state, _, done, info = environment.step(action)
            state = make_state(info)
            environment.render()

        # close the environment
        env.close()
