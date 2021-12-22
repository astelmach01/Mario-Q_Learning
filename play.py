"""Methods for playing the game from the value iteration agent."""

from DeepQLearningAgent import *
from QLearningAgent import *
from DoubleQLearningAgent import *
episodes = 100


def play_q(env: JoypadSpace, args, actions):
    """Play the game using the Q-learning agent."""
    agent: QLearningAgent = QLearningAgent(env)
    
    for _ in range(episodes):

        environment = None
        if actions is None:
            actions = env.action_space.n
        else:
            environment = SkipFrame(JoypadSpace(gym.make(args.env)), skip=5)
            state = environment.reset()

        done = False
        _, _, _, info, = environment.step(0)
        _, _, _, info, = environment.step(0)
        _, _, _, info, = environment.step(0)


        state = agent.make_state(info)
        while not done:
            action = agent.get_action(state)
            
            _, _, done, info = environment.step(action)
            state = agent.make_state(info)
            
            environment.render()

        # close the environment
        env.close()


def play_double_q(env: JoypadSpace, args, actions):
    """Play the game using the Q-learning agent."""
    agent: DoubleQLearningAgent = DoubleQLearningAgent(env, actions)
    for _ in range(episodes):

        environment = None
        if actions is None:
            actions = env.action_space.n
        else:
            environment = JoypadSpace(gym.make(args.env), actions)
            environment.reset()

        done = False
        _, _, _, info, = environment.step(0)
        state = agent.make_state(info)
        while not done:
            if done:
                _ = environment.reset()

            action = agent.get_action(state)
            _, _, done, info = environment.step(action)
            state = agent.make_state(info)
            environment.render()

        # close the environment
        try:
            env.close()
        except:
            pass
