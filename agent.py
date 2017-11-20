# -*- coding: utf-8 -*-

"""Agent that interacts with the host program"""

import sys
import random
from environments import CartPoleEnv

def main():
    env = CartPoleEnv()

    for i_episode in range(2):
        # Reset command
        obs = env.reset()

        while True:
            sys.stderr.write('Obs: {}\n'.format(obs))
            action = random.randint(0,1) * 2 - 1
            obs, reward, done = env.step(action)

            if done:
                break

    # Terminate the host program
    env.terminate()

if __name__ == '__main__':
    main()
