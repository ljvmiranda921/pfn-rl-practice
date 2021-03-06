# -*- coding: utf-8 -*-

"""Agent that interacts with the host program"""

import csv
import sys
import random
from argparse import ArgumentParser

from environments import CartPoleEnv
from linear_model import LinearModel
from optimizers import CrossEntropyMethod

EPISODES = 100
SAMPLING_RATE = 100
TOP_SAMPLES = 0.1
PRINT_STEP = 100
RANDOM_SEED = 42
STEP_SIZE = 500

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('-e','--episodes',
                        dest='episodes',help='number of episodes',
                        type=int, default=EPISODES)
    parser.add_argument('-n', '--sampling-size',
                        dest='n', help='no. of samples to generate',
                        type=int, default=SAMPLING_RATE)
    parser.add_argument('-p', '--top-samples',
                        dest='p', help='no. of top samples to take',
                        type=float, default=TOP_SAMPLES)
    parser.add_argument('-z', '--step-size',
                        dest='step_size', help='no. of steps for each episode',
                        type=int, default=STEP_SIZE)
    parser.add_argument('-s', '--print-step',
                        dest='print_step', help='amount of steps to print the output observation',
                        type=int, default=PRINT_STEP)
    parser.add_argument('-o', '--output-file',
                        dest='output_file', help='file to store win ratio per episode',
                        required=False)
    parser.add_argument('-r', '--random-seed',
                        dest='random_seed', help='sets the random seed',
                        type=int, default=RANDOM_SEED)
    return parser

def noisy_evaluation(model, env, steps, noisy_params):
    """Runs an episode based on the noisy parameters sampled by CEM
    and returns the reward"""
    model = update_model(model, noisy_params)
    reward = run_episode(model, env, steps=steps)
    return reward

def update_model(model, parameters):
    """Updates the model using the parameters"""
    model.params = parameters
    return model

def run_episode(model, env, steps, print_step=False):
    """Runs an episode for a number of steps and returns the total reward"""
    obs = env.reset()
    episode_reward = 0

    for s in range(steps):

        if print_step:
            if s % print_step == 0:
                sys.stderr.write('Step {} (obs): {}\n'.format(s, obs))

        action = model.action(obs)
        obs, reward, done = env.step(action)
        episode_reward += reward

        if done:
            break

    return episode_reward

def main():
    # Build parser
    parser = build_parser()
    options = parser.parse_args()

    # Set random seed
    random.seed(options.random_seed)

    # Get CEM methods
    cem = CrossEntropyMethod(N=options.n, p=options.p)
    # Create environment object
    env = CartPoleEnv()
    # Create linear model
    model = LinearModel(dims=env.obs_dim())

    # Initialize parameters
    params = model.params

    # Episode scores
    win_ratio_list = []

    successful_episodes = 0
    for i_episode in range(options.episodes):
        sys.stderr.write('\n###### Episode {} of {} ###### \n'.format(i_episode+1, options.episodes))

        # Sample N parameter vectors
        noisy_params = cem.sample_parameters(params)
        # Evaluate the sampled vectors
        rewards = [noisy_evaluation(model, env, options.step_size, i) for i in noisy_params]
        # Get elite parameters based on reward
        elite_params = cem.get_elite_parameters(noisy_params,rewards)
        # Update parameters
        params = cem.get_parameter_mean(elite_params)
        episode_reward = run_episode(model=update_model(model,params), env=env, steps=options.step_size, print_step=options.print_step)
        win_ratio = episode_reward / options.step_size
        sys.stderr.write('Episode reward: {} ({:.2f}%)\n'.format(episode_reward, win_ratio))
        # Save win_ratio
        win_ratio_list.append(win_ratio)

        if episode_reward >= options.step_size:
            successful_episodes += 1

    sys.stderr.write('\nFinal params: {}'.format(model.params))
    sys.stderr.write('\nRun finished. {} out of {} episodes ({:.2f}%) have a reward of atleast {}\n'.format(successful_episodes, options.episodes, successful_episodes / options.episodes, options.step_size))

    # If output_file is given, write scores to disk
    if options.output_file:
        sys.stderr.write('\nWriting scores to file: {}.csv...\n'.format(options.output_file))
        with open(options.output_file + '.csv', 'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(win_ratio_list)
        sys.stderr.write('Done!\n')

    # Terminate the host program
    env.terminate()

if __name__ == '__main__':
    main()
