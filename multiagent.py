# -*- coding: utf-8 -*-

"""Runs multiple agents in parallel to test hyperparameters"""

import sys
from subprocess import Popen
from argparse import ArgumentParser


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('-n', '--sample-size',
                        dest='n', help='sweep sample size',
                        action='store_true', default=False)
    parser.add_argument('-p', '--elite-size',
                        dest='p', help='sweep elite size',
                        action='store_true', default=False)
    parser.add_argument('-z', '--step-size',
                        dest='z', help='sweep step size',
                        action='store_true', default=False)
    return parser

def agent_sample_size():
    """Runs the agent with multiple CEM sample size (N) for five trials"""
    sizes=[25,50,75,100,125,150,175,200]
    for sz in sizes:
        commands = [
            ['./cartpole.out', 'python3 agent.py -n {} -o ./output/n-{}-1'.format(sz,sz)],
            ['./cartpole.out', 'python3 agent.py -n {} -o ./output/n-{}-2'.format(sz,sz)],
            ['./cartpole.out', 'python3 agent.py -n {} -o ./output/n-{}-3'.format(sz,sz)],
            ['./cartpole.out', 'python3 agent.py -n {} -o ./output/n-{}-4'.format(sz,sz)],
            ['./cartpole.out', 'python3 agent.py -n {} -o ./output/n-{}-5'.format(sz,sz)],
        ]

        # Run in parallel
        processes = [Popen(cmd) for cmd in commands]
        for p in processes: p.wait()

def agent_elite_size():
    """Runs the agent with multiple elite size (p) for five trials"""
    elite_size = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for sz in elite_size:
        commands = [
            ['./cartpole.out', 'python3 agent.py -p {} -o ./output/p-{}-1'.format(sz,sz)],
            ['./cartpole.out', 'python3 agent.py -p {} -o ./output/p-{}-2'.format(sz,sz)],
            ['./cartpole.out', 'python3 agent.py -p {} -o ./output/p-{}-3'.format(sz,sz)],
            ['./cartpole.out', 'python3 agent.py -p {} -o ./output/p-{}-4'.format(sz,sz)],
            ['./cartpole.out', 'python3 agent.py -p {} -o ./output/p-{}-5'.format(sz,sz)],
        ]

        # Run in parallel
        processes = [Popen(cmd) for cmd in commands]
        for p in processes: p.wait()

def agent_step_size():
    """Runs the agent with multiple step sizes for five trials"""
    step_size = [100, 250, 500, 750, 1000]
    for sz in step_size:
        commands = [
            ['./cartpole.out', 'python3 agent.py -z {} -o ./output/z-{}-1'.format(sz,sz)],
            ['./cartpole.out', 'python3 agent.py -z {} -o ./output/z-{}-2'.format(sz,sz)],
            ['./cartpole.out', 'python3 agent.py -z {} -o ./output/z-{}-3'.format(sz,sz)],
            ['./cartpole.out', 'python3 agent.py -z {} -o ./output/z-{}-4'.format(sz,sz)],
            ['./cartpole.out', 'python3 agent.py -z {} -o ./output/z-{}-5'.format(sz,sz)],
        ]

        # Run in parallel
        processes = [Popen(cmd) for cmd in commands]
        for p in processes: p.wait()

def main():
    # Build parser
    parser = build_parser()
    options = parser.parse_args()

    if options.n:
        agent_sample_size()
    elif options.p:
        agent_elite_size()
    elif options.z:
        agent_step_size()
    else:
        sys.stderr.write('Invalid argument passed')

if __name__ == '__main__':
    main()