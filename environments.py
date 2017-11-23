# -*- coding: utf-8 -*-

"""Contains all environment classes EasyEnv and CartPoleEnv"""

import sys
import random
import subprocess


class EasyEnv(object):
    """An environment where training the agent is very easy"""

    def __init__(self):
        """Initializes the environment

        Attributes
        ----------
        prev_obs : list
            the previous observation or current state before action is
            applied
        """
        self.prev_obs = None

    def reset(self):
        """A method that resets the environment.

        The return value is a vector that indicates the observation
        reward of the initial state of the environment.

        Returns
        -------
        list
            1-dimensional vector sampled uniformly in the interval [-1,1]
        """
        self.prev_obs = [random.uniform(-1, 1)]
        self.step_counter = 0

        return self.prev_obs

    def obs_dim(self):
        """Returns the number of dimensions of the observation vector

        Returns
        -------
        int
            always returns 1
        """
        return 1

    def step(self, action):
        """Applies an action to the environment.

        Call the previous step’s observation (the component from the
        1-dimensional vector) :code:`prev_obs`. If :code:`step` is called
        immediately after :code:`reset`, :code:`prev_obs` will take the
        value equal to the return value of :code:`reset`. The
        observation for this step will be sampled uniformly from [−1,1].
        The reward is :code:`action * prev_obs`. The environment will
        end after taking 10 steps

        Parameters
        ----------
        action : int
            the action to be taken. Either -1 or 1.

        Returns
        -------
        list
            the new observation
        float
            reward signal
        bool
            stop signal

        Raises
        ------
        AssertionError
            if input is not -1 or 1.
        """
        assert action in [-1, 1], 'Invalid input. Must be -1 or 1'

        current_state = self.prev_obs
        reward = action * current_state[0]
        self.step_counter += 1

        # Obtain next observation
        self.prev_obs = [random.uniform(-1, 1)]

        # Check if episode is done
        done = True if self.step_counter > 9 else False

        return (self.prev_obs, reward, done)

class CartPoleEnv(object):
    """Environment that interacts with the host program"""

    def __init__(self):
        """Initializes the environment

        Attributes
        ----------
        prev_obs : list
            the previous observation or current state before action is
            applied
        """
        self.prev_obs = None

    def reset(self):
        """A method that resets the environment.

        The return value is a vector that indicates the observation
        reward of the initial state of the environment. This method
        pipes to :code:`cartpole.out` to send reset instructions and
        obtains the new observation.

        Returns
        -------
        list
            4-dimensional vector sampled uniformly in the interval [-1,1]
        """
        # Flush reset to stdout
        print('r')
        sys.stdout.flush()
        feedback = input()
        feedback = feedback.split()
        self.prev_obs = [float(i) for i in feedback[1:]]

        return self.prev_obs

    def obs_dim(self):
        """Returns the number of dimensions of the observation vector

        Returns
        -------
        int
            the number of dimensions in  the observation vector
        """
        return 4

    def step(self, action):
        """Applies an action to the environment

        This method pipes the output of :code:`cartpole.out` to obtain
        the state. The reward is always 1 and one episode consists of
        500 steps.

        Parameters
        ----------
        action : int
            the action to be taken. Either -1 or 1.

        Returns
        -------
        list
            the new observation
        float
            reward signal, always 1
        bool
            stop signal

        Raises
        ------
        AssertionError
            if input is not -1 or 1.
        """
        assert action in [-1,1], 'Invalid input. Must be -1 or 1'

        current_state = self.prev_obs
        reward = 1

        # Obtain next observation
        print('s {}'.format(action))
        sys.stdout.flush()
        feedback = input()
        feedback = feedback.split()
        self.prev_obs = [float(i) for i in feedback[1:]]

        # Check if episode is done
        done = True if (feedback[0]=='done') else False

        return (self.prev_obs, reward, done)

    def terminate(self):
        """Terminates the host program"""
        print('q')
        sys.stdout.flush()