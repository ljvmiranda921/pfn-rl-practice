# -*- coding: utf-8 -*-

"""Contains all environment classes EasyEnv and CartPoleEnv"""

import random


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
        assert (action in [-1, 1]), 'Invalid input. Must be -1 or 1'

        current_state = self.prev_obs
        reward = action * current_state[0]
        done = False
        self.step_counter += 1

        if self.step_counter > 9:
            done = True

        self.prev_obs = [random.uniform(-1, 1)]
        return (self.prev_obs, reward, done)
