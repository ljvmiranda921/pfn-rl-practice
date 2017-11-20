# -*- coding: utf-8 -*-

"""Contains the linear model"""

import random


class LinearModel(object):
    """A linear function modelling the agent's policy."""

    def __init__(self, dims):
        """Initializes the model

        Parameters
        ----------
        dims : int
            number of dimensions for the linear model.

        Attributes
        ----------
        params : list
            the weight of the linear model with dimensions :code:`dims`
        """
        try:
            self.params = [random.uniform(-1,1) for i in range(dims)]
        except TypeError:
            raise('No. of dimensions should be an integer')

    def action(self, obs):
        """Takes the inner product of the observation and model parameters

        Parameters
        ----------
        obs : list
            the observations to perform inner product into

        Returns
        -------
        int
            1 if inner product is positive and -1 otherwise
        """
        assert len(self.params) == len(obs), "Length of two lists aren't the same"
        inner_prod = sum([x*y for x,y in zip(self.params,obs)])
        sign = lambda k: (k>0) - (k<=0)
        return sign(inner_prod)
