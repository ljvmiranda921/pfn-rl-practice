# -*- coding: utf-8 -*-

"""Contains the cross entropy method"""

import random


class CrossEntropyMethod(object):
    """Method for optimizing the parameters in the model"""

    def __init__(self, N, p):
        """Initializes the model

        Parameters
        ----------
        N : int
            number of samples to generate
        p : float
            the amount of elite parameters to obtain
        """
        self.N = N
        self.p = p

    def sample_parameters(self, params):
        """Generates an N x dims matrix of parameter samples

        Parameters
        ----------
        params : list
            the original set of parameters

        Returns
        -------
        list of lists
            an N x dims matrix of parameter samples
        """
        noisy_params = []

        for i in range(self.N):
            noise = [random.normalvariate(0,1) for i in range(len(params))]
            params_i = [x+y for x,y in zip(params, noise)]
            noisy_params.append(params_i)

        return noisy_params

    def get_elite_parameters(self, samples, rewards):
        """Obtains the top  N * p% parameters based on the reward

        Parameters
        ----------
        samples : list of lists
            a set of sampled parameters with rows as samples and columns
            as the parameter value
        rewards : list
            the sample's corresponding reward

        Returns
        -------
        list of lists
            the top parameters in the sample
        """
        elite_indices = sorted(range(len(rewards)),
                               key=lambda i: rewards[i],
                               reverse=True)[:int(self.N*self.p)]

        top_samples = []
        for i in elite_indices:
            top_samples.append(samples[i])

        return top_samples

    def get_parameter_mean(self, params):
        """Obtains the mean (by column) of params.

        Parameters
        ----------
        params : list of lists
            the set of parameters with rows as samples and columns as
            parameter value

        Returns
        -------
        list
            a list with the same dimension as :code:`params`
        """
        mean = lambda x : sum(x) / len(x)
        mean_params = [mean(x) for x in zip(*params)]

        return mean_params
