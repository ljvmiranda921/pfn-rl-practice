# -*- coding: utf-8 -*-

"""Tests all environment classes EasyEnv and CartPoleEnv"""

import unittest
import random

from optimizers import CrossEntropyMethod


class TestCEM(unittest.TestCase):

    def setUp(self):
        self.cem = CrossEntropyMethod(N=5,p=0.2)
        self.x = [1,2,3,4]
        self.r = [100,28,1,400]
        self.X = [[1,2,3,4],
                  [4,2,1,4],
                  [3,1,4,2],
                  [2,3,4,1]]

    def test_sample_parameters_shape(self):
        """Check if the returned dimensions of the list is as expected"""
        samples = self.cem.sample_parameters(self.x)
        self.assertEqual(len(samples), self.cem.N)
        self.assertEqual(len(samples[0]), len(self.x))

    def test_get_elite_parameters(self):
        """Check if the top is obtained"""
        elites = self.cem.get_elite_parameters(self.X, self.r)
        self.assertEqual(elites, [[2,3,4,1]])

    def test_get_parameter_mean(self):
        """Check if the expected mean is obtained"""
        mean = self.cem.get_parameter_mean(self.X)
        self.assertEqual(mean, [2.5, 2.0, 3.0, 2.75])
