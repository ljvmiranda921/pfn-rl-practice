# -*- coding: utf-8 -*-

"""Tests all environment classes EasyEnv and CartPoleEnv"""

import unittest
import random

from linear_model import LinearModel

class TestLinearModel(unittest.TestCase):

    def test_dims_input(self):
        """Check if error is raised when initializing the class with wrong type"""
        with self.assertRaises(TypeError):
            model = LinearModel(312.34)

    def test_params_length(self):
        """Check if length of params attribute equals to dims vector"""
        obs_dim = random.randint(1,10)
        model = LinearModel(obs_dim)
        self.assertEqual(len(model.params), obs_dim)

    def test_action_input(self):
        """Check if error is raised when length of params and obs are not the same"""
        obs_dim = 4
        new_obs = [random.uniform(-1,1) for i in range(random.randint(5,10))]
        model = LinearModel(obs_dim)
        with self.assertRaises(AssertionError):
            model.action(new_obs)

    def test_return_sign(self):
        """Check if the sign returned is as expected given the input"""
        obs_dim = 1
        model = LinearModel(obs_dim)
        pos_obs = [random.uniform(1,2)]
        neg_obs = [random.uniform(-2,-1)]
        zero_obs = [0]

        self.assertEqual(model.action(pos_obs),1)
        self.assertEqual(model.action(neg_obs),-1)
        self.assertEqual(model.action(zero_obs),-1)