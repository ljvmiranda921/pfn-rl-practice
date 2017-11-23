# -*- coding: utf-8 -*-

"""Tests all environment classes EasyEnv and CartPoleEnv"""

import unittest
import random

from linear_model import LinearModel

class TestLinearModel(unittest.TestCase):

    def setUp(self):
        self.regular_obs_dim = 4
        self.model_ = LinearModel(1)
        self.model_.params = [5]

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
        new_obs = [random.uniform(-1,1) for i in range(random.randint(5,10))]
        model = LinearModel(self.regular_obs_dim)
        with self.assertRaises(AssertionError):
            model.action(new_obs)

    def test_return_sign_positive(self):
        """Check if the sign returned is as expected given the input"""
        pos_obs = [random.uniform(1,2)]
        self.assertEqual(self.model_.action(pos_obs),1)

    def test_return_sign_negative(self):
        """Check if the sign returned is as expected given the input"""
        neg_obs = [random.uniform(-2,-1)]
        self.assertEqual(self.model_.action(neg_obs),-1)
    
    def test_return_sign_negative_for_zero(self):
        """Check if the sign returned is as expected given the input"""
        zero_obs = [0]
        self.assertEqual(self.model_.action(zero_obs),-1)