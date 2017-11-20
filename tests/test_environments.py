# -*- coding: utf-8 -*-

"""Tests all environment classes EasyEnv and CartPoleEnv"""

import unittest
import random

from environments import EasyEnv, CartPoleEnv


class TestEasyEnv(unittest.TestCase):

    def setUp(self):
        self.env = EasyEnv()

    def test_reset_return_type(self):
        """Check if a list is returned"""
        self.assertIsInstance(self.env.reset(), list)

    def test_step_return_type(self):
        """Check if a 3-tuple of list, float, and bool is returned"""
        env = EasyEnv()
        env.reset()
        obs, reward, done = env.step(action=-1)

        self.assertIsInstance(obs, list)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)

    def test_obs_dim_return_type(self):
        """Check if an integer is returned"""
        self.assertIsInstance(self.env.obs_dim(), int)

    def test_reset_return_dims(self):
        """Check if a 1-dimensional list is returned"""
        self.assertEqual(len(self.env.reset()), 1)

    def test_obs_dim_return_value(self):
        """Check if 1 is returned"""
        env = EasyEnv()
        env.reset()
        self.assertEqual(env.obs_dim(), 1)

    def test_step_wrong_input(self):
        """Check if assertion is raised with wrong input"""
        with self.assertRaises(AssertionError):
            self.env.step(43892.42)

    def test_done_signal_per_episode(self):
        """Check if done signal is triggered at the end of the episode"""
        env = EasyEnv()
        env.reset()

        for t in range(10):
            _, _, done = env.step(action=-1)
            if t != 9:
                # Must be false within 10 steps
                self.assertFalse(done)

        # Must be true at the end of the episode
        self.assertTrue(done)

class TestCartPoleEnv(unittest.TestCase):

    def setUp(self):
        self.env = CartPoleEnv()

    def test_reset_return_type(self):
        """Check if a list is returned"""
        self.assertIsInstance(self.env.reset(), list)

    def test_step_return_type(self):
        """Check if a 3-tuple of list, float, and bool is returned"""
        env = CartPoleEnv()
        env.reset()
        obs, reward, done = env.step(action=-1)

        self.assertIsInstance(obs, list)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)

    def test_obs_dim_return_type(self):
        """Check if an integer is returned"""
        self.assertIsInstance(self.env.obs_dim(), int)

    def test_reset_return_dims(self):
        """Check if a 4-dimensional list is returned"""
        self.assertEqual(len(self.env.reset()), 4)

    def test_obs_dim_return_value(self):
        """Check if 4 is returned"""
        env = CartPoleEnv()
        env.reset()
        self.assertEqual(env.obs_dim(), 4)

    def test_step_wrong_input(self):
        """Check if assertion is raised with wrong input"""
        with self.assertRaises(AssertionError):
            self.env.step(43892.42)

    def test_done_signal_per_episode(self):
        """Check if done signal is triggered at the end of the episode"""
        env = CartPoleEnv()
        env.reset()

        for t in range(10):
            _, _, done = env.step(action=-1)
            if t != 499:
                # Must be false within 10 steps
                self.assertFalse(done)

        # Must be true at the end of the episode
        self.assertTrue(done)
