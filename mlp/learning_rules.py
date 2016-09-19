# -*- coding: utf-8 -*-
"""Learning rules."""

import numpy as np


class GradientDescentLearningRule(object):

    def __init__(self, learning_rate=1e-3):
        self.learning_rate = learning_rate

    def initialise(self, params):
        self.params = params

    def reset(self):
        pass

    def update_params(self, grads_wrt_params):
        for param, grad in zip(self.params, grads_wrt_params):
            param -= self.learning_rate * grad


class MomentumLearningRule(object):

    def __init__(self, learning_rate=1e-3, mom_coeff=0.9):
        self.learning_rate = learning_rate
        self.mom_coeff = mom_coeff

    def initialise(self, params):
        self.params = params
        self.moms = []
        for param in self.params:
            self.moms.append(np.zeros_like(param))

    def reset(self):
        for mom in zip(self.moms):
            mom *= 0.

    def update_params(self, grads_wrt_params):
        for param, mom, grad in zip(self.params, self.moms, grads_wrt_params):
            mom *= self.mom_coeff
            mom -= self.learning_rate * grad
            param += mom
