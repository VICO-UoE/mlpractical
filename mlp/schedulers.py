# -*- coding: utf-8 -*-
"""Training schedulers.

This module contains classes implementing schedulers which control the
evolution of learning rule hyperparameters (such as learning rate) over a
training run.
"""

import numpy as np


class ConstantLearningRateScheduler(object):
    """Example of scheduler interface which sets a constant learning rate."""

    def __init__(self, learning_rate):
        """Construct a new constant learning rate scheduler object.

        Args:
            learning_rate: Learning rate to use in learning rule.
        """
        self.learning_rate = learning_rate

    def update_learning_rule(self, learning_rule, epoch_number):
        """Update the hyperparameters of the learning rule.

        Runs at the beginning of each epoch.

        Args:
            learning_rule: Learning rule object being used in training run,
                any scheduled hyperparameters to be altered should be
                attributes of this object.
            epoch_number: Integer index of training epoch about to be run.
        """
        learning_rule.learning_rate = self.learning_rate


class ExponentialLearningRateScheduler(object):
    """Exponential decay learning rate scheduler."""

    def __init__(self, init_learning_rate, decay_param):
        """Construct a new learning rate scheduler object.

        Args:
            init_learning_rate: Initial learning rate at epoch 0. Should be a
                positive value.
            decay_param: Parameter governing rate of learning rate decay.
                Should be a positive value.
        """
        self.init_learning_rate = init_learning_rate
        self.decay_param = decay_param

    def update_learning_rule(self, learning_rule, epoch_number):
        """Update the hyperparameters of the learning rule.

        Runs at the beginning of each epoch.

        Args:
            learning_rule: Learning rule object being used in training run,
                any scheduled hyperparameters to be altered should be
                attributes of this object.
            epoch_number: Integer index of training epoch about to be run.
        """
        learning_rule.learning_rate = (
            self.init_learning_rate * np.exp(-epoch_number / self.decay_param))


class ReciprocalLearningRateScheduler(object):
    """Reciprocal decay learning rate scheduler."""

    def __init__(self, init_learning_rate, decay_param):
        """Construct a new learning rate scheduler object.

        Args:
            init_learning_rate: Initial learning rate at epoch 0. Should be a
                positive value.
            decay_param: Parameter governing rate of learning rate decay.
                Should be a positive value.
        """
        self.init_learning_rate = init_learning_rate
        self.decay_param = decay_param

    def update_learning_rule(self, learning_rule, epoch_number):
        """Update the hyperparameters of the learning rule.

        Runs at the beginning of each epoch.

        Args:
            learning_rule: Learning rule object being used in training run,
                any scheduled hyperparameters to be altered should be
                attributes of this object.
            epoch_number: Integer index of training epoch about to be run.
        """
        learning_rule.learning_rate = (
            self.init_learning_rate / (1. + epoch_number / self.decay_param)
        )


class ReciprocalMomentumCoefficientScheduler(object):
    """Reciprocal growth momentum coefficient scheduler."""

    def __init__(self, max_mom_coeff=0.99, growth_param=3., epoch_offset=5.):
        """Construct a new reciprocal momentum coefficient scheduler object.

        Args:
            max_mom_coeff: Maximum momentum coefficient to tend to. Should be
                in [0, 1].
            growth_param: Parameter governing rate of increase of momentum
                coefficient over training. Should be >= 0 and <= epoch_offset.
            epoch_offset: Offset to epoch counter to in scheduler updates to
                govern how quickly momentum initially increases. Should be
                >= 1.
        """
        assert max_mom_coeff >= 0. and max_mom_coeff <= 1.
        assert growth_param >= 0. and growth_param <= epoch_offset
        assert epoch_offset >= 1.
        self.max_mom_coeff = max_mom_coeff
        self.growth_param = growth_param
        self.epoch_offset = epoch_offset

    def update_learning_rule(self, learning_rule, epoch_number):
        """Update the hyperparameters of the learning rule.

        Runs at the beginning of each epoch.

        Args:
            learning_rule: Learning rule object being used in training run,
                any scheduled hyperparameters to be altered should be
                attributes of this object.
            epoch_number: Integer index of training epoch about to be run.
        """
        learning_rule.mom_coeff = self.max_mom_coeff * (
            1. - self.growth_param / (epoch_number + self.epoch_offset)
        )
