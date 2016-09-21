# -*- coding: utf-8 -*-
"""Model costs.

This module defines cost functions, with the aim of model training being to
minimise the cost function given a set of inputs and target outputs. The cost
functions typically measure some concept of distance between the model outputs
and target outputs.
"""

import numpy as np


class MeanSquaredErrorCost(object):
    """Mean squared error cost."""

    def __call__(self, outputs, targets):
        """Calculates cost function given a batch of outputs and targets.

        Args:
            outputs: Array of model outputs of shape (batch_size, output_dim).
            targets: Array of target outputs of shape (batch_size, output_dim).

        Returns:
            Scalar cost function value.
        """
        return 0.5 * np.mean(np.sum((outputs - targets)**2, axis=1))

    def grad(self, outputs, targets):
        """Calculates gradient of cost function with respect to outputs.

        Args:
            outputs: Array of model outputs of shape (batch_size, output_dim).
            targets: Array of target outputs of shape (batch_size, output_dim).

        Returns:
            Gradient of cost function with respect to outputs.
        """
        return outputs - targets

    def __repr__(self):
        return 'MeanSquaredErrorCost'


class BinaryCrossEntropyCost(object):
    """Binary cross entropy cost."""

    def __call__(self, outputs, targets):
        """Calculates cost function given a batch of outputs and targets.

        Args:
            outputs: Array of model outputs of shape (batch_size, output_dim).
            targets: Array of target outputs of shape (batch_size, output_dim).

        Returns:
            Scalar cost function value.
        """
        return -np.mean(
            targets * np.log(outputs) + (1. - targets) * np.log(1. - ouputs))

    def grad(self, outputs, targets):
        """Calculates gradient of cost function with respect to outputs.

        Args:
            outputs: Array of model outputs of shape (batch_size, output_dim).
            targets: Array of target outputs of shape (batch_size, output_dim).

        Returns:
            Gradient of cost function with respect to outputs.
        """
        return (1. - targets) / (1. - outputs) - (targets / outputs)

    def __repr__(self):
        return 'BinaryCrossEntropyCost'


class BinaryCrossEntropySigmoidCost(object):
    """Binary cross entropy cost with logistic sigmoid applied to outputs."""

    def __call__(self, outputs, targets):
        """Calculates cost function given a batch of outputs and targets.

        Args:
            outputs: Array of model outputs of shape (batch_size, output_dim).
            targets: Array of target outputs of shape (batch_size, output_dim).

        Returns:
            Scalar cost function value.
        """
        probs = 1. / (1. + np.exp(-outputs))
        return -np.mean(
            targets * np.log(probs) + (1. - targets) * np.log(1. - probs))

    def grad(self, outputs, targets):
        """Calculates gradient of cost function with respect to outputs.

        Args:
            outputs: Array of model outputs of shape (batch_size, output_dim).
            targets: Array of target outputs of shape (batch_size, output_dim).

        Returns:
            Gradient of cost function with respect to outputs.
        """
        probs = 1. / (1. + np.exp(-outputs))
        return probs - targets

    def __repr__(self):
        return 'BinaryCrossEntropySigmoidCost'


class CrossEntropyCost(object):
    """Multi-class cross entropy cost."""

    def __call__(self, outputs, targets):
        """Calculates cost function given a batch of outputs and targets.

        Args:
            outputs: Array of model outputs of shape (batch_size, output_dim).
            targets: Array of target outputs of shape (batch_size, output_dim).

        Returns:
            Scalar cost function value.
        """
        return -np.mean(np.sum(targets * np.log(outputs), axis=1))

    def grad(self, outputs, targets):
        """Calculates gradient of cost function with respect to outputs.

        Args:
            outputs: Array of model outputs of shape (batch_size, output_dim).
            targets: Array of target outputs of shape (batch_size, output_dim).

        Returns:
            Gradient of cost function with respect to outputs.
        """
        return -targets / outputs

    def __repr__(self):
        return 'CrossEntropyCost'


class CrossEntropySoftmaxCost(object):
    """Multi-class cross entropy cost with Softmax applied to outputs."""

    def __call__(self, outputs, targets):
        """Calculates cost function given a batch of outputs and targets.

        Args:
            outputs: Array of model outputs of shape (batch_size, output_dim).
            targets: Array of target outputs of shape (batch_size, output_dim).

        Returns:
            Scalar cost function value.
        """
        probs = np.exp(outputs)
        probs /= probs.sum(-1)[:, None]
        return -np.mean(np.sum(targets * np.log(probs), axis=1))

    def grad(self, outputs, targets):
        """Calculates gradient of cost function with respect to outputs.

        Args:
            outputs: Array of model outputs of shape (batch_size, output_dim).
            targets: Array of target outputs of shape (batch_size, output_dim).

        Returns:
            Gradient of cost function with respect to outputs.
        """
        probs = np.exp(outputs)
        probs /= probs.sum(-1)[:, None]
        return probs - targets

    def __repr__(self):
        return 'CrossEntropySoftmaxCost'
