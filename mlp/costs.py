"""Model costs."""


import numpy as np


class MeanSquaredErrorCost(object):
    """
    """

    def __call__(self, outputs, targets):
        return 0.5 * np.mean(np.sum((outputs - targets)**2, axis=1))

    def grad(self, outputs, targets):
        return outputs - targets

    def __repr__(self):
        return 'MeanSquaredErrorCost'


class BinaryCrossEntropyCost(object):
    """
    """

    def __call__(self, outputs, targets):
        return -np.mean(
            targets * np.log(outputs) + (1. - targets) * np.log(1. - ouputs))

    def grad(self, outputs, targets):
        return (1. - targets) / (1. - outputs) - (targets / outputs)

    def __repr__(self):
        return 'BinaryCrossEntropyCost'


class BinaryCrossEntropySigmoidCost(object):
    """
    """

    def __call__(self, outputs, targets):
        probs = 1. / (1. + np.exp(-outputs))
        return -np.mean(
            targets * np.log(probs) + (1. - targets) * np.log(1. - probs))

    def grad(self, outputs, targets):
        probs = 1. / (1. + np.exp(-outputs))
        return probs - targets

    def __repr__(self):
        return 'BinaryCrossEntropySigmoidCost'


class BinaryAccuracySigmoidCost(object):
    """
    """

    def __call__(self, outputs, targets):
        return ((outputs > 0) == targets).mean()

    def ___repr__(self):
        return 'BinaryAccuracySigmoidCost'


class CrossEntropyCost(object):
    """
    """

    def __call__(self, outputs, targets):
        return -np.mean(np.sum(targets * np.log(outputs), axis=1))

    def grad(self, outputs, targets):
        return -targets / outputs

    def __repr__(self):
        return 'CrossEntropyCost'


class CrossEntropySoftmaxCost(object):
    """
    """

    def __call__(self, outputs, targets):
        probs = np.exp(outputs)
        probs /= probs.sum(-1)[:, None]
        return -np.mean(np.sum(targets * np.log(probs), axis=1))

    def grad(self, outputs, targets):
        probs = np.exp(outputs)
        probs /= probs.sum(-1)[:, None]
        return probs - targets

    def __repr__(self):
        return 'CrossEntropySoftmaxCost'


class MulticlassAccuracySoftmaxCost(object):
    """
    """

    def __call__(self, outputs, targets):
        probs = np.exp(outputs)
        return np.mean(np.argmax(probs, -1) == np.argmax(targets, -1))

    def __repr__(self):
        return 'MulticlassAccuracySoftmaxCost'
