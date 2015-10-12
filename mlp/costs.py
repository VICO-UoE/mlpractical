# Machine Learning Practical (INFR11119),
# Pawel Swietojanski, University of Edinburgh


import numpy


class Cost(object):
    """
    Defines an interface for the cost object
    """
    def cost(self, y, t, **kwargs):
        """
        Implements a cost for monitoring purposes
        :param y: matrix -- an output of the model
        :param t: matrix -- an expected output the model should produce
        :param kwargs: -- some optional parameters required by the cost
        :return: the scalar value representing the cost given y and t
        """
        raise NotImplementedError()

    def grad(self, y, t, **kwargs):
        """
        Implements a gradient of the cost w.r.t y
        :param y: matrix -- an output of the model
        :param t: matrix -- an expected output the model should produce
        :param kwargs: -- some optional parameters required by the cost
        :return: matrix - the gradient of the cost w.r.t y
        """
        raise NotImplementedError()

    def get_name(self):
        return 'cost'


class MSECost(Cost):
    def cost(self, y, t, **kwargs):
        se = 0.5*numpy.sum((y - t)**2, axis=1)
        return numpy.mean(se)

    def grad(self, y, t, **kwargs):
        return y - t

    def get_name(self):
        return 'mse'


class CECost(Cost):
    """
    Cross Entropy (Negative log-likelihood) cost for multiple classes
    """
    def cost(self, y, t, **kwargs):
        #assumes t is 1-of-K coded and y is a softmax
        #transformed estimate at the output layer
        nll = t * numpy.log(y)
        return -numpy.mean(numpy.sum(nll, axis=1))

    def grad(self, y, t, **kwargs):
        #assumes t is 1-of-K coded and y is a softmax
        #transformed estimate at the output layer
        return y - t

    def get_name(self):
        return 'ce'
