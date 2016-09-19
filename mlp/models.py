"""
Model definitions.
"""

from mlp.layers import LayerWithParameters


class SingleLayerModel(object):
    """
    """
    def __init__(self, layer):
        self.layer = layer

    @property
    def params(self):
        """
        """
        return self.layer.params

    def fprop(self, inputs):
        """
        """
        activations = [inputs, self.layer.fprop(inputs)]
        return activations

    def grads_wrt_params(self, activations, grads_wrt_outputs):
        """
        """
        return self.layer.grads_wrt_params(activations[0], grads_wrt_outputs)

    def params_cost(self):
        """
        """
        return self.layer.params_cost()

    def __repr__(self):
        return 'SingleLayerModel(' + str(layer) + ')'


class MultipleLayerModel(object):
    """
    """
    def __init__(self, layers):
        self.layers = layers

    @property
    def params(self):
        """
        """
        params = []
        for layer in self.layers:
            if isinstance(layer, LayerWithParameters):
                params += layer.params
        return params

    def fprop(self, inputs):
        """
        """
        activations = [inputs]
        for i, layer in enumerate(self.layers):
            activations.append(self.layers[i].fprop(activations[i]))
        return activations

    def grads_wrt_params(self, activations, grads_wrt_outputs):
        """
        """
        grads_wrt_params = []
        for i, layer in enumerate(self.layers[::-1]):
            inputs = activations[-i - 2]
            outputs = activations[-i - 1]
            grads_wrt_inputs = layer.bprop(inputs, outputs, grads_wrt_outputs)
            if isinstance(layer, LayerWithParameters):
                grads_wrt_params += layer.grads_wrt_params(
                    inputs, grads_wrt_outputs)[::-1]
            grads_wrt_outputs = grads_wrt_inputs
        return grads_wrt_params[::-1]

    def params_cost(self):
        """
        """
        params_cost = 0.
        for layer in self.layers:
            if isinstance(layer, LayerWithParameters):
                params_cost += layer.params_cost()
        return params_cost

    def __repr__(self):
        return (
            'MultiLayerModel(\n    ' +
            '\n    '.join([str(layer) for layer in self.layers]) +
            '\n)'
        )
