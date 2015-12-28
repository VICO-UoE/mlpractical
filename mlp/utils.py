# Machine Learning Practical (INFR11119),
# Pawel Swietojanski, University of Edinburgh

import numpy
from mlp.layers import Layer


def numerical_gradient(f, x, eps=1e-4, **kwargs):
    """
    Implements the following numerical gradient rule
    df(x)/dx = (f(x+eps)-f(x-eps))/(2eps)
    """

    xc = x.copy()
    g = numpy.zeros_like(xc)
    xf = xc.ravel()
    gf = g.ravel()

    for i in xrange(xf.shape[0]):
        xx = xf[i]
        xf[i] = xx + eps
        fp_eps, ___ = f(xc, **kwargs)
        xf[i] = xx - eps
        fm_eps, ___ = f(xc, **kwargs)
        xf[i] = xx
        gf[i] = (fp_eps - fm_eps)/(2*eps)

    return g


def verify_gradient(f, x, eps=1e-4, tol=1e-6, **kwargs):
    """
    Compares the numerical and analytical gradients.
    """
    fval, fgrad = f(x=x, **kwargs)
    ngrad = numerical_gradient(f=f, x=x, eps=eps, tol=tol, **kwargs)

    fgradnorm = numpy.sqrt(numpy.sum(fgrad**2))
    ngradnorm = numpy.sqrt(numpy.sum(ngrad**2))
    diffnorm = numpy.sqrt(numpy.sum((fgrad-ngrad)**2))

    if fgradnorm > 0 or ngradnorm > 0:
        norm = numpy.maximum(fgradnorm, ngradnorm)
        if not (diffnorm < tol or diffnorm/norm < tol):
            raise Exception("Numerical and analytical gradients "
                            "are different: %s != %s!" % (ngrad, fgrad))
    else:
        if not (diffnorm < tol):
            raise Exception("Numerical and analytical gradients "
                            "are different: %s != %s!" % (ngrad, fgrad))
    return True


def verify_layer_gradient(layer, x, eps=1e-4, tol=1e-6):

    assert isinstance(layer, Layer), (
        "Expected to get the instance of Layer class, got"
        " %s " % type(layer)
    )

    def grad_layer_wrapper(x, **kwargs):
        h = layer.fprop(x)
        deltas, ograds = layer.bprop(h=h, igrads=numpy.ones_like(h))
        return numpy.sum(h), ograds

    return verify_gradient(f=grad_layer_wrapper, x=x, eps=eps, tol=tol, layer=layer)


def test_conv_linear_fprop(layer, kernel_order='ioxy', kernels_first=True,
                           dtype=numpy.float):
    """ 
    Tests forward propagation method of a convolutional layer.
    
    Checks the outputs of `fprop` method for a fixed input against known
    reference values for the outputs and raises an AssertionError if
    the outputted values are not consistent with the reference values. If
    tests are all passed returns True.
    
    Parameters
    ----------
    layer : instance of Layer subclass
        Convolutional (linear only) layer implementation. It must implement
        the methods `get_params`, `set_params` and `fprop`.
    kernel_order : string
        Specifes dimension ordering assumed for convolutional kernels
        passed to `layer`. Default is `ioxy` which corresponds to:
            input channels, output channels, image x, image y
        The other option is 'oixy' which corresponds to
            output channels, input channels, image x, image y
        Any other value will raise a ValueError exception.
    kernels_first : boolean
        Specifies order in which parameters are passed to and returned from
        `get_params` and `set_params`. Default is True which corresponds
        to signatures of `get_params` and `set_params` being:
            kernels, biases = layer.get_params()
            layer.set_params([kernels, biases])
        If False this corresponds to signatures of `get_params` and 
        `set_params` being:
            biases, kernels = layer.get_params()
            layer.set_params([biases, kernels])
    dtype : numpy data type
         Data type to use in numpy arrays passed to layer methods. Default
         is `numpy.float`.
            
    Raises
    ------
    AssertionError
        Raised if output of `layer.fprop` is inconsistent with reference
        values either in shape or values.
    ValueError
        Raised if `kernel_order` is not a valid order string.
    """
    inputs = numpy.arange(96).reshape((2, 3, 4, 4)).astype(dtype)
    kernels = numpy.arange(-12, 12).reshape((3, 2, 2, 2)).astype(dtype)
    if kernel_order == 'oixy':
        kernels = kernels.swapaxes(0, 1)
    elif kernel_order != 'ioxy':
        raise ValueError('kernel_order must be one of "ioxy" and "oixy"')
    biases = numpy.arange(2).astype(dtype)
    true_output = numpy.array(
      [[[[  496.,   466.,   436.],
         [  376.,   346.,   316.],
         [  256.,   226.,   196.]],
        [[ 1385.,  1403.,  1421.],
         [ 1457.,  1475.,  1493.],
         [ 1529.,  1547.,  1565.]]],
       [[[ -944.,  -974., -1004.],
         [-1064., -1094., -1124.],
         [-1184., -1214., -1244.]],
        [[ 2249.,  2267.,  2285.],
         [ 2321.,  2339.,  2357.],
         [ 2393.,  2411.,  2429.]]]], dtype=dtype)
    try:
        orig_params = layer.get_params()
        if kernels_first:
            layer.set_params([kernels, biases])
        else:
            layer.set_params([biases, kernels])
        layer_output = layer.fprop(inputs)
        assert layer_output.shape == true_output.shape, (
            'Layer fprop gives incorrect shaped output. '
            'Correct shape is {0} but returned shape is {1}.'
            .format(true_output.shape, layer_output.shape)
        )
        assert numpy.allclose(layer_output, true_output), (
            'Layer fprop does not give correct output. '
            'Correct output is {0}\n but returned output is {1}.'
            .format(true_output, layer_output)
        )
    finally:
        layer.set_params(orig_params)
    return True

  
def test_conv_linear_bprop(layer, kernel_order='ioxy', kernels_first=True,
                           dtype=numpy.float):
    """ 
    Tests input gradients backpropagation method of a convolutional layer.
    
    Checks the outputs of `bprop` method for a fixed input against known
    reference values for the outputs and raises an AssertionError if
    the outputted values are not consistent with the reference values. If
    tests are all passed returns True.
    
    Parameters
    ----------
    layer : instance of Layer subclass
        Convolutional (linear only) layer implementation. It must implement
        the methods `get_params`, `set_params` and `bprop`.
    kernel_order : string
        Specifes dimension ordering assumed for convolutional kernels
        passed to `layer`. Default is `ioxy` which corresponds to:
            input channels, output channels, image x, image y
        The other option is 'oixy' which corresponds to
            output channels, input channels, image x, image y
        Any other value will raise a ValueError exception.
    kernels_first : boolean
        Specifies order in which parameters are passed to and returned from
        `get_params` and `set_params`. Default is True which corresponds
        to signatures of `get_params` and `set_params` being:
            kernels, biases = layer.get_params()
            layer.set_params([kernels, biases])
        If False this corresponds to signatures of `get_params` and 
        `set_params` being:
            biases, kernels = layer.get_params()
            layer.set_params([biases, kernels])
    dtype : numpy data type
         Data type to use in numpy arrays passed to layer methods. Default
         is `numpy.float`.
            
    Raises
    ------
    AssertionError
        Raised if output of `layer.bprop` is inconsistent with reference
        values either in shape or values.
    ValueError
        Raised if `kernel_order` is not a valid order string.
    """
    inputs = numpy.arange(96).reshape((2, 3, 4, 4)).astype(dtype)
    kernels = numpy.arange(-12, 12).reshape((3, 2, 2, 2)).astype(dtype)
    if kernel_order == 'oixy':
        kernels = kernels.swapaxes(0, 1)
    elif kernel_order != 'ioxy':
        raise ValueError('kernel_order must be one of "ioxy" and "oixy"')
    biases = numpy.arange(2).astype(dtype)
    igrads = numpy.arange(-20, 16).reshape((2, 2, 3, 3)).astype(dtype)
    true_ograds = numpy.array(
      [[[[ 328.,  605.,  567.,  261.],
         [ 534.,  976.,  908.,  414.],
         [ 426.,  772.,  704.,  318.],
         [ 170.,  305.,  275.,  123.]],
        [[  80.,  125.,  119.,   45.],
         [  86.,  112.,  108.,   30.],
         [  74.,  100.,   96.,   30.],
         [  18.,   17.,   19.,    3.]],
        [[-168., -355., -329., -171.],
         [-362., -752., -692., -354.],
         [-278., -572., -512., -258.],
         [-134., -271., -237., -117.]]],
       [[[ -32.,  -79., -117.,  -63.],
         [-114., -248., -316., -162.],
         [-222., -452., -520., -258.],
         [-118., -235., -265., -129.]],
        [[   8.,   17.,   11.,    9.],
         [  14.,   40.,   36.,   30.],
         [   2.,   28.,   24.,   30.],
         [  18.,   53.,   55.,   39.]],
        [[  48.,  113.,  139.,   81.],
         [ 142.,  328.,  388.,  222.],
         [ 226.,  508.,  568.,  318.],
         [ 154.,  341.,  375.,  207.]]]], dtype=dtype)
    try:
        orig_params = layer.get_params()
        if kernels_first:
            layer.set_params([kernels, biases])
        else:
            layer.set_params([biases, kernels])
        layer_deltas, layer_ograds = layer.bprop(None, igrads)
        assert layer_deltas.shape == igrads.shape, (
            'Layer bprop give incorrectly shaped deltas output.'
            'Correct shape is {0} but returned shape is {1}.'
            .format(igrads.shape, layer_deltas.shape)
        )
        assert numpy.allclose(layer_deltas, igrads), (
            'Layer bprop does not give correct deltas output. '
            'Correct output is {0}\n but returned output is {1}.'
            .format(igrads, layer_deltas)
        )
        assert layer_ograds.shape == true_ograds.shape, (
            'Layer bprop gives incorrect shaped ograds output. '
            'Correct shape is {0} but returned shape is {1}.'
            .format(true_ograds.shape, layer_ograds.shape)
        )
        assert numpy.allclose(layer_ograds, true_ograds), (
            'Layer bprop does not give correct ograds output. '
            'Correct output is {0}\n but returned output is {1}.'
            .format(true_ograds, layer_ograds)
        )
    finally:
        layer.set_params(orig_params)
    return True

   
def test_conv_linear_pgrads(layer, kernel_order='ioxy', kernels_first=True,
                            dtype=numpy.float):
    """ 
    Tests parameter gradients backpropagation method of a convolutional layer.
    
    Checks the outputs of `pgrads` method for a fixed input against known
    reference values for the outputs and raises an AssertionError if
    the outputted values are not consistent with the reference values. If
    tests are all passed returns True.
    
    Parameters
    ----------
    layer : instance of Layer subclass
        Convolutional (linear only) layer implementation. It must implement
        the methods `get_params`, `set_params` and `pgrads`.
    kernel_order : string
        Specifes dimension ordering assumed for convolutional kernels
        passed to `layer`. Default is `ioxy` which corresponds to:
            input channels, output channels, image x, image y
        The other option is 'oixy' which corresponds to
            output channels, input channels, image x, image y
        Any other value will raise a ValueError exception.
    kernels_first : boolean
        Specifies order in which parameters are passed to and returned from
        `get_params` and `set_params`. Default is True which corresponds
        to signatures of `get_params` and `set_params` being:
            kernels, biases = layer.get_params()
            layer.set_params([kernels, biases])
        If False this corresponds to signatures of `get_params` and 
        `set_params` being:
            biases, kernels = layer.get_params()
            layer.set_params([biases, kernels])
    dtype : numpy data type
         Data type to use in numpy arrays passed to layer methods. Default
         is `numpy.float`.
            
    Raises
    ------
    AssertionError
        Raised if output of `layer.pgrads` is inconsistent with reference
        values either in shape or values.
    ValueError
        Raised if `kernel_order` is not a valid order string.
    """
    inputs = numpy.arange(96).reshape((2, 3, 4, 4)).astype(dtype)
    kernels = numpy.arange(-12, 12).reshape((3, 2, 2, 2)).astype(dtype)
    biases = numpy.arange(2).astype(dtype)
    deltas = numpy.arange(-20, 16).reshape((2, 2, 3, 3)).astype(dtype)
    true_kernel_grads = numpy.array(
      [[[[  390.,   264.],
         [ -114.,  -240.]],
        [[ 5088.,  5124.],
         [ 5232.,  5268.]]],
       [[[-1626., -1752.],
         [-2130., -2256.]],
        [[ 5664.,  5700.],
         [ 5808.,  5844.]]],
       [[[-3642., -3768.],
         [-4146., -4272.]],
        [[ 6240.,  6276.],
         [ 6384.,  6420.]]]], dtype=dtype)
    if kernel_order == 'oixy':
        kernels = kernels.swapaxes(0, 1)
        true_kernel_grads = true_kernel_grads.swapaxes(0, 1)
    elif kernel_order != 'ioxy':
        raise ValueError('kernel_order must be one of "ioxy" and "oixy"')
    true_bias_grads = numpy.array([-126.,   36.], dtype=dtype)
    try:
        orig_params = layer.get_params()
        if kernels_first:
            layer.set_params([kernels, biases])
        else:
            layer.set_params([biases, kernels])
        layer_kernel_grads, layer_bias_grads = layer.pgrads(inputs, deltas)
        assert layer_kernel_grads.shape == true_kernel_grads.shape, (
            'Layer pgrads gives incorrect shaped kernel gradients output. '
            'Correct shape is {0} but returned shape is {1}.'
            .format(true_kernel_grads.shape, layer_kernel_grads.shape)
        )
        assert numpy.allclose(layer_kernel_grads, true_kernel_grads), (
            'Layer pgrads does not give correct kernel gradients output. '
            'Correct output is {0}\n but returned output is {1}.'
            .format(true_kernel_grads, layer_kernel_grads)
        )
        assert layer_bias_grads.shape == true_bias_grads.shape, (
            'Layer pgrads gives incorrect shaped bias gradients output. '
            'Correct shape is {0} but returned shape is {1}.'
            .format(true_bias_grads.shape, layer_bias_grads.shape)
        )
        assert numpy.allclose(layer_bias_grads, true_bias_grads), (
            'Layer pgrads does not give correct bias gradients output. '
            'Correct output is {0}\n but returned output is {1}.'
            .format(true_bias_grads, layer_bias_grads)
        )
    finally:
        layer.set_params(orig_params)
    return True

