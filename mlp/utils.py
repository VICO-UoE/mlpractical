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