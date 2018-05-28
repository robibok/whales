# Copied from Lasagne

"""
Functions to create initializers for parameter variables.

Usage
-------
>>from lasagne.layers import Dense
>>from lasagne.init import Constant, Glorot
>>Dense((100,20), num_units=50, W=GlorotUniform(), b=Constant(0.0))
"""

import numpy as np
import theano

def floatX(arr):
    """
    Shortcut to turn a numpy array into an array with the
    correct dtype for Theano.
    """
    return arr.astype(theano.config.floatX)

# TODO: numpy rng?
class Initializer(object):
    """Initializer class

    The :class:`Initializer` class represents a weight initializer used
    to initialize weight parameters in a neural network layer. It should be
    subclassed when implementing new types of weight initializers.

    """
    def __call__(self, shape):
        """
        Makes :class:`Initializer` instances callable like a function, invoking
        their :meth:`sample()` method.
        """
        return self.sample(shape)

    def sample(self, shape):
        """
        Sample should return a theano.tensor of size shape and data type
        theano.config.floatX.

        Parameters
        -----------
        shape : tuple or int
            Integer or tuple specifying the size of the returned
            matrix.
        returns : theano.tensor
            Matrix of size shape and dtype theano.config.floatX.
        """
        raise NotImplementedError()


class Normal(Initializer):
    """Sample initial weights from the Gaussian distribution

    Initial weight parameters are sampled from N(mean, std).

    Parameters
    ----------
    std : float
        Std of initial parameters.
    mean : float
        Mean of initial parameters.
    """
    def __init__(self, std=0.01, mean=0.0):
        self.std = std
        self.mean = mean

    def sample(self, shape):
        print shape
        return floatX(np.random.normal(self.mean, self.std, size=shape))

    def __str__(self):
        return 'Normal(std=%f, mean=%f)' % (self.std, self.mean)


class Uniform(Initializer):
    """Sample initial weights from the uniform distribution

    Parameters are sampled from U(a, b).

    Parameters
    ----------
    range : float or tuple
        When std is None then range determines a, b. If range is a float the
        weights are sampled from U(-range, range). If range is a tuple the
        weights are sampled from U(range[0], range[1]).
    std : float or None
        If std is a float then the weights are sampled from
        U(mean - np.sqrt(3) * std, mean + np.sqrt(3) * std).
    mean : float
        see std for description.
    """
    def __init__(self, range=0.01, std=None, mean=0.0):
        import warnings
        warnings.warn("The uniform initializer no longer uses Glorot et al.'s "
                      "approach to determine the bounds, but defaults to the "
                      "range (-0.01, 0.01) instead. Please use the new "
                      "GlorotUniform initializer to get the old behavior. "
                      "GlorotUniform is now the default for all layers.")

        if std is not None:
            a = mean - np.sqrt(3) * std
            b = mean + np.sqrt(3) * std
        else:
            try:
                a, b = range  # range is a tuple
            except TypeError:
                a, b = -range, range  # range is a number

        self.range = (a, b)


    def sample(self, shape):
        return floatX(np.random.uniform(
            low=self.range[0], high=self.range[1], size=shape))

    def __str__(self):
        return 'Uniform(a=%f, b=%f)' % (self.range[0], self.range[1])


class Constant(Initializer):
    """Initialize weights with constant value.

    Parameters
    ----------
     val : float
        Constant value for weights.
    """
    def __init__(self, val=0.0):
        self.val = val

    def sample(self, shape):
        return floatX(np.ones(shape) * self.val)

    def __str__(self):
        return 'Constant(val=%f)' % (self.val)


class Fixed(Initializer):
    def __init__(self, arr):
        self.arr = arr

    def sample(self, shape):
        assert(shape == self.arr.shape)
        return floatX(self.arr)

    def __str__(self):
        return 'Fixed()' + str(self.arr)


class He(Initializer):
    """He weight initialization [1]_.
    Weights are initialized with a standard deviation of
    :math:`\\sigma = gain \\sqrt{\\frac{1}{fan_{in}}}`.
    Parameters
    ----------
    initializer : lasagne.init.Initializer
        Initializer used to sample the weights, must accept `std` in its
        constructor to sample from a distribution with a given standard
        deviation.
    gain : float or 'relu'
        Scaling factor for the weights. Set this to 1.0 for linear and sigmoid
        units, to 'relu' or sqrt(2) for rectified linear units. Other transfer
        functions may need different factors.
    c01b : bool
        For a :class:`lasagne.layers.cuda_convnet.Conv2DCCLayer` constructed
        with ``dimshuffle=False``, `c01b` must be set to ``True`` to compute
        the correct fan-in and fan-out.
    References
    ----------
    .. [1] Kaiming He et al. (2015):
           Delving deep into rectifiers: Surpassing human-level performance on
           imagenet classification. arXiv preprint arXiv:1502.01852.
    See Also
    ----------
    HeNormal  : Shortcut with Gaussian initializer.
    HeUniform : Shortcut with uniform initializer.
    """
    def __init__(self, initializer, gain=1.0, c01b=False):
        if gain == 'relu':
            gain = np.sqrt(2)

        self.initializer = initializer
        self.gain = gain
        self.c01b = c01b

    def sample(self, shape):
        if self.c01b:
            if len(shape) != 4:
                raise RuntimeError(
                    "If c01b is True, only shapes of length 4 are accepted")

            fan_in = np.prod(shape[:3])
        else:
            if len(shape) == 2:
                fan_in = shape[0]
            elif len(shape) > 2:
                fan_in = np.prod(shape[1:])
            else:
                raise RuntimeError(
                    "This initializer only works with shapes of length >= 2")

        std = self.gain * np.sqrt(1.0 / fan_in)
        return self.initializer(std=std).sample(shape)


class HeNormal(He):
    """He initializer with weights sampled from the Normal distribution.
    See :class:`He` for a description of the parameters.
    """
    def __init__(self, gain=1.0, c01b=False):
        super(HeNormal, self).__init__(Normal, gain, c01b)


class HeUniform(He):
    """He initializer with weights sampled from the Uniform distribution.
    See :class:`He` for a description of the parameters.
    """
    def __init__(self, gain=1.0, c01b=False):
        super(HeUniform, self).__init__(Uniform, gain, c01b)