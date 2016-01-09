from .base import FeedforwardLayer
import theano
import theano.tensor as T
import numpy as np


class ReshapeLayer(FeedforwardLayer):
    """
    Layer that does reshaping on input data
    """
    def __init__(self, *args, **kwargs):
        FeedforwardLayer.__init__(self)
        self.args = args
        self.kwargs = kwargs

    def get_passthrough(self, x):
        return x.reshape(*self.args, **self.kwargs)

    def get_learning_passthrough(self, x):
        return self.get_passthrough(x)

    def get_parameters(self):
        return []

    def __repr__(self):
        return 'Reshape: %s' % repr(self.shape, )

