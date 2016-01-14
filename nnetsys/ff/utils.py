from __future__ import division
from .layers import FeedforwardLayer
import theano
import theano.tensor as T
import numpy as np
from .teacher import MinibatchSGDTeacher

class Validator(MinibatchSGDTeacher):
    """
    A network validator. Used to compute statistics about given neural network in
    context of a validation/test set.
    """
    def __init__(self, ff_net, validate_set, l1=0, l2=0):
        """
        Initialize the validator

        :param ff_net: a Network to validate
        :param train_set: a training set. A tuple of (X values vector, Y values vector)
        :param l1: specify optionally if you want to use a loss function with L1 regularization
        :param l2: specify optionally if you want to use a loss function with L2 regularization
        """
        self.ff_net = ff_net
        self.validate_set_x, self.validate_set_y = validate_set
        self.x = T.matrix('x', dtype=theano.config.floatX)
        self.y = T.ivector('y')
        self.classes = max(self.validate_set_y.get_value())+1

        shape = self.validate_set_x.get_value().shape[0]

        self._validate = theano.function([],
                                         T.sum(
                                         T.eq(
                                         T.argmax(
                                                  ff_net.get_passthrough(self.validate_set_x),
                                                  axis=1
                                                  ), self.validate_set_y)
                                         ) / shape)

        self.classify = theano.function([self.x],
                                T.argmax(
                                            ff_net.get_passthrough(self.x),
                                            axis=1
                                ))



        self.lossfun = MinibatchSGDTeacher.get_lossfun(self, l1, l2)

        self.get_loss = theano.function(
            inputs=[],
            outputs=self.lossfun,
            givens={
                self.x: self.validate_set_x,
                self.y: self.validate_set_y,
            }
        )

    def calculate_loss(self, returning='mean'):
        """
        Calculate loss function
        :param returning: what to return
            'max' - maximum loss function on minibatches
            'mean' - mean loss on minibatches
        """

        loss = self.get_loss()
        if returning == 'mean':
            return np.mean(loss)
        elif returning == 'max':
            return np.max(loss)
        else:
            raise ValueError('Unknown returning mode')

    def calculate_confusion_matrix(self):
        """
        Return a confusion matrix. First dimension is the computed value, second - the true value
        """
        confusion_matrix = np.zeros((self.classes, self.classes), dtype=np.int32)

        vals = self.classify(self.validate_set_x.get_value())

        for computed_val, true_val in zip(vals, self.validate_set_y.get_value()):
            confusion_matrix[computed_val][true_val] += 1

        return confusion_matrix

    def validate(self):
        """Return a fraction of correctly classified entries.
        :return: 0 <= x <= 1"""
        return self._validate()

    def __repr__(self):
        return '<Validator over %s>' % (repr(self.ff_net), )

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


class Classifier(object):
    """
    A wrapper around a Network to make it into a classifier.

    Classifier assumes the last layer is done using softmax
    """

    def __init__(self, net, dtype=theano.config.floatX):
        self.ff_net = net
        self.x = T.matrix('x', dtype=dtype)

        self.classify = theano.function([self.x],
                                T.argmax(
                                             net.get_passthrough(self.x),
                                             axis=1
                                         )
                                        )

    def __repr__(self):
        return '<Classifier over %s>' % (repr(self.ff_net), )


class Evaluator(object):
    """
    A wrapper around a Network to make it into evaluator.

    It just outputs the last vector
    """
    def __init__(self, net, dtype=theano.config.floatX):
        self.ff_net = net
        self.x = T.matrix('x', dtype=dtype)

        self.evaluate = theano.function([self.x],
                                        net.get_passthrough(self.x))

    def __repr__(self):
        return '<Evaluator over %s>' % (repr(self.ff_net), )