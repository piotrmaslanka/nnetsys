# -*- coding: utf-8 -*-
"""
Module that implements teachers to train your network,
and validators to validate it

The network must be a FeedforwardLayer instance
:author: Piotr Maślanka
"""
from __future__ import division
import theano
import theano.tensor as T
import numpy as np


class Validator(object):
    """
    A network validator
    """
    def __init__(self, ff_net, validate_set):
        """
        Initialize the validator    
        
        :param ff_net: a Network to validate
        :param train_set: a training set. A tuple of (X values vector, Y values vector)
        """        
        self.ff_net = ff_net
        self.validate_set_x, self.validate_set_y = validate_set
        self.x = T.dmatrix('x')

        shape = self.validate_set_x.get_value().shape[0]

        self._validate = theano.function([], 
                                         T.sum(
                                         T.eq(
                                         T.argmax(
                                                  ff_net.get_passthrough(self.validate_set_x), 
                                                  axis=1
                                                  ), self.validate_set_y)
                                         ) / shape)

    def validate(self):
        """Return a percentage of correctly classified entries"""
        return self._validate()
    
    def __repr__(self):
        return '<Validator over %s>' % (repr(self.ff_net), )

class MinibatchSGDTeacher(object):
    """
    Minibatch stochastic gradient descent teacher with momentum support.
    This teacher uses mean negative log-likelihood as loss function, with optional L1 and L2 terms
    
    Single epoch means iterating over every minibatch
    """
    def __init__(self, ff_net, train_set, batch_size=10, 
                                          learning_rate=0.1, 
                                          momentum=0, 
                                          l1=None, 
                                          l2=None,
                                          dtype=theano.config.floatX):
        """
        Initialize the teacher.        
        
        :param ff_net: a Network to train
        :param train_set: a training set. A tuple of (X values vector, Y values vector)
        :param batch_size: minibatch size
        :param alpha: starting learning rate
        :param momentum: starting momentum
        :param l1: weight of L1 term. None (default) for no L1 term
        :param l2: weight of L2 term. None (default) for no L2 term
        :param dtype: target types for x and y. Leave default for GPU-friendly
        """
        self.ff_net = ff_net
        self.batch_size = batch_size
        self.train_set_x, self.train_set_y = train_set
        if dtype == 'float32':
            learning_rate = np.float32(learning_rate)
        self._learning_rate = theano.shared(learning_rate)
        self.batch_count = self.train_set_x.get_value(borrow=True).shape[0] // batch_size
        self.x = T.matrix('x', dtype=dtype)
        self.y = T.ivector('y')
        self.index = T.iscalar()        # minibatch index      
        
        self.lossfun = self.get_lossfun(l1, l2)
        
        self.updates = [
            (parameter, parameter - self._learning_rate * T.grad(self.lossfun, parameter))
            for parameter in self.ff_net.get_parameters()
        ]

        self.train_model = theano.function(
            inputs = [self.index],
            outputs=self.lossfun,
            updates=self.updates,
            givens={
                self.x: self.train_set_x[self.index*self.batch_size : (self.index+1)*self.batch_size],
                self.y: self.train_set_y[self.index*self.batch_size : (self.index+1)*self.batch_size],
            }
        )


    def get_lossfun(self, l1=None, l2=None):
        """
        Generate a loss function
        
        The default one is mean negative log-likelihood
        
        :param l1: weight of L1 term, None for no L1 term
        :param l2: weight of L2 term, None for no L2 term
        """
        q = -T.mean(
                    T.log(
                            self.ff_net.get_learning_passthrough(self.x)
                         )[T.arange(self.y.shape[0]), self.y]
                      )

        try:
            if l1 is not None:
                q = q + self.ff_net.l1 * l1
        except AttributeError:
            pass

        try:
            if l2 is not None:
                q = q + self.ff_net.l2 * l2
        except AttributeError:
            pass

        return q


    @property
    def momentum(self):
        """Get current momentum value"""
        try:
            return self._momentum.get_value()
        except AttributeError:
            return 0.0

    @momentum.setter
    def momentum(self, value):
        """Set new momentum value"""
        self._momentum.set_value(value)

    @property
    def learning_rate(self):
        """Get current learning rate"""
        return self._learning_rate.get_value()
        
    @learning_rate.setter
    def learning_rate(self, value):
        """Set new learning rate"""
        self._learning_rate.set_value(np.float32(value))
        
    def train_epoch(self):
        for minibatch_index in xrange(0, self.batch_count):
            self.train_model(minibatch_index)
        
        
