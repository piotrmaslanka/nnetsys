# -*- coding: utf-8 -*-
"""
This module contains classes that are feed-forward layers or provide capability 
to aggregate layers into a logical layer

:author: Piotr Maślanka
"""
import theano
import theano.tensor as T
import numpy as np

__all__ = ('Perceptron', 'Network')

class FeedforwardLayer(object):
    """
    ABSTRACT. Base class for feedforward layers.
    """
    
    def __init__(self, n_in, n_out):
        """
        :param n_in: Number of inputs
        :param n_out: Number of outputs
        """
        
        self.n_in = n_in      #: public attribute
        self.n_out = n_out    #: public attribute
        
        # self.l1- declare that if L1 norm can be computed. L1 norm is a Theano graph
        # self.l2- declare that if L2 norm can be computed. L2 norm is a Theano graph
        
    def get_parameters(self):
        """Return a list of (Theano shared variables) which are parameters of this layer"""
        return []

    def get_passthrough(self, x):
        """Return a function that given NIN-dimensional input x (Theano symbolic) returns
           NOUT-dimensional output"""
        return None

    def get_learning_passthrough(self, x):
        """a get_passthrough() used only when learning. 
        This is left to implement things like dropout"""
        return None

class Perceptron(FeedforwardLayer):
    """
    A single layer of neurons - a perceptron with relu/sigmoid/tanh/softmax activation
    """
    def __init__(self, n_in, n_out, activation='relu', dropout=0, rng=None, dtype=theano.config.floatX):
        """
        :param n_in: length of input vector
        :param n_out: length of output vector (and amount of neurons in this layer)
        :param activation: activation function. Pass relu or tanh or sigmoid or softmax
        :param dropout: probability of dropping a neuron. 0 is default, and means no dropout
                        this parameter is only relevant for teaching the network, validation
                        and classification is done without dropping neurons
        :param rng: numpy RandomState instance. If left default (None) one will be created
        :param dtype: internal Theano datatype
        """
        
        FeedforwardLayer.__init__(self, n_in, n_out)
        
        self.rng = rng or np.random.RandomState(0)
        
        self.activation_name = activation        
        
        if activation == 'tanh':
            w = np.asarray(
                self.rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=dtype
            )
            b = np.zeros((n_out, ), dtype=dtype)
            self.activation = T.tanh
        elif activation == 'relu':
            w = np.asarray(
                self.rng.normal(0, 0.2, size=(n_in, n_out)),
                dtype=dtype
            )
            b = np.zeros((n_out, ), dtype=dtype)
            try:
                self.activation = T.nnet.relu   # supported in Theano 0.7.2+
            except AttributeError:
                self.activation = T.nnet.softplus    # almost same thing
        elif activation == 'sigmoid':
            w = np.asarray(
                self.rng.uniform(
                    low=-4*np.sqrt(6. / (n_in + n_out)),
                    high=4*np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=dtype
            )
            b = np.zeros((n_out, ), dtype=dtype)
            self.activation = T.nnet.sigmoid    
        elif activation == 'softmax':
            w = np.zeros((n_in, n_out), dtype=dtype)
            b = np.zeros((n_out, ), dtype=dtype)
            self.activation = T.nnet.softmax
        else:
            raise ValueError('Invalid activation function. Use relu/tanh/sigmoid/softmax')
            
        self.W = theano.shared(w)
        self.B = theano.shared(b)      
        
        self.l1 = abs(self.W).sum()
        self.l2 = abs(self.W ** 2).sum()
        
        self.dropout = dropout

    def get_learning_passthrough(self, x):
        ptx = self.get_passthrough(x)        
        
        if self.dropout == 0:
            return ptx
        else:
            # build a mask
            srng = T.shared_randomstreams.RandomStreams(self.rng.randint(999999))        
                # p=1-p because 1's indicate keep and p is prob of dropping
            mask = srng.binomial(n=1, p=1-self.dropout, size=(x.shape[0], self.n_out))
            # The cast is important because int * float32 = float64 which pulls things off the gpu
            return ptx * T.cast(mask, theano.config.floatX)

    def get_passthrough(self, x):
        if self.dropout == 0:
            return self.activation(T.dot(x,self.W)+self.B)
        else:
            return self.activation(T.dot(x,self.W * self.dropout)+self.B)

    def get_parameters(self):
        return [self.W, self.B]            
            
    def __repr__(self):
        return '<Perceptron %s->%s (%s)>' % (self.n_in, self.n_out, self.activation_name)
            
            
class Network(FeedforwardLayer):
    """
    A virtual layer that represents a bunch of feed-forward layers.
    
    It is used to construct feed-forward neural networks. Use it like:
    
        s = Network(Perceptron(200, 100), 
                    Perceptron(100, 50), 
                    Perceptron(50, 5, activation='softmax'))
    """

    def __init__(self, *layers):
        """
        Build the network.        
        
        :param layers: list of layers
        """
        self.layers = layers
        
        can_l1 = True       
        can_l2 = True
        
        for layer in layers:
            if not isinstance(layer, FeedforwardLayer):
                raise ValueError('All layers must be feed-forward')
            can_l1 = can_l1 and hasattr(layer, 'l1')
            can_l2 = can_l2 and hasattr(layer, 'l2')
            
        if can_l1:
            self.l1 = sum([x.l1 for x in layers])

        if can_l2:
            self.l2 = sum([x.l2 for x in layers])


        FeedforwardLayer.__init__(self, self.layers[0].n_in, self.layers[-1].n_out)

    def get_parameters(self):
        a = []
        for layer in self.layers:
            for param in layer.get_parameters():
                a.append(param)
        return a

    def get_passthrough(self, x):
        pt = self.layers[0].get_passthrough(x)
        for layer in self.layers[1:]:
            pt = layer.get_passthrough(pt)
        return pt

    def get_learning_passthrough(self, x):
        pt = self.layers[0].get_learning_passthrough(x)
        for layer in self.layers[1:]:
            pt = layer.get_learning_passthrough(pt)
        return pt
            
    def __repr__(self):
        return '<Network %s->%s (%s layers)>' % (self.n_in, self.n_out, len(self.layers))
            
            
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
