"""
An example of a Deep Neural Network system
"""

from __future__ import division
from __future__ import print_function
import theano, numpy as np, pickle

from nnetsys.ff import Network, Perceptron, MinibatchSGDTeacher, Validator

data = pickle.load(open('../mnist.pkl', 'rb'))
data = [(
   theano.shared(np.asarray(das[0], dtype=theano.config.floatX), borrow=True),
   theano.shared(np.asarray(das[1], dtype=np.int32), borrow=True)
 ) for das in data]


nnet = Network(
            Perceptron(28*28, 500, activation='relu', dropout=0.1),
            Perceptron(500, 300, activation='relu', dropout=0.1),
            Perceptron(300, 100, activation='relu', dropout=0.1),
            Perceptron(100, 30, activation='relu', dropout=0.1),
            Perceptron(30, 10, activation='softmax')
   )

teacher = MinibatchSGDTeacher(nnet, data[0], batch_size=500, learning_rate=0.1)
validator = Validator(nnet, data[1])
test_validator = Validator(nnet, data[2])

for i in xrange(0, 100):
    error = validator.validate() * 100
    print('Accuracy (validation) is %s%%' % (error, ))
#    teacher.learning_rate = 1 / (i+2)     -  uncommen to adapt learning rate
    teacher.train_epoch()


test_error = 100 - (test_validator.validate() * 100)
print('Model error rate is %s%%' % (test_error, ))

