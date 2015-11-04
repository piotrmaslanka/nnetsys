from __future__ import division
from __future__ import print_function
import theano, numpy as np, pickle

from nnetsys.ff import Network, Perceptron, MinibatchSGDTeacher, Validator, Classifier

data = pickle.load(open('mnist.pkl', 'rb'))
data = [(
   theano.shared(np.asarray(das[0], dtype=theano.config.floatX), borrow=True),
   theano.shared(np.asarray(das[1], dtype=np.int32), borrow=True)                                
 ) for das in data]


nnet = Network(
            Perceptron(28*28, 200, activation='tanh'),
            Perceptron(200, 30, activation='tanh'),
            Perceptron(30, 10, activation='softmax')
   )

teacher = MinibatchSGDTeacher(nnet, data[0], batch_size=20, learning_rate=0.5, l2=0.0001)
validator = Validator(nnet, data[1])
classifier = Classifier(nnet)

#while error > 3:            # learn!
for i in xrange(0, 100):
        
    if (i % 10) == 0:
        error = validator.validate() * 100
        print('Accuracy is %s%%' % (error, ))
    
    teacher.learning_rate = 1 / (i+2)
    teacher.train_epoch()
    

