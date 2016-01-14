from __future__ import division
from __future__ import print_function
import theano, numpy as np, pickle

from nnetsys.ff import Network, PerceptronLayer, MinibatchSGDTeacher, Validator, Classifier, ConvolutionalLayer, \
                       ReshapeLayer, MaxpoolingLayer, LossFunctionComputer

data = pickle.load(open('../mnist.pkl', 'rb'))
data = [(
   theano.shared(np.asarray(das[0], dtype=theano.config.floatX), borrow=True),
   theano.shared(np.asarray(das[1], dtype=np.int32), borrow=True)
 ) for das in data]


nnet = Network(
            ReshapeLayer((-1, 1, 28, 28)),
            ConvolutionalLayer(5, 5, 1, 20),     # after this is x20x24x24
            MaxpoolingLayer(2, 2),              # after this is x20x12x12
            ConvolutionalLayer(5, 5, 20, 50),     # after this is x50x8x8
            MaxpoolingLayer(2, 2),              # after this is x50x4x4
            ReshapeLayer((-1, 50*4*4)),

            PerceptronLayer(50*4*4, 500, activation='tanh'),
            PerceptronLayer(500, 10, activation='softmax')
   )

teacher = MinibatchSGDTeacher(nnet, data[0], batch_size=500, learning_rate=0.1)
loss_comp = LossFunctionComputer(nnet, data[1], batch_size=250)
validator = Validator(nnet, data[2])
classifier = Classifier(nnet)

for i in xrange(0, 100):

    error = validator.validate() *100
    print('Accuracy is %s%%' % (error, ))

    print('Loss function for validation data set is %s' % (loss_comp.compute_loss(returning='mean'), ))

#    teacher.learning_rate = 1 / (i+2)
    teacher.train_epoch()


