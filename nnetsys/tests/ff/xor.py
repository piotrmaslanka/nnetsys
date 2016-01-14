import unittest
import theano
import numpy as np

from nnetsys.ff import Perceptron, MinibatchSGDTeacher,Network, Validator, Classifier, Evaluator


def theanify_datasets(test_set):
    data, labels = test_set

    return theano.shared(np.asarray(data, dtype=theano.config.floatX), borrow=True), \
           theano.shared(np.asarray(labels, dtype=np.int32), borrow=True)


test_set = [
    [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ],
        [0,
         1,
         1,
         0]
]

test_set = theanify_datasets(test_set)


class TestXor(unittest.TestCase):

    def test_xor_sigmoid(self):
        nnet = Network(
            Perceptron(2, 2, activation='sigmoid'),
            Perceptron(2, 1, activation='sigmoid')
        )

        teacher = MinibatchSGDTeacher(nnet, test_set, batch_size=1, learning_rate=0.6)
        e = Evaluator(nnet)

        for _ in xrange(0, 200):
            teacher.train_epoch()

        self.assertEquals(map(lambda x: round(x[0]), e.evaluate(test_set[0].get_value())), [0, 1, 1, 0])

    def test_xor_tanh(self):
        nnet = Network(
            Perceptron(2, 2, activation='tanh'),
            Perceptron(2, 1, activation='tanh')
        )

        teacher = MinibatchSGDTeacher(nnet, test_set, batch_size=1, learning_rate=0.6)
        e = Evaluator(nnet)

        for _ in xrange(0, 200):
            teacher.train_epoch()

        self.assertEquals(map(lambda x: round(x[0]), e.evaluate(test_set[0].get_value())), [0, 1, 1, 0])


    def test_xor_relu(self):
        nnet = Network(
            Perceptron(2, 2, activation='relu'),
            Perceptron(2, 2, activation='softmax')
        )

        teacher = MinibatchSGDTeacher(nnet, test_set, batch_size=1, learning_rate=0.05)
        e = Evaluator(nnet)
        c = Classifier(nnet)

        for _ in xrange(0, 400):
            teacher.train_epoch()

        self.assertTrue(all(p == q for p, q in zip(c.classify(test_set[0].get_value()), [0,1,1,0])))