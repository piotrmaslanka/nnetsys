nnetsys
=======

*nnetsys* is a library enabling very easy creation, teaching, and testing of neural networks.

For example, if you wanted to train a simple DNN, you would just type:

```python
net = Network(
    Perceptron(500, 300, activation='relu', dropout=0.5),
    Perceptron(300, 200, activation='relu', dropout=0.5),
    Perceptron(200, 50, activation='relu', dropout=0.5),
    Perceptron(50, 10, activation='softmax')
    )
    
teacher = MinibatchSGDTeacher(net, training_data, batch_size=10, learning_rate=0.1)
for i in xrange(0, 100):
    teacher.train_epoch()
```

It can't get much simpler, can it?

nnetsys supports:
* multilayer perceptrons (sigmoid, tanh)
* deep neural networks (relu + dropout)
* convolutional neural networks