# coding: utf-8

# Learning layer
# Fully connected layer
#
# TODO: add weights-decay, weights-limit
#
# Date: Feb 16, 2016
# Author: Satoshi SHIMADA

import numpy

import learning_layer
import neural_layer

class FullyConnectedLayer \
    (neural_layer.NeuralLayer, learning_layer.LearningLayer):
    def __init__(self, input_size, output_size, activation, learning_rate, dropout_rate, momentum_rate \
            weights=None, biases=None):
        if weights == None:
            weights = numpy.random.randn(output_size, input_size)
        if biases == None:
            biases = numpy.zeros((output_size, 1))
        self.weights = weights
        self.biases  = biases
        self.input_size    = input_size
        self.output_size   = output_size
        self.activation    = activation
        self.learning_rate = learning_rate
        self.dropout_rate  = dropout_rate
        self.momentum_rate = momentum_rate
        self.erase_units   = int(input_size * dropout_rate)
        self.prev_delta_weights = 0.

    def feedForward(self, inputs, train=False):
        self.used_index = []
        if train:
            # dropout process
            for i in xrange(self.erase_units):
                # delete connection
                index = numpy.random.randint(self.output_size)
                while True:
                    if index in self.used_index:
                        index = numpy.random.randint(self.output_size)
                    else:
                        break
                inputs[index] = 0.
            weights = self.weights
            biases = self.biases
        else:
            weights = self.weights * self.dropout_rate 
            biases = self.biases * self.dropout_rate
        self.before_activation = numpy.dot(weights, inputs) + biases
        return self.activation._apply(self.before_activation)

    def backPropagation(self, inputs, delta, prev_out=None):
        # calc delta
        if prev_out != None:
            next_delta = numpy.dot(self.weights.transpose(), delta) * self.activation._diff(prev_out)
        else:
            next_delta = None
        delta_weights = self.learning_rate * numpy.dot(delta, inputs.transpose())
        delta_biases  = self.learning_rate * delta
        self.weights -= delta_weights + self.prev_delta_weights
        self.biases  -= delta_biases
        self.prev_delta_weights = delta_weights * self.momentum_rate
        return next_delta

    def getOutput(self):
        return self.before_activation

    def getWeights(self):
        return self.weights

    def getBiases(self):
        return self.biases

