# coding: utf-8

# Learning layer
# Fully connected layer
#
# TODO: add dropout, momentum and weight-decay
#
# Date: Feb 13, 2016
# Author: Satoshi SHIMADA

import numpy

import learning_layer
import neural_layer

class FullyConnectedLayer \
    (neural_layer.NeuralLayer, learning_layer.LearningLayer):
    def __init__(self, input_size, output_size, activation, learning_rate, dropout_rate, \
            weights=None, biases=None):
        if weights == None:
            weights = numpy.random.randn(output_size, input_size)
        if biases == None:
            biases = numpy.random.randn(output_size, 1)
        self.weights = weights
        self.biases  = biases
        self.input_size    = input_size
        self.output_size   = output_size
        self.activation    = activation
        self.learning_rate = learning_rate
        self.dropout_rate  = dropout_rate

    def feedForward(self, inputs):
        # dropout process
        self.before_activation = numpy.dot(self.weights, inputs) + self.biases
        return self.activation.apply(self.before_activation)

    def backPropagation(self, inputs, delta, prev_out=None):
        # calc delta
        if prev_out != None:
            next_delta = numpy.dot(self.weights.transpose(), delta) * self.activation.diff(prev_out)
        else:
            next_delta = None
        self.weights -= self.learning_rate * numpy.dot(delta, inputs.transpose())
        self.biases  -= self.learning_rate * delta
        return next_delta

    def getOutput(self):
        return self.before_activation

    def getWeights(self):
        return self.weights

    def getBiases(self):
        return self.biases

