# coding: utf-8

# Learning layer
# Convolution layer
#
# Date: Feb 20, 2016
# Author: Satoshi SHIMADA

import numpy

import learning_layer
import neural_layer

class ConvolutionLayer \
    (neural_layer.NeuralLayer, learning_layer.LearningLayer):
    def __init__(self, input_dim, filter_dim, output_dim, activation, learning_rate):
        self.weights = numpy.random.randn(filter_dim[2], input_dim[2], filter_dim[1], filter_dim[0])
        self.biases  = numpy.zeros((filter_dim[2], 1))
        # width, height, channels
        self.input_dim    = input_dim
        self.input_size   = input_dim[0] * input_dim[1] * input_dim[2]
        # width, height, channels
        self.output_dim   = output_dim
        self.output_size  = output_dim[0] * output_dim[1] * output_dim[2]
        # width, height, count
        self.filter_dim   = filter_dim
        self.activation    = activation
        self.learning_rate = learning_rate

    def feedForward(self, inputs, train=True):
        inputs = inputs.reshape((self.input_dim[2], self.input_dim[1], self.input_dim[0]))
        out = [numpy.zeros((self.output_dim[1], self.output_dim[1])) for k in xrange(self.output_dim[2])]
        for k in xrange(self.filter_dim[2]):
            for i in xrange(self.output_dim[1]):
                for j in xrange(self.output_dim[0]):
                    for c in xrange(self.input_dim[2]):
                        for s in xrange(self.filter_dim[1]):
                            for t in xrange(self.filter_dim[0]):
                                out[k][i][j] += self.weights[k][c][s][t] * inputs[c][i+s][j+t]
                    out[k][i][j] += self.biases[k][0]
        self.before_activation = numpy.array(out).reshape((self.output_size, 1))
        act = self.activation._apply(self.before_activation)
        return act

    def backPropagation(self, inputs, delta, prev_out=None):
        inputs = inputs.reshape((1, 28, 28))
        # calculation delta for next layer
        delta = delta.reshape((self.output_dim[2], self.output_dim[1], self.output_dim[0]))
        next_delta = numpy.zeros((self.input_dim[2], self.output_dim[1], self.output_dim[0]))
        for c in xrange(self.input_dim[2]):
            for i in xrange(self.output_dim[1]):
                for j in xrange(self.output_dim[0]):
                    for k in xrange(self.filter_dim[2]):
                        for s in xrange(self.filter_dim[1]):
                            for t in xrange(self.filter_dim[0]):
                                next_delta[c][i][j] += delta[k][i-s][j-t] * self.weights[k][c][s][t]
        # update parameters
        delta_weights = numpy.zeros(self.weights.shape)
        delta_biases  = numpy.zeros(self.biases.shape)
        for k in xrange(self.filter_dim[2]):
            for c in xrange(self.input_dim[2]):
                for s in xrange(self.filter_dim[1]):
                    for t in xrange(self.filter_dim[0]):
                        for i in xrange(self.output_dim[1]):
                            for j in xrange(self.output_dim[0]):
                                delta_weights[k][c][s][t] += delta[k][i][j] * inputs[c][i+s][j+t]
            for i in xrange(self.output_dim[1]):
                for j in xrange(self.output_dim[0]):
                    delta_biases[k] += delta[k][i][j]
        self.weights -= delta_weights
        self.biases  -= delta_biases
        return next_delta

    def getOutput(self):
        return self.before_activation

    def getWeights(self):
        return self.weights

    def getBiases(self):
        return self.biases

