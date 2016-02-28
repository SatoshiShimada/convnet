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
        out = numpy.zeros((self.output_dim[2], self.output_dim[1], self.output_dim[0]))
        for k in xrange(self.filter_dim[2]):
            for i in xrange(self.output_dim[1]):
                for j in xrange(self.output_dim[0]):
                    for c in xrange(self.input_dim[2]):
                        for s in xrange(self.filter_dim[1]):
                            for t in xrange(self.filter_dim[0]):
                                out[k][i][j] += self.weights[k][c][s][t] * inputs[c][i+s][j+t]
                    out[k][i][j] += self.biases[k][0]
        self.before_activation = out.reshape((self.output_size, 1))
        act = self.activation._apply(self.before_activation)
        return act

    def backPropagation(self, inputs, delta, prev_out=None):
        inputs = inputs.reshape((self.input_dim[2], self.input_dim[1], self.input_dim[0]))
        delta = delta.reshape((self.output_dim[2], self.output_dim[1], self.output_dim[0]))
        before_activation = self.before_activation.reshape((self.output_dim[2], self.output_dim[1], self.output_dim[0]))

        # calculation delta for next layer
        next_delta = numpy.zeros((self.input_dim[2], self.input_dim[1], self.input_dim[0]))
        for c in xrange(self.input_dim[2]):
            for i in xrange(self.input_dim[1]):
                for j in xrange(self.input_dim[0]):
                    for k in xrange(self.filter_dim[2]):
                        for s in xrange(self.filter_dim[1]):
                            for t in xrange(self.filter_dim[0]):
                                #index1 = i - (self.filter_dim[1] - 1) - s
                                #index2 = j - (self.filter_dim[0] - 1) - t
                                index1 = i - (self.filter_dim[1]) - s
                                index2 = j - (self.filter_dim[0]) - t
                                if index1 < 0 or index2 < 0:
                                    continue
                                next_delta[c][i][j] += delta[k][index1][index2] * self.activation._diff(before_activation[k][index1][index2] + self.biases[k]) * self.weights[k][c][s][t]

        delta_weights = numpy.zeros(self.weights.shape)
        delta_biases  = numpy.zeros(self.biases.shape)
        # calculation gradient
        for k in xrange(self.filter_dim[2]):
            for i in xrange(self.output_dim[1]):
                for j in xrange(self.output_dim[0]):
                    d = delta[k][i][j] * self.activation._diff(before_activation[k][i][j] + self.biases[k])
                    delta_biases[k] += d
                    for c in xrange(self.input_dim[2]):
                        for s in xrange(self.filter_dim[1]):
                            for t in xrange(self.filter_dim[0]):
                                delta_weights[k][c][s][t] += d * inputs[c][i+s][j+t]
        # update parameters
        for k in xrange(self.filter_dim[2]):
            self.biases[k][0] -= self.learning_rate * delta_biases[k]
            for c in xrange(self.input_dim[2]):
                for s in xrange(self.filter_dim[1]):
                    for t in xrange(self.filter_dim[0]):
                        self.weights[k][c][s][t] -= self.learning_rate * delta_weights[k][c][s][t]
        return next_delta.reshape((-1, 1))

    def getOutput(self):
        return self.before_activation

    def getWeights(self):
        return self.weights

    def getBiases(self):
        return self.biases

