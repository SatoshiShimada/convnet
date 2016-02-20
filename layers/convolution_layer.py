# coding: utf-8

# Learning layer
# Convolution layer
#
# Date: Feb 15, 2016
# Author: Satoshi SHIMADA

import numpy

import learning_layer
import neural_layer

class ConvolutionLayer \
    (neural_layer.NeuralLayer, learning_layer.LearningLayer):
    def __init__(self, input_size, filter_size, output_size, activation, learning_rate):
        self.weights = numpy.random.randn(filter_size[2], input_size[2], filter_size[1], filter_size[0])
        self.biases  = numpy.zeros((filter_size[2], 1))
        # width, height, channels
        self.input_size    = input_size
        # width, height, channels
        self.output_size   = output_size
        # width, height, count
        self.filter_size   = filter_size
        self.activation    = activation
        self.learning_rate = learning_rate

    def feedForward(self, inputs):
        inputs = inputs.reshape((self.input_size[1], self.input_size[0]))
        out = [numpy.zeros((self.output_size[1], self.output_size[1])) for k in xrange(filter_size[2])]
        for k in xrange(self.filter_size[2]):
            for i in xrange(self.output_size[1]):
                for j in xrange(self.output_size[0]):
                    for c in xrange(self.output_size[1]):
                        for s in xrange(self.filter_size[1]):
                            for t in xrange(self.filter_size[0]):
                                out[k][i][j] += self.weights[k][c][s][t] * inputs[c][i+s][j+t]
                    out[k][i][j] += self.biases[k][0]
        act = self.activation._apply(out)
        return act

    def backPropagation(self, inputs, delta, prev_out=None):
        pass

    def getOutput(self):
        return self.before_activation

    def getWeights(self):
        return self.weights

    def getBiases(self):
        return self.biases

