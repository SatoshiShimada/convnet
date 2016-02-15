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
        self.weights = numpy.random.randn(output_size, input_size)
        self.biases  = numpy.zeros((output_size, 1))
        self.filters = numpy.random.randn(filter_size[2], input_size[2], filter_size[1], filter_size[0])
        # width, height, channels
        self.input_size    = input_size
        # width, height, channels
        self.output_size   = output_size
        # width, height
        self.filter_size   = filter_size
        self.activation    = activation
        self.learning_rate = learning_rate

    def feedForward(self, inputs):
        pass

    def backPropagation(self, inputs, delta, prev_out=None):
        pass

    def getOutput(self):
        return self.before_activation

    def getWeights(self):
        return self.weights

    def getBiases(self):
        return self.biases

