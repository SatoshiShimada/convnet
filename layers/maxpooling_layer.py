# coding: utf-8

# Learning layer
# Max-Pooling layer
#
# Date: Feb 20, 2016
# Author: Satoshi SHIMADA

import numpy

import neural_layer

class MaxPoolingLayer \
    (neural_layer.NeuralLayer):
    def __init__(self, input_dim, kernel_dim, output_dim, stride=1):
        # width, height, channels
        self.input_dim    = input_dim
        self.input_size   = input_dim[0] * input_dim[1] * input_dim[2]
        # width, height, channels
        self.output_dim   = output_dim
        self.output_size  = output_dim[0] * output_dim[1] * output_dim[2]
        # width, height
        self.kernel_dim   = kernel_dim
        self.stride       = stride

    def feedForward(self, inputs, train=True):
        inputs = inputs.reshape((self.input_dim[2], self.input_dim[1], self.input_dim[0]))
        self.prev_input = inputs
        out = numpy.zeros((self.output_dim[2], self.output_dim[1], self.output_dim[0]))
        for c in xrange(self.output_dim[2]):
            for i in xrange(self.output_dim[1]):
                for j in xrange(self.output_dim[0]):
                    buf = []
                    for s in xrange(self.kernel_dim[1]):
                        for t in xrange(self.kernel_dim[0]):
                            buf.append(inputs[c][self.kernel_dim[1] * i+s][self.kernel_dim[0] * j+t])
                    out[c][i][j] = max(buf)
        self.before_activation = out.reshape((self.output_size, 1))
        return self.before_activation

    def backPropagation(self, inputs, delta, prev_out=None):
        inputs = inputs.reshape((self.input_dim[2], self.input_dim[1], self.input_dim[0]))
        out = self.before_activation.reshape((self.output_dim[2], self.output_dim[1], self.output_dim[0]))
        # calculation delta for next layer
        delta = delta.reshape((self.output_dim[2], self.output_dim[1], self.output_dim[0]))
        next_delta = numpy.zeros((self.input_dim[2], self.input_dim[1], self.input_dim[0]))
        for c in xrange(self.output_dim[2]):
            for i in xrange(self.output_dim[1]):
                for j in xrange(self.output_dim[0]):
                    for s in xrange(self.kernel_dim[1]):
                        for t in xrange(self.kernel_dim[0]):
                            if inputs[c][self.kernel_dim[1] * i+s][self.kernel_dim[0] * j+t] == out[c][i][j]:
                                next_delta[c][self.kernel_dim[1] * i+s][self.kernel_dim[0] * j+t] = delta[c][i][j]
        return next_delta.reshape((-1, 1))

    def getOutput(self):
        return self.before_activation

