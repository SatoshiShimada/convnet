# coding: utf-8

# Learning layer
# data input layer
#
# Date: Feb 14, 2016
# Author: Satoshi SHIMADA

import numpy

import neural_layer

class InputLayer(neural_layer.NeuralLayer):
    def __init__(self, output_size):
        self.output_size = output_size

    def setData(self, data):
        self.data = data

    def getOutput(self):
        return self.data

    def feedForward(self, inputs):
        return self.data

    def backPropagation(self, inputs, delta):
        print 'Input layer have not to learning'
        return
