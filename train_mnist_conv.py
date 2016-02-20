#!/usr/bin/python
# coding: utf-8

import sys
sys.path.append('activation')
sys.path.append('layers')

# Network
import network
# Activation functions
import logistic
import rectified_linear
# Layers
import fullyconnected_layer
import convolution_layer
import input_layer

import mnist_loader_with_pickle as loader

if __name__ == '__main__':
    # Load training data and test data
    training_data, validation_data, test_data = \
    loader.load_data_wrapper()

    # Create layers
    input_layer = input_layer.InputLayer(784)
    input_layer.setData(training_data[:1000])
    #full1 = fullyconnected_layer.FullyConnectedLayer( \
    #    784, 40, logistic.LogisticFunction(), 0.1, 0.9, 0.0)
    conv1 = convolution_layer.ConvolutionLayer( \
        (28, 28, 1), (4, 4, 2), (24, 24, 2), logistic.LogisticFunction(), 0.1)
    full2 = fullyconnected_layer.FullyConnectedLayer( \
        1152, 10, logistic.LogisticFunction(), 0.1, 1.0, 0.0)

    # Create network
    net = network.Network()
    net.appendLayer(input_layer)
    #net.appendLayer(full1)
    net.appendLayer(conv1)
    net.appendLayer(full2)

    # Training start
    epoch = 50
    net.train(epoch, test_data=test_data[:100])

