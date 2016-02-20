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
import maxpooling_layer
import input_layer

import mnist_loader_with_pickle as loader

if __name__ == '__main__':
    # Load training data and test data
    training_data, validation_data, test_data = \
    loader.load_data_wrapper()

    # Create layers
    input_layer = input_layer.InputLayer(784)
    input_layer.setData(training_data[:1000])
    conv1 = convolution_layer.ConvolutionLayer( \
        (28, 28, 1), (4, 4, 6), (24, 24, 6), logistic.LogisticFunction(), 0.1)
    pool1 = maxpooling_layer.MaxPoolingLayer( \
        (24, 24, 6), (2, 2), (12, 12, 6))
    conv2 = convolution_layer.ConvolutionLayer( \
        (12, 12, 6), (4, 4, 14), (8, 8, 14), logistic.LogisticFunction(), 0.1)
    pool2 = maxpooling_layer.MaxPoolingLayer( \
        (8, 8, 14), (2, 2), (4, 4, 14))
    full1 = fullyconnected_layer.FullyConnectedLayer( \
        224, 100, logistic.LogisticFunction(), 0.1, 1.0, 0.0)
    full2 = fullyconnected_layer.FullyConnectedLayer( \
        100, 10, logistic.LogisticFunction(), 0.1, 1.0, 0.0)

    # Create network
    net = network.Network()
    net.appendLayer(input_layer)
    net.appendLayer(conv1)
    net.appendLayer(pool1)
    net.appendLayer(conv2)
    net.appendLayer(pool2)
    net.appendLayer(full1)
    net.appendLayer(full2)

    # Training start
    epoch = 50
    import random
    random.shuffle(test_data)
    net.train(epoch, test_data=test_data[:100])

