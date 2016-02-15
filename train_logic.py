#!/usr/bin/python
# coding: utf-8

import sys
sys.path.append('activation')
sys.path.append('layers')

import random
import numpy as np

# Network
import network
# Activation functions
import logistic
# Layers
import fullyconnected_layer
import input_layer

if __name__ == '__main__':
    ####
    logic_and = [[[0, 0], 0], \
                 [[0, 1], 0], \
                 [[1, 0], 0], \
                 [[1, 1], 1]]
    data = []
    test_data = []
    labels = []
    labels_vec = []
    for x, y in logic_and:
        data.append(np.array(x).reshape(-1, 1))
        test_data.append(np.array(x).reshape(-1, 1))
        label = np.zeros((2, 1))
        label[y] = 1.
        labels_vec.append(label)
        labels.append(y)
    inputs = zip(data, labels_vec)
    test_inputs= zip(test_data, labels)
    ####

    # Create layers
    input_layer = input_layer.InputLayer(2)
    input_layer.setData(inputs)
    full1 = fullyconnected_layer.FullyConnectedLayer( \
        2, 3, logistic.LogisticFunction(), 0.2, 1.0, 0.1)
    full2 = fullyconnected_layer.FullyConnectedLayer( \
        3, 2, logistic.LogisticFunction(), 0.2, 1.0, 0.1)
    
    # Create network
    net = network.Network()
    net.appendLayer(input_layer)
    net.appendLayer(full1)
    net.appendLayer(full2)

    epoch = 1000
    net.train(epoch, test_data=test_inputs)

