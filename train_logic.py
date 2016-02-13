#!/usr/bin/python
# coding: utf-8

import sys
sys.path.append('activation')
sys.path.append('layers')

import random

import numpy as np

import logistic_function
import fullyconnected_layer

if __name__ == '__main__':
    ####
    logic_and = [[[0, 0], 0], \
                 [[0, 1], 0], \
                 [[1, 0], 0], \
                 [[1, 1], 1]]
    data = []
    labels = []
    for x, y in logic_and:
        data.append(np.array(x).reshape(-1, 1))
        label = np.zeros((2, 1))
        label[y] = 1.
        labels.append(label)
    inputs = zip(data, labels)
    ####

    # Create layers
    full1 = fullyconnected_layer.FullyConnectedLayer( \
        2, 3, logistic_function.LogisticFunction(), 0.5, 0.7)
    full2 = fullyconnected_layer.FullyConnectedLayer( \
        3, 2, logistic_function.LogisticFunction(), 0.5, 0.9)
    
    for i in xrange(300):
        random.shuffle(inputs)
        for d, l in inputs:
            # feed forward
            ret1 = full1.feedForward(d)
            ret2 = full2.feedForward(ret1)
            # calc error
            delta2 = (ret2 - l) * full2.activation.diff(full2.getOutput())
            # back propagation
            delta1 = full2.backPropagation(ret1, delta2, prev_out=full1.getOutput())
            delta0 = full1.backPropagation(d, delta1)
    print '--- result ---'
    print full2.feedForward(full1.feedForward(data[0]))
    print full2.feedForward(full1.feedForward(data[1]))
    print full2.feedForward(full1.feedForward(data[2]))
    print full2.feedForward(full1.feedForward(data[3]))

