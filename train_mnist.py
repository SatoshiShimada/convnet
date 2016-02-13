#!/usr/bin/python
# coding: utf-8

import sys
sys.path.append('activation')
sys.path.append('layers')

import random

import numpy as np

import logistic_function
import fullyconnected_layer

import mnist_loader_with_pickle as loader

if __name__ == '__main__':
    ####
    training_data, validation_data, test_data = \
    loader.load_data_wrapper()
    ####

    # Create layers
    full1 = fullyconnected_layer.FullyConnectedLayer( \
        784, 30, logistic_function.LogisticFunction(), 0.5, 0.7)
    full2 = fullyconnected_layer.FullyConnectedLayer( \
        30, 10, logistic_function.LogisticFunction(), 0.5, 0.9)
    
    epoch = 50
    for i in xrange(epoch):
        random.shuffle(training_data)
        for d, l in training_data:
            # feed forward
            ret1 = full1.feedForward(d)
            ret2 = full2.feedForward(ret1)
            # calc error
            delta2 = (ret2 - l) * full2.activation.diff(full2.getOutput())
            # back propagation
            delta1 = full2.backPropagation(ret1, delta2, prev_out=full1.getOutput())
            delta0 = full1.backPropagation(d, delta1)

        acc = 0
        count = 0
        for d, l in test_data:
            result = full2.feedForward(full1.feedForward(d))
            classifly = np.argmax(result)
            if classifly == l:
                acc += 1
            count += 1
        print '--- result ---'
        print 'Epochs: {0}'.format(i)
        print '{0} / {1}'.format(acc, count)

