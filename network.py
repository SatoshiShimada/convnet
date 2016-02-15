# coding: utf-8

# Network
# It's have any layers
# Execute training
#
# Date: Feb 15, 2016
# Author: Satoshi SHIMADA

import random
import numpy

class Network(object):
    def __init__(self):
        self.network = []

    def appendLayer(self, layer):
        self.network.append(layer)

    def train(self, epochs, test_data=False):
        training_data = self.network[0].getOutput()
        for ep in xrange(epochs):
            random.shuffle(training_data)
            for d, l in training_data:
                # feed forward
                Z = [d]
                for net in self.network[1:]:
                    Z.append(net.feedForward(Z[-1], train=True))
                # calc error
                delta = [numpy.zeros((n.output_size, 1)) for n in self.network[1:]]
                delta[-1] = (Z[-1] - l) * self.network[-1].activation._diff(self.network[-1].getOutput())
                # back propagation
                for i in xrange(len(self.network[1:])):
                    if i == len(self.network[1:]) - 1:
                        self.network[-i-1].backPropagation(Z[-i-2], delta[-i-1])
                    else:
                        delta[-i-2] = self.network[-i-1].backPropagation(Z[-i-2], delta[-i-1], prev_out=self.network[-i-2].getOutput())

            if test_data:
                acc = 0
                count = 0
                for d, l in test_data:
                    result = d
                    for net in self.network[1:]:
                        result = net.feedForward(result)
                    classifly = numpy.argmax(result)
                    if classifly == l:
                        acc += 1
                    count += 1
                print 'Epochs: {0}'.format(ep)
                print '{0} / {1} [{2}%]'.format(acc, count, float(acc) / count * 100.)

