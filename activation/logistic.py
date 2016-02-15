# coding: utf-8

# Activation function
# logistic sigmoid function
#
# Date: Feb 13, 2016
# Author: Satoshi SHIMADA

import numpy
import activation

class LogisticFunction(activation.ActivationFunction):
    def __init__(self):
        self._apply = numpy.vectorize(self.apply)
        self._diff  = numpy.vectorize(self.diff)

    def apply(self, value):
        return 1. / (1. + numpy.exp(-value))

    def applyAfter(self, values):
        for i in xrange(len(values)):
            values[i] = self.apply(values[i])
        return values

    def diff(self, value):
        return self.apply(value) * (1. - self.apply(value))

