# coding: utf-8

# Activation function
# Rectified Linear function
#
# Date: Feb 14, 2016
# Author: Satoshi SHIMADA

import numpy
import activation

class RectifiedLinearFunction(activation.ActivationFunction):
    def __init__(self):
        self._apply = numpy.vectorize(self.apply)
        self._diff  = numpy.vectorize(self.diff)

    def apply(self, value):
        if value >= 0:
            return value
        else:
            return 0

    def applyAfter(self, values):
        for i in xrange(len(values)):
            values[i] = self.apply(values[i])
        return values

    def diff(self, value):
        if value >= 0:
            return 1
        else:
            return 0

