# coding: utf-8

# Activation function
# softmax function
#
# Date: Feb 14, 2016
# Author: Satoshi SHIMADA

import numpy
import activation

class LogisticFunction(activation.ActivationFunction):
    def __init__(self):
        pass

    def _apply(self, values)
        if values.shape != values.reshape((-1, 1)).shape:
            print 'Matrix size error'
            print 'in file softmax.py'
        class_count = values.shape[0]
        values - values.max() # don't overflow
        ret = values.copy()
        buf = 0.
        for i in xrange(class_count):
            buf +=  numpy.exp(values[i])
        self.input_sum = buf
        for i in xrange(class_count):
            ret[i] = self.apply(values[i])
        return ret

    def _diff(self, values):
        class_count = values.shape[0]
        pass

    def apply(self, value):
        return value / self.input_sum

    def applyAfter(self, values):
        for i in xrange(len(values)):
            values[i] = self.apply(values[i])
        return values

    def diff(self, value):
        return self.apply(value) * (1. - self.apply(value))

