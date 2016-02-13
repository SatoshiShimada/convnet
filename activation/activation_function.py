# coding: utf-8

# Activation function
#
# Date: Feb 13, 2016
# Author: Satoshi SHIMADA

import abc

class ActivationFunction(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def apply(self, value):
        pass

    @abc.abstractmethod
    def applyAfter(self, values):
        pass

    @abc.abstractmethod
    def diff(self, value):
        pass

