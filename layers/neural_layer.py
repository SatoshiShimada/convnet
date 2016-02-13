# coding: utf-8

# Neural layer
# not only need learning
#
# Date: Feb 13, 2016
# Author: Satoshi SHIMADA

import abc

class NeuralLayer(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def feedForward(self, inputs):
        pass

    @abc.abstractmethod
    def backPropagation(self, inputs, delta):
        pass
