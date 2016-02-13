# coding: utf-8

# Learning layer
#
# Date: Feb 13, 2016
# Author: Satoshi SHIMADA

import abc

class LearningLayer(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def getWeights(self):
        return

    @abc.abstractmethod
    def getBiases(self):
        return

