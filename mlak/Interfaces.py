#! /usr/bin/python3
"""
Created on Tue Nov 9 2017
author: Wojciech Peisert

Models interfaces
"""


class NotImplementedException(Exception):
    def __init__(self, className, methodName):
        self.value = "class: " + className + " method: " + methodName + " - not implemented!"
    def __str__(self):
        return repr(self.value)


class AbstractModel:

    def train(self, X, y, **kwArgs):
        raise NotImplementedException()

    def predict(self, X):
        raise NotImplementedException()


class AbstractSaveable:

    #loads model; name should be unique and valid filename; internal storage structure it up to model
    def load(path, name):
        raise NotImplementedException()

    #saves model; name should be unique and valid filename; internal storage structure it up to model
    def save(path, name):
        raise NotImplementedException()


class AbstractEvaluableModel(AbstractModel):

    # it returns valuation of trained model on given data
    #
    # this is to be thinked over, if it's a good interface, since valuation depend on model
    # maybe better solution is to create different interfaces
    #     depending on type of problem and result meaning
    def evaluate(self, X, y):
        raise NotImplementedException()
