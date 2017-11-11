#! /usr/bin/python3
"""
Created on Tue Nov 9 2017
author: Wojciech Peisert

Models interfaces
"""

from keras.models import load_model
import FeatureTools as ft

class NotImplementedException(Exception):
	def __init__(self, className, methodName):
		self.value = "class: " + className + " method: " + methodName + " - not implemented"
	def __str__(self):
		return repr(self.value)

class InterfaceNotImplemented(Exception):
	def __init__(self, className, interfaceName):
		self.value = "class: " + className + " is not instance of " + interfaceName
	def __str__(self):
		return repr(self.value)

#
# each model should implement this interface, to be trainable
#
class AbstractModelInterface:

	#trains model; can be done incrementally
	#train(X, y, batch_size=32, epochs=10)
	def train(self, X, y, **kwArgs):
		raise NotImplementedException()

	def predict(self, X):
		raise NotImplementedException()

	def evaluate(self, X, y):
		raise NotImplementedException()

	def getState(self):
		raise NotImplementedException()

	def setState(self, stage):
		raise NotImplementedException()

	def load(self, name, path):
		raise NotImplementedException()

	def save(self, name, path):
		raise NotImplementedException()


#
# This class provides some facilities to AbstractModelInterface
#
class AbstractModel(AbstractModelInterface):

	def __init__(self, model):
		self.model = model
		self.savedStates = {}

	def loadState(self, stateName=""):
		if not self.savedStates.has_key(stateName):
			raise Exception("No previously saved state with name " + stateName)
		self.setState(self.savedStates[stateName])

	def saveState(self, stateName=""):
		self.savedStates[stateName] = self.getState()


"""
class FeaturedAbstractModel(AbstractModel):

	def __init__(self, model, add_features_functions = []):
		super().__init__(model)
		self.add_features_functions = add_features_functions

	def train(self, X, y, **kwArgs):
		super().train(self.__add_features(X), y, **kwArgs)

	def predict(self, X):
		super().predict(self.__add_features(X))

	def evaluate(self, X, y):
		super().evaluate(self.__add_features(X), y)

	def __add_features(self, X):
		return ft.add_features(X, self.add_features_functions)
"""

#
# This class implements AbstractModel for Keras (sequantial) model
#
class KerasAbstractModel(AbstractModel):

	def train(self, X, y, **kwArgs):
		self.model.fit(X, y, **kwArgs)

	def predict(self, X):
		return self.model.predict(X)

	def evaluate(self, X, y):
		return self.model.evaluate(X, y)

	def getState(self):
		return self.model.get_weights()

	def setState(self, state):
		self.model.set_weights(state)

	def load(self, name, path):
		self.model.save(path + name)

	def save(self, name, path):
		self.model = load_model(path + name)


class KerasModelFactory:
	#
	def createModel(**kwArgs):
		raise Exception("not implemented")



class ModelTester:
	def __init__(self, model):
		if not isinstance(model, AbstractModel):
			raise InterfaceNotImplemented(obj.__class__.__name__, AbstractModel.__name__)
		self.model = model
