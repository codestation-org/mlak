#! /usr/bin/python3
"""
Created on Tue Nov 9 2017
author: Wojciech Peisert

Models interfaces
"""

from keras.models import load_model
from keras.models import Model
from keras.layers import Input

import mlak.FeatureTools as ft
import mlak.ModelAnalyzer as ma


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
# This class implements AbstractModel for Keras (sequential) model
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


#
# This class implements AbstractModel for Keras (sequential) model
#
class KerasSequentialModelFactory:

	def createModel(input_shape, layers_definitions, compile_options):
		module = __import__("keras")
		module_layers = getattr(module, "layers")
		inputs = Input(input_shape)
		x = inputs
		for layer_definition in layers_definitions:
			class_name = layer_definition.class_name
			config = layer_definition.config
			layer_class = getattr(module_layers, class_name)
			x = layer_class(**config)(x)
		predictions = x

		model = Model(inputs=inputs, outputs=predictions)
		model.compile(**compile_options)

		return model


class ModelRunner:
	def __init__(self, model):
		if not isinstance(model, AbstractModel):
			raise InterfaceNotImplemented(model.__class__.__name__, AbstractModel.__name__)
		self.model = model
		self.model.saveState()

	def train(self, X, y, **trainKwArgs):
		self.model.loadState()
		self.model.train(X, y, **trainKwArgs)

	def evaluate(X, y):
		return self.model.evaluate(X, y)


class ModelTester:

	DEFAULT_EPOCHS = 10  #should be as Keras	 has

	def __init__(self, model, X, y, **trainKwArgs):
		self.modelRunner = ModelRunner(model)
		self.X = X
		self.y = y
		self.trainKwArgs = trainKwArgs

	def getIterationsLearningCurve(self, steps=10):
		ds = ma.split_data(self.X, self.y, cvFraction = 0, testFraction = 0.3)
		X_tr = ds.trainSet.X
		y_tr = ds.trainSet.y
		X_tt = ds.testSet.X
		y_tt = ds.testSet.y

		runningTrainKwArgs = self.trainKwArgs
		epochs = runningTrainKwArgs.get('epochs', ModelTester.DEFAULT_EPOCHS)

		results_tr = []
		results_tt = []
		for step in range(steps):
			runningEpochs = int(epochs*(step+1)/steps)
			runningTrainKwArgs['epochs'] = runningEpochs
			self.modelRunner.train(X_tr, y_tr, **runningTrainKwArgs)

			result_tr = self.modelRunner(evaluate(X_tr, y_tr))
			result_tt = self.modelRunner(evaluate(X_tt, y_tt))
			results_tr.append(result_tr)
			results_tt.append(result_tt)

		return {'tr': results_tr, 'tt': results_tt}


	def getLearningCurve(self, steps=10, trials=50):
		count = np.size(self.X, axis=0)
		results_tr = []
		results_tt = []
		for step in range(steps):
			runningCount = int(count*(step+1)/steps)
			result_tr = np.zeros((1, 2))
			result_tt = np.zeros((1, 2))
			for trial in range(trials):

#				perm = <=== TUTAJ RANDOM PERM O MOCY = count
#				running_X <== TUTAJ WYBRANE X
#				running_y <== TUTAJ WYBRANE y

				ds = ma.split_data(running_X, running_y, cvFraction = 0, testFraction = 0.3)
				X_tr = ds.trainSet.X
				y_tr = ds.trainSet.y
				X_tt = ds.testSet.X
				y_tt = ds.testSet.y

				self.modelRunner.train(X_tr, y_tr, **runningTrainKwArgs)
				trial_result_tr = self.modelRunner(evaluate(X_tr, y_tr))
				trial_result_tt = self.modelRunner(evaluate(X_tt, y_tt))

				result_tr += trial_result_tr
				result_tt += trial_result_tt

			result_tr /= trials
			result_tt += trials

			results_tr.append(result_tr)
			results_tt.append(result_tt)

		return {'tr': results_tr, 'tt': results_tt}


class MultiModelTester:
	pass


"""
1. Analiza czy jest overfitting (to zawsze)
2. Analiza krzywej uczenia dla konkretnego modelu (co daje zwiększanie iteracji)
3. Analiza krzywej uczenia dla konkretnego modelu (co daje zwiększanie zbioru treningowego)
3. Analiza krzywej uczenia dla konkretnego modelu (co daje zwiększanie batch size)
3. Analiza przy jednej warstwie ukrytej, co daje zwiększanie jej rozmiaru
4. Analiza co daje wielkość regularyzacji (jedna warstwa ukryta, kernel_regularizer)
5.
"""

#given:
#X, y
#For given parameters: param1=val1, param2=val2, ...
#for given optimisation parameters: param1 = [val1, val2, ...], param2 = [val1, val2, ...], ...
