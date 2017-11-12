#! /usr/bin/python3

import numpy as np
from pypeg2 import *

from math import sqrt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

import keras.models
import keras
import types
import tempfile

import ModelAnalyzer as ma

def make_keras_picklable():
	def __getstate__(self):
		model_str = ""
		with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
			keras.models.save_model(self, fd.name, overwrite=True)
			model_str = fd.read()
		d = { 'model_str': model_str }
		return d

	def __setstate__(self, state):
		with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
			fd.write(state['model_str'])
			fd.flush()
			model = keras.models.load_model(fd.name)
		self.__dict__ = model.__dict__

	cls = keras.models.Model
	cls.__getstate__ = __getstate__
	cls.__setstate__ = __setstate__

class Float( Symbol ):
	regex = re.compile( "\\d+\\.\\d+" )

class Integer( Symbol ):
	regex = re.compile( "\\d+" )

params = "(", attr( "values", csl( [ Float, Integer ] ) ), ")"

class ConvolutionParser:
	grammar = [ K( "Convolution" ), K( "C" ) ], params
	def make( self_, model, **kwArgs ):
		model.add( Convolution2D( int( self_.values[0] ), ( int( self_.values[1] ), int( self_.values[2] ) ), activation='relu', **kwArgs ) )

class DropoutParser:
	grammar = [ K( "Dropout" ), K( "D" ) ], params
	def make( self_, model, **kwArgs ):
		model.add( Dropout( float( self_.values[0] ), **kwArgs ) )

class MaxPoolingParser:
	grammar = [ K( "MaxPooling" ), K( "MP" ) ], params
	def make( self_, model, **kwArgs ):
		model.add( MaxPooling2D( pool_size = ( int( self_.values[0] ), int( self_.values[1] ) ), **kwArgs ) )

class FlattenParser:
	grammar = [ K( "Flatten" ), K( "F" ) ]
	def make( self_, model, **kwArgs ):
		model.add( Flatten( **kwArgs ) )

class SoftmaxParser:
	grammar = [ K( "Softmax" ), K( "S" ) ]

class DenseParser( Integer ):
	pass

class TopologyParser( List ):
	grammar = optional( csl( [
		ConvolutionParser,
		DropoutParser,
		MaxPoolingParser,
		DenseParser,
		FlattenParser,
		SoftmaxParser
	] ) )

class KerasSolver:
	def __prepare_model( shaper, **kwArgs ):
		sampleSize = int( sqrt( shaper.feature_count() ) )
		topology = kwArgs.get( "nnTopology", [] )
		topology = parse( topology, TopologyParser )
		model = Sequential()
		first = True
		for l in topology:
			if first:
				l.make( model, input_shape = ( sampleSize, sampleSize, 1 ) )
				first = False
			else:
				l.make( model )
		model.add(Dense(128, activation='relu'))
		model.add(Dropout(0.5))
		model.add( Dense( shaper.class_count(), activation = 'softmax' ) )
		model.compile( loss = 'categorical_crossentropy',
			optimizer = 'adam',
			metrics = ['accuracy']
		)

		return model

	def train( self_, X, y, **kwArgs ):
		shaper = ma.DataShaper( X, y, **kwArgs )
		sampleSize = int( sqrt( shaper.feature_count() ) )
		iters = kwArgs.get( "iters", 50 )
		Lambda = kwArgs.get( "Lambda", 0 )
		y = shaper.map_labels( y )
		y = keras.utils.to_categorical( y, num_classes = None )
		model = kwArgs.get( "model", KerasSolver.__prepare_model( shaper, **kwArgs ) )
		X = shaper.conform( X, addOnes = False )
		X = X.reshape( X.shape[0], sampleSize, sampleSize, 1 )
		X = X.astype('float32')
		model.fit( X, y, batch_size = 64, epochs = 20, verbose=1)
		return ma.Solution( model = model, shaper = shaper )

	def verify( self_, solution, X, y ):
		X = solution.shaper().conform( X, addOnes = False )
		sampleSize = int( sqrt( solution.shaper().feature_count() ) )
		X = X.reshape( X.shape[0], sampleSize, sampleSize, 1 )
		X = X.astype('float32')
		model = solution.model()
		yp = model.predict( X, verbose = 0 )
		yp = np.argmax( yp, axis = 1 )
		accuracy = np.mean( 1.0 * ( y.flatten() == solution.shaper().labels( yp ) ) )
		return 1 - accuracy

	def predict( self_, solution, X ):
		X = solution.shaper().conform( X, addOnes = False )
		sampleSize = int( sqrt( solution.shaper().feature_count() ) )
		X = X.reshape( X.shape[0], sampleSize, sampleSize, 1 )
		X = X.astype('float32')
		model = solution.model()
		yp = model.predict( X, verbose = 0 )
		yp = np.argmax( yp, axis = 1 )
		return solution.shaper().labels( yp )

