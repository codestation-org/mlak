#! /usr/bin/python3

import numpy as np
from pypeg2 import *
import functools

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

class Text( Symbol ):
	regex = re.compile( "[\\w\\d\\=]+" )

def params( required_, optional_ = () ):
	typeRule = ( required_[0], )
	for t in required_[1:]:
		typeRule = *typeRule, ",", t
	for t in optional_:
		typeRule = *typeRule, optional( ( ",", t ) )
	return "(", attr( "values", typeRule ), ")"

def coerce( layer ):
	if type( layer.args ) == tuple:
		d = functools.reduce( ( lambda x,y: x + y ), layer.args )
		if type( layer.values ) == list:
			for i in range( len( layer.values ) ):
				if d[i] == Integer:
					layer.values[i] = int( layer.values[i] )
				elif d[i] == Float:
					layer.values[i] = float( layer.values[i] )
				else:
					s = str( layer.values[i] )
					if s.find( "=" ) >= 0:
						if not hasattr( layer, "kwArgs" ):
							layer.kwArgs = {}
						s = s.split( "=", 1 )
						layer.kwArgs[s[0]] = s[1]
					else:
						layer.values[i] = s
		else:
			if d[0] == Integer:
				layer.values = int( layer.values )
			elif d[0] == Float:
				layer.values = float( layer.values )
			else:
				layer.values = str( layer.values )
	else:
		if layer.args == Integer:
			layer.value = int( layer )
		elif layer.args == Float:
			layer.value = float( layer )
		else:
			layer.value = layer.__class__.__name__
	return layer

class ConvolutionParser:
	args = ( Integer, Integer, Integer ), ( Text, )
	grammar = [ K( "Convolution" ), K( "C" ) ], params( *args )
	def make( self_, model, **kwArgs_ ):
		kwArgs = {}
		kwArgs.update( kwArgs_ )
		if hasattr( self_, "kwArgs" ):
			kwArgs.update( self_.kwArgs )
		model.add( Convolution2D( self_.values[0], ( self_.values[1], self_.values[2] ), **kwArgs ) )

class DropoutParser:
	args = ( Float, ),
	grammar = [ K( "Dropout" ), K( "D" ) ], params( *args )
	def make( self_, model, **kwArgs ):
		model.add( Dropout( float( self_.values ), **kwArgs ) )

class MaxPoolingParser:
	args = ( Integer, Integer ),
	grammar = [ K( "MaxPooling" ), K( "MP" ) ], params( *args )
	def make( self_, model, **kwArgs ):
		model.add( MaxPooling2D( pool_size = ( self_.values[0], self_.values[1] ), **kwArgs ) )

class FlattenParser:
	args = None
	grammar = [ K( "Flatten" ), K( "F" ) ]
	def make( self_, model, **kwArgs ):
		model.add( Flatten( **kwArgs ) )

class DenseParser:
	args = ( Integer, ), ( Text, )
	grammar = [ K( "Dense" ), K( "N" ) ], params( *args )
	def make( self_, model, **kwArgs ):
		model.add( Dense( self_.values[0], **kwArgs ) )

class SoftmaxParser:
	args = None
	grammar = [ K( "Softmax" ), K( "S" ) ]

class TopologyParser( List ):
	grammar = optional( csl( [
		ConvolutionParser,
		DenseParser,
		FlattenParser,
		SoftmaxParser,
		DropoutParser,
		MaxPoolingParser
	] ) )

class KerasSolver:
	def __prepare_model( shaper, **kwArgs ):
		sampleSize = int( sqrt( shaper.feature_count() ) )
		topology = kwArgs.get( "nnTopology", [] )
		topology = list( map( coerce, parse( topology, TopologyParser ) ) )
		model = Sequential()
		first = True
		for l in topology:
			if first:
				l.make( model, input_shape = ( sampleSize, sampleSize, 1 ) )
				first = False
			else:
				l.make( model )
		if first:
			model.add( Flatten( input_shape = ( sampleSize, sampleSize, 1 ) ) )
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
		model.fit( X, y, batch_size = 64, epochs = 20, verbose = 1 )
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

