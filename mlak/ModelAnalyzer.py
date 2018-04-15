import numpy as np
import math
from enum import Enum
from itertools import product
from collections import namedtuple
import inspect

import mlak.LinearAlgebra as la
import mlak.FeatureTools as ft
import mlak.Terminal as term
from mlak.Logger import Logger
from mlak.utils import stringify

Observations = namedtuple( "Observations", "X y" )
DataSet = namedtuple( "DataSet", "trainSet crossValidationSet testSet" )
OptimizationResult = namedtuple( "OptimizationResult", "solution parameters failureRateTest" )
SampleCountAnalyzis = namedtuple( "SampleCountAnalyzis", "sampleCount errorTrain errorCV" )
IterationCountAnalyzis = namedtuple( "IterationCountAnalyzis", "iterationCount errorTrain errorCV" )
AnalyzerResult = namedtuple( "AnalyzerResult", "sampleCountAnalyzis iterationCountAnalyzis" )

class SolverType( Enum ):
	VALUE_PREDICTOR = 1
	CLASSIFIER = 2

class DataShaper:
	def __init__( self_, X, functions = [], **kwArgs ):
		self_._functions = functions
		if self_._functions is None:
			self_._functions = []
		self_._featureCount = np.size( X, axis = 1 ) + len( self_._functions )
		self_._classesIdToLabel = None
		self_._classesLabelToId = None

		X = ft.add_features( X, self_._functions )
		self_._mu, self_._sigma = ft.find_normalization_params( X )

	def learn_labels( self_, y ):
		self_._classesIdToLabel, y = np.unique( y, return_inverse = True )
		self_._classesLabelToId = {}
		for id, label in enumerate( self_._classesIdToLabel ):
			self_._classesLabelToId[label] = id

	def map_labels( self_, y_ ):
		assert self_._classesIdToLabel is not None, "Call shaper.learn_labels( y ) first!"
		if type( y_ ) is np.ndarray:
			y_ = y_.flatten()
		y = np.zeros( len( y_ ), dtype = int )
		for idx, label in enumerate( y_ ):
			y[idx] = self_._classesLabelToId.get( label )
		la.columnize( y )
		return y

	def conform( self_, X, addOnes = True, **kwArgs ):
		X = ft.add_features( X, self_._functions )
		X = ft.normalize_features( X, self_._mu, self_._sigma )
		if addOnes:
			X = la.add_ones_column( X )
		return X

	def labels( self_, y ):
		return self_._classesIdToLabel[ y ]

	def mu( self_ ):
		return self_._mu

	def sigma( self_ ):
		return self_._sigma

	def class_count( self_ ):
		assert self_._classesIdToLabel is not None, "Call shaper.learn_labels( y ) first!"
		return len( self_._classesIdToLabel )

	def feature_count( self_ ):
		return self_._featureCount

	def is_classifier( self_ ):
		return True if self_._classesIdToLabel is not None else False

	def __functions( self_ ):
		return ", ".join( map( inspect.getsource, self_._functions ) )

	def __repr__( self_ ):
		return "Shaper( mu = {}, sigma = {}, classesIdToLabel = {}, functions = [{}] )".format(
			self_._mu, self_._sigma, self_._classesIdToLabel, self_.__functions()
		)

class Solution:
	def __init__( self_, **kwArgs ):
		self_._model = kwArgs.get( "model", 0 )
		self_._shaper =  kwArgs.get( "shaper", None )
	def set( self_, **kwArgs ):
		if "model" in kwArgs:
			self_._model = kwArgs.get( "model" )
		if "shaper" in kwArgs:
			self_._mu = kwArgs.get( "shaper" )
	def model( self_ ):
		return self_._model
	def shaper( self_ ):
		return self_._shaper
	def __repr__( self_ ):
		return "Solution( model = {}, shaper = {} )".format( self_._model, self_._shaper )

# Split data to train set, cross validation set and test set.
# *Fraction tells how much of the data shall go into given set.
def split_data( X, y, cvFraction = 0.2, testFraction = 0.2, **kwArgs ):
	assert cvFraction + testFraction < 1
	m = np.size( X, 0 )
	cvSize = int( m * cvFraction )
	testSize = int( m * testFraction )
	trainSize = m - ( cvSize + testSize )

	perm = np.random.permutation( m )

	XTrain = X[perm[ : trainSize]]
	yTrain = y[perm[ : trainSize]]

	XCV = X[perm[trainSize : trainSize + cvSize]]
	yCV = y[perm[trainSize : trainSize + cvSize]]

	XTest = X[perm[trainSize + cvSize : ]]
	yTest = y[perm[trainSize + cvSize : ]]

	return DataSet( Observations( XTrain, yTrain ), Observations( XCV, yCV ), Observations( XTest, yTest ) )

def find_solution(
	solver, X, y,
	showFailureRateTrain = False,
	optimizationParams = { "dummy" : [ 0 ] },
	log = { "log_file_name": "model-analyzer" },
	verbose = False,
	debug = False,
	files = [],
	**kwArgs
):
	print( ">>> Looking for a solution..." )
	for ign in [ "speech", "func", "data_set", "solution", "topology" ]:
		kwArgs.pop( ign, None )

	names = []
	values = []

	for k, v in optimizationParams.items():
		if type( v ) == list:
			names.append( k )
			if len( v ) == 0:
				v.append( None )
			values.append( v )
		print( "{}: {}".format( k, stringify( v ) ) )

	profiles = list( product( *values ) )

	profileCount = len( profiles )
	if profileCount == 1:
		kwArgs["testFraction"] = 0

	dataSet = split_data( X, y, **kwArgs )

	failureRate = math.inf
	solution = None
	optimizationParam = None
	for p in profiles:
		op = {}
		op.update( kwArgs )
		for i in range( len( names ) ):
			op[names[i]] = p[i]
		print( "testing solution for: {}   ".format( stringify( op ) ) )
		s = solver.train( dataSet.trainSet.X, dataSet.trainSet.y, verbose = verbose, debug = debug, **op )
		if showFailureRateTrain:
			fr = solver.verify( s, dataSet.trainSet.X, dataSet.trainSet.y )
			print( "failureRateTrain = {}         ".format( fr ) )
		fr = solver.verify( s, dataSet.crossValidationSet.X, dataSet.crossValidationSet.y )
		Logger.log(
			data = {
				"profile": op,
				"failureRateCV": fr,
			},
			files = files,
			**log
		)
		if fr < failureRate:
			failureRate = fr
			solution = s
			optimizationParam = op
		print( "failureRateCV = {}           ".format( fr ) )
	print( ">>> ... solution found." )
	return OptimizationResult(
		solution, optimizationParam,
		solver.verify( solution, dataSet.testSet.X, dataSet.testSet.y ) if profileCount != 1 else failureRate
	)

def analyze( solver, X, y, verbose = False, debug = False, **kwArgs ):
	print( ">>> Analyzing a model architecture..." )

# We must ensure that for classifier each train() invocation
# gets all classes available...
	startingSampleCount = 1
	if solver.type() == SolverType.CLASSIFIER:
		sortedIndex = np.argsort( y.flatten() )
		X = X[sortedIndex]
		y = y[sortedIndex]
		qq, classStartIndexes = np.unique( y, return_index = True )
		classCount = len( classStartIndexes )
		m = len( y )
		uniformRepresentationIndex = []
		for i in range( m ):
			classId = i % classCount
			uniformRepresentationIndex.append( classStartIndexes[classId] )
			classStartIndexes[classId] += 1
		uniformRepresentationIndex = np.array( uniformRepresentationIndex )
		X = X[uniformRepresentationIndex]
		y = y[uniformRepresentationIndex]
		startingSampleCount = classCount

	optimizationParams = kwArgs.pop( "optimizationParams", { "dummy" : [ 0 ] } )
	for ign in [ "speech", "func", "data_set", "solution", "topology" ]:
		kwArgs.pop( ign, None )
	optimizationParams.update( kwArgs )

	tries = optimizationParams.pop( "tries", None )
	if tries is None:
		tries = 10

	step = optimizationParams.pop( "step", None )
	if step is None:
		step = 1.5

	sampleIterations = optimizationParams.pop( "sample_iterations", None )
	if sampleIterations is None:
		sampleIterations = 50

	iterations = optimizationParams.pop( "iterations", None )
	if iterations is None:
		iterations = 50

	dataSet = split_data( X, y, testFraction = 0 )

	m = len( dataSet.trainSet.y )

	if verbose:
		steps = int( math.floor( math.log( m, step ) ) ) if step > 1 else m - 1
		p = term.Progress( steps, "Analyzing model (sample count): " )

	i = 0
	count = startingSampleCount
	sampleCount = []
	errorTrain = []
	errorCV = []
	while count < m:
		c = int( count )
		sampleCount.append( c )
		errorTrain.append( 0 )
		errorCV.append( 0 )
		for k in range( tries ):
			perm = np.random.permutation( m )[:c]
			Xt = dataSet.trainSet.X[perm]
			yt = dataSet.trainSet.y[perm]
			s = solver.train( Xt, yt, iterations = sampleIterations, verbose = verbose, debug = debug, **optimizationParams )
			errorTrain[i] += solver.verify( s, Xt, yt )
			errorCV[i] += solver.verify( s, dataSet.crossValidationSet.X, dataSet.crossValidationSet.y )
		if verbose:
			next( p )
		i += 1
		if step > 1:
			count *= step
		else:
			count += 1

	errorTrain = np.array( errorTrain )
	errorCV = np.array( errorCV )
	errorTrain /= tries
	errorCV /= tries

	sampleCountAnalyzis = SampleCountAnalyzis( sampleCount = sampleCount, errorTrain = errorTrain, errorCV = errorCV )

	if verbose:
		steps = int( math.floor( math.log( m, step ) ) ) if step > 1 else m - 1
		p = term.Progress( steps, "Analyzing model (iteration count): " )

	i = 0
	count = 1
	iterationCount = []
	errorTrain = []
	errorCV = []
	oldIterations = 0
	s = None
	while count < iterations:
		c = int( count )
		iterationCount.append( c )
		errorTrain.append( 0 )
		errorCV.append( 0 )
		s = solver.train(
			dataSet.trainSet.X,
			dataSet.trainSet.y,
			iterations = c - oldIterations,
			model = s.model() if s else None,
			verbose = verbose,
			debug = debug,
			**optimizationParams
		)
		oldIterations = c
		errorTrain[i] += solver.verify( s, dataSet.trainSet.X, dataSet.trainSet.y )
		errorCV[i] += solver.verify( s, dataSet.crossValidationSet.X, dataSet.crossValidationSet.y )
		if verbose:
			next( p )
		i += 1
		if step > 1:
			count *= step
		else:
			count += 1

	errorTrain = np.array( errorTrain )
	errorCV = np.array( errorCV )
	errorTrain /= tries
	errorCV /= tries

	iterationCountAnalyzis = IterationCountAnalyzis( iterationCount = iterationCount, errorTrain = errorTrain, errorCV = errorCV )

	return AnalyzerResult( sampleCountAnalyzis = sampleCountAnalyzis, iterationCountAnalyzis = iterationCountAnalyzis )

