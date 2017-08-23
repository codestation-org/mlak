import numpy as np
from collections import namedtuple
import OptimizationAlgorithms as oa
import LinearAlgebra as la
import FeatureTools as ft
import inspect
from itertools import product
import math

Observations = namedtuple( "Observations", "X y" )
DataSet = namedtuple( "DataSet", "trainSet crossValidationSet testSet" )
OptimizationResult = namedtuple( "OptimizationResult", "solution parameters failureRateTest" )

class DataShaper:
	def __init__( self_, X, y, **kwArgs ):
		self_._functions = kwArgs.get( "functions", [] )
		if self_._functions is None:
			self_._functions = []
		self_._featureCount = np.size( X, axis = 1 ) + len( self_._functions )
		self_._classes = None

		X = ft.add_features( X, self_._functions )
		self_._mu, self_._sigma = ft.find_normalization_params( X )

	def map_labels( self_, y ):
		self_._classes, y = np.unique( y, return_inverse = True )
		la.columnize( y )
		return y

	def conform( self_, X, **kwArgs ):
		addOnes = kwArgs.get( "addOnes", True )
		X = ft.add_features( X, self_._functions )
		X = ft.normalize_features( X, self_._mu, self_._sigma )
		if addOnes:
			X = la.add_ones_column( X )
		return X

	def labels( self_, y ):
		return self_._classes[ y ]

	def mu( self_ ):
		return self_._mu

	def sigma( self_ ):
		return self_._sigma

	def class_count( self_ ):
		assert self_._classes is not None, "Call shaper.map_labels( y ) first!"
		return len( self_._classes )

	def feature_count( self_ ):
		return self_._featureCount

	def __functions( self_ ):
		return ", ".join( map( inspect.getsource, self_._functions ) )

	def __repr__( self_ ):
		return "Shaper( mu = {}, sigma = {}, classes = {}, functions = [{}] )".format(
			self_._mu, self_._sigma, self_._classes, self_.__functions()
		)

class Solution:
	def __init__( self_, **kwArgs ):
		self_._theta = kwArgs.get( "theta", 0 )
		self_._shaper =  kwArgs.get( "shaper", None )
	def set( self_, **kwArgs ):
		if "theta" in kwArgs:
			self_._theta = kwArgs.get( "theta" )
		if "shaper" in kwArgs:
			self_._mu = kwArgs.get( "shaper" )
	def theta( self_ ):
		return self_._theta
	def shaper( self_ ):
		return self_._shaper
	def __repr__( self_ ):
		return "Solution( theta = {}, shaper = {} )".format( self_._theta, self_._shaper )

# Split data to train set, cross validation set and test set.
# *Fraction tells how much of the data shall go into given set.
def split_data( X, y, **kwArgs ):
	cvFraction = kwArgs.get( "cvFraction", 0.2 )
	testFraction = kwArgs.get( "testFraction", 0.2 )
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

def find_solution( solver, X, y, **kwArgs ):
	print( ">>> Looking for a solution..." )
	showFailureRateTrain = kwArgs.get( "showFailureRateTrain", False )
	optimizationParams = kwArgs.get( "optimizationParams", { "dummy" : [ 0 ] } )
	names = []
	values = []
	for k, v in optimizationParams.items():
		if type( v ) == list:
			names.append( k )
			if len( v ) == 0:
				v.append( None )
			values.append( v )
		print( "{}: {}".format( k, v ) )

	dataSet = split_data( X, y, **kwArgs )

	failureRate = math.inf
	solution = None
	optimizationParam = None
	for p in product( *values ):
		op = {}
		for i in range( len( names ) ):
			op[names[i]] = p[i]
		print( "testing solution for: {}".format( op ) )
		op.update( kwArgs )
		s = solver.train( dataSet.trainSet.X, dataSet.trainSet.y, **op )
		if showFailureRateTrain:
			fr = solver.verify( s, dataSet.trainSet.X, dataSet.trainSet.y )
			print( "failureRateTrain = {}".format( fr ) )
		fr = solver.verify( s, dataSet.crossValidationSet.X, dataSet.crossValidationSet.y )
		if fr < failureRate:
			failureRate = fr
			solution = s
			optimizationParam = op
		print( "failureRateCV = {}".format( fr ) )
	print( ">>> ... solution found." )
	return OptimizationResult( solution, optimizationParam, solver.verify( solution, dataSet.testSet.X, dataSet.testSet.y ) )

