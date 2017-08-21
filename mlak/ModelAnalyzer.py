import numpy as np
from collections import namedtuple
import OptimizationAlgorithms as oa
import LinearAlgebra as la
import FeatureTools as ft

Observations = namedtuple( "Observations", "X y" )
DataSet = namedtuple( "DataSet", "trainSet crossValidationSet testSet" )

class DataShaper:
	def __init__( self_, X, y, **kwArgs ):
		self_._functions = kwArgs.get( "functions", [] )
		self_._featureCount = np.size( X, axis = 1 ) + len( self_._functions ) + 1

		X = ft.add_features( X, self_._functions )
		self_._mu, self_._sigma = ft.find_normalization_params( X )

	def map_labels( self_, y ):
		self_._classes, y = np.unique( y, return_inverse = True )
		la.columnize( y )
		return y

	def conform( self_, X ):
		X = ft.add_features( X, self_._functions )
		X = ft.normalize_features( X, self_._mu, self_._sigma )
		X = la.add_ones_column( X )
		return X

	def labels( self_, y ):
		return self_._classes[ y ]

	def mu( self_ ):
		return self_._mu

	def sigma( self_ ):
		return self_._sigma

	def class_count( self_ ):
		return len( self_._classes )

	def initial_theta( self_ ):
		return np.zeros( self_._featureCount )

	def __repr__( self_ ):
		return "Shaper( mu = {}, sigma = {}, classes = {} )".format( self_._mu, self_._sigma, self_._classes )

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
	lambdaRange = kwArgs.get( "lambdaRange", ( 0, 100 ) )
	lambdaQuality = kwArgs.get( "lambdaQuality", 1 )
	dataSet = split_data( X, y, **kwArgs )

	lo = lambdaRange[0]
	hi = lambdaRange[1]
	med = lo + ( hi - lo ) / 2
	failureRate = {}

	def train_and_verify( x ):
		print( "training with lambda: {}".format( x ), end = "" )
		solution = solver.train( dataSet.trainSet.X, dataSet.trainSet.y, Lambda = x, **kwArgs )
		failureRate = solver.verify( solution, dataSet.crossValidationSet.X, dataSet.crossValidationSet.y )
		print( ", failureRate = {}".format( failureRate ) )
		return ( solution, failureRate )

	def get_solution( x ):
		if x in failureRate:
			return failureRate.get( x )
		return failureRate.setdefault( x, train_and_verify( x ) )

	solution = None
	old = []
	while hi - lo > lambdaQuality:
		sLo = get_solution( lo )
		sMed = get_solution( med )
		sHi = get_solution( hi )
		minFrIdx = np.argmin( [sLo[1], sMed[1], sHi[1]] )
		if old == [lo, med, hi]: #unstable
			med, solution = ( lo, sLo[0] ) if minFrIdx == 0 else ( hi, sHi[0] )
			break
		old = [lo, med, hi]
		if minFrIdx == 0:
			hi = med
		elif minFrIdx == 2:
			lo = med
		else:
			loT = lo
			hiT = hi
			while hiT - loT > 0.01:
				loT = np.mean( [loT, med] )
				hiT = np.mean( [med, hiT] )
				sLoT = get_solution( loT )
				sHiT = get_solution( hiT )
				minFrIdx = np.argmin( [sLoT[1], sMed[1], sHiT[1]] )
				if minFrIdx == 0:
					hi = med
					break
				elif minFrIdx == 2:
					lo = med
					break
		med = np.mean( [lo, hi] )
		solution = sMed[0]
	return solution, solver.verify( solution, dataSet.testSet.X, dataSet.testSet.y ), med

