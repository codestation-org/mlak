import numpy as np
from collections import namedtuple
import OptimizationAlgorithms as oa

Observations = namedtuple( "Observations", "X y" )
DataSet = namedtuple( "DataSet", "trainSet crossValidationSet testSet" )

class Solution:
	def __init__( self_, **kwArgs ):
		self_._theta = kwArgs.get( "theta", 0 )
		self_._mu =  kwArgs.get( "mu", 0 )
		self_._sigma =  kwArgs.get( "sigma", 1 )
	def set( self_, **kwArgs ):
		if "theta" in kwArgs:
			self_._theta = kwArgs.get( "theta" )
		if "mu" in kwArgs:
			self_._mu = kwArgs.get( "mu" )
		if "sigma" in kwArgs:
			self_._sigma =  kwArgs.get( "sigma" )
	def theta( self_ ):
		return self_._theta
	def mu( self_ ):
		return self_._mu
	def sigma( self_ ):
		return self_._sigma
	def __repr__( self_ ):
		return "Solution( theta = {}, mu = {}, sigma = {} )".format( self_._theta, self_._mu, self_._sigma )

# Split data to train set, cross validation set and test set.
# *Fraction tells how much of the data shall go into given set.
def split_data( X, y, cvFraction, testFraction ):
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

def preprocess_data( solver, X, y, **kwArgs ):
	cvFrequency = kwArgs.get( "cvFrequency", 0.2 )
	testFrequency = kwArgs.get( "testFrequency", 0.2 )

	X, y = solver.preprocess( X, y )

	XNormalized, mu, sigma = oa.feature_normalize( X )
	m = np.size( X, 0 )
	XWithOnes = np.c_[ np.ones( m ), XNormalized ]

	return ( mu, sigma, split_data( XWithOnes, y, cvFrequency, testFrequency ) )

def find_solution( solver, X, y, **kwArgs ):
	lambdaRange = kwArgs.get( "lambdaRange", ( 0, 100 ) )
	lambdaQuality = kwArgs.get( "lambdaQuality", 1 )

	mu, sigma, dataSet = preprocess_data( solver, X, y, **kwArgs )

	lo = lambdaRange[0]
	hi = lambdaRange[1]
	med = lo + ( hi - lo ) / 2
	failureRate = {}

	def train_and_verify( x ):
		print( "training with lambda: {}".format( x ), end = "" )
		solution = solver.train( dataSet.trainSet.X, dataSet.trainSet.y, x, **kwArgs )
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
	solution.set( mu = mu, sigma = sigma )
	return solution, solver.verify( solution, dataSet.testSet.X, dataSet.testSet.y ), med

