from copy import deepcopy
import LinearAlgebra as la
import MathTools as mt
import ModelAnalyzer as ma
import numpy as np
import scipy.optimize as optimize

def compute_cost( theta, *args ):
	X, y, regularizationParam = args
	theta = deepcopy( theta )
	la.columnize( theta )

	hr = mt.sigmoid_v( np.dot( X, theta ) )

	m = len( y )
	err = -( np.dot( y.T, mt.log_v( hr ) ) + np.dot( ( 1 - y.T ), mt.log_v( 1 - hr ) ) )

	theta[ 0 ] = 0
	r = regularizationParam * np.sum( theta ** 2 ) / ( m * 2 )
	cost = err / m + r
	return cost

def compute_grad( theta, *args ):
	X, y, regularizationParam = args
	theta = deepcopy( theta )
	la.columnize( theta )
	hr = mt.sigmoid_v( np.dot( X, theta ) )
	theta[ 0 ] = 0
	m = len( y )
	grad = ( np.dot( X.T, ( hr - y ) ) + theta * regularizationParam ) / m
#	print( "grad = {}".format( grad.flatten() ) )
	return grad.flatten()

def predict( X, theta ):
	p = mt.sigmoid( np.dot( X, theta ) ) >= 0.5
	return p

def predict_one_vs_all( X, all_theta ):
	return np.argmax( np.dot( X, all_theta.T ), axis = 1 )

class LogisticRegressionSolver:
	def __init__( self_, **kwArgs ):
		self_._iterations = kwArgs.get( "iters", 50 )

	def preprocess( self_, X, y ):
		self_._classes, yReverseIndex = np.unique( y, return_inverse = True )
		la.columnize( yReverseIndex )
		self_._classCount = len( self_._classes )
		return ( X, yReverseIndex )

	def train( self_, X, y, Lambda, **kwArgs ):
		n = np.size( X, axis = 1 )
		thetas = np.zeros( ( self_._classCount, n ) )
		for c in range( self_._classCount ):
			thetas[c] = optimize.fmin_cg(
				compute_cost,
				thetas[c], fprime = compute_grad,
				args = ( X, ( y == c ), Lambda ),
				maxiter = self_._iterations,
				disp = False
			)
		return ma.Solution( theta = thetas )

	def verify( self_, solution, X, y ):
		yp = predict_one_vs_all( X, solution.theta() )
		accuracy = np.mean( 1.0 * ( y.flatten() == yp ) )
		return 1 - accuracy

