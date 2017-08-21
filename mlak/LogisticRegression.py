from copy import deepcopy
import LinearAlgebra as la
import MathTools as mt
import ModelAnalyzer as ma
import numpy as np
import scipy.optimize as optimize
import ModelAnalyzer as mo

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

	def train( self_, X, y, **kwArgs ):
		Lambda = kwArgs.get( "Lambda" )
		self_._shaper = mo.DataShaper( X, y, **kwArgs )
		thetas = []
		X = self_._shaper.conform( X )
		y = self_._shaper.map_labels( y )
		for c in range( self_._shaper.class_count() ):
			theta = optimize.fmin_cg(
				compute_cost,
				self_._shaper.initial_theta(), fprime = compute_grad,
				args = ( X, ( y == c ), Lambda ),
				maxiter = self_._iterations,
				disp = False
			)
			thetas.append( theta )
		return ma.Solution( theta = np.array( thetas ), shaper = self_._shaper )

	def verify( self_, solution, X, y ):
		X = self_._shaper.conform( X )
		yp = predict_one_vs_all( X, solution.theta() )
		accuracy = np.mean( 1.0 * ( y.flatten() == self_._shaper.labels( yp ) ) )
		return 1 - accuracy

