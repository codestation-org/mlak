from copy import deepcopy
import LinearAlgebra as la
import MathTools as mt
import ModelAnalyzer as ma
import numpy as np
import scipy.optimize as optimize

def compute_cost( theta, *args ):
	X, y, Lambda = args
	theta = deepcopy( theta )
	la.columnize( theta )

	hr = mt.sigmoid_v( np.dot( X, theta ) )

	m = len( y )
	err = -( np.dot( y.T, mt.log_v( hr ) ) + np.dot( ( 1 - y.T ), mt.log_v( 1 - hr ) ) )

	theta[ 0 ] = 0
	r = Lambda * np.sum( theta ** 2 ) / ( m * 2 )
	cost = err / m + r
	return cost

def compute_grad( theta, *args ):
	X, y, Lambda = args
	theta = deepcopy( theta )
	la.columnize( theta )
	hr = mt.sigmoid_v( np.dot( X, theta ) )
	theta[ 0 ] = 0
	m = len( y )
	grad = ( np.dot( X.T, ( hr - y ) ) + theta * Lambda ) / m
#	print( "grad = {}".format( grad.flatten() ) )
	return grad.flatten()

def predict( X, theta ):
	p = mt.sigmoid( np.dot( X, theta ) ) >= 0.5
	return p

def predict_one_vs_all( X, all_theta ):
	return np.argmax( np.dot( X, all_theta.T ), axis = 1 )

class LogisticRegressionSolver:
	def __initial_theta( shaper ):
		return np.zeros( ( shaper.class_count(), shaper.feature_count() + 1 ) )

	def train( self_, X, y, **kwArgs ):
		shaper = ma.DataShaper( X, y, **kwArgs )
		iters = kwArgs.get( "iters", 50 )
		Lambda = kwArgs.get( "Lambda", 0 )
		y = shaper.map_labels( y )
		theta = kwArgs.get( "theta", LogisticRegressionSolver.__initial_theta( shaper ) )
		X = shaper.conform( X )
		for c in range( shaper.class_count() ):
			theta[c] = optimize.fmin_cg(
				compute_cost,
				theta[c], fprime = compute_grad,
				args = ( X, ( y == c ), Lambda ),
				maxiter = iters,
				disp = False
			)
		return ma.Solution( theta = theta, shaper = shaper )

	def verify( self_, solution, X, y ):
		X = solution.shaper().conform( X )
		yp = predict_one_vs_all( X, solution.theta() )
		accuracy = np.mean( 1.0 * ( y.flatten() == solution.shaper().labels( yp ) ) )
		return 1 - accuracy

	def predict( self_, solution, X ):
		X = solution.shaper().conform( X )
		yp = predict_one_vs_all( X, solution.theta() )
		return solution.shaper().labels( yp )

