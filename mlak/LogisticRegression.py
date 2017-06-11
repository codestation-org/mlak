from copy import deepcopy
import LinearAlgebra as la
import MathTools as mt
import numpy as np

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

