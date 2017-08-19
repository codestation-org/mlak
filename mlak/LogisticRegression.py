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

class LogisticRegressionSolver( ma.SolverBase ):
	def __init__( self_, X, y, **kwArgs ):
		self_._classes, yReverseIndex = np.unique( y, return_inverse = True )
		la.columnize( yReverseIndex )
		self_._classCount = len( self_._classes )
		super().__init__( X, yReverseIndex, **kwArgs )

	def solve( self_, Lambda, **kwArgs ):
		print( "trying lambda: {}".format( Lambda ), end = "" )
		n = np.size( self_._dataSet.trainSet.X, 1 )
		thetas = np.zeros( ( self_._classCount, n ) )
		for c in range( self_._classCount ):
			thetas[c] = optimize.fmin_cg(
				compute_cost,
				thetas[c], fprime = compute_grad,
				args = ( self_._dataSet.trainSet.X, ( self_._dataSet.trainSet.y == c ), Lambda ),
				maxiter = self_._iterations,
				disp = False
			)
		ypCV = predict_one_vs_all( self_._dataSet.crossValidationSet.X, thetas )
		accuracy = np.mean( 1.0 * ( self_._dataSet.crossValidationSet.y.flatten() == ypCV ) )
		print( ", accuracy = {}".format( accuracy ) )
		return ma.Solution( thetas, accuracy, Lambda )

	def test( self_, solution ):
		yp = predict_one_vs_all( self_._dataSet.testSet.X, solution.theta )
		accuracy = np.mean( 1.0 * ( self_._dataSet.testSet.y.flatten() == yp ) )
		return accuracy
