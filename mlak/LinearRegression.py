from copy import deepcopy
import numpy as np

import LinearAlgebra as la
import ModelAnalyzer as mo

import MathTools as mt
import ModelAnalyzer as ma
import numpy as np
import OptimizationAlgorithms as oa


# Compute cost value for given theta.
def compute_cost( X, y, theta, lambda_val ):
	m = len( y )
	theta = deepcopy( theta )
	la.columnize( theta )
	hr = np.dot( X, theta )
	sqrErr = ( hr - y ) ** 2
	cost = np.sum( sqrErr ) / ( 2 * m )

	costReg = lambda_val / (2 * m) * (np.sum(theta * theta)-theta[0]*theta[0])
	costReg = costReg[0]

	return cost + costReg

# Compute gradient \delta for given \theta for optimization algorithms.
def compute_grad( X, y, theta, lambda_val ):
	m = len( y )
	theta = deepcopy( theta )
	la.columnize( theta )
	grad = np.dot( ( np.dot( X, theta ) - y ).T, X ).T / m + lambda_val / m * theta
	grad[0] = grad[0] - lambda_val / m * theta[0]

	return grad

# Compute both `cost` and `gradient` for given \theta and given equation system.
def compute_cost_grad( X, y, theta, lambda_val ):
	return compute_cost( X, y, theta, lambda_val ), compute_grad( X, y, theta, lambda_val )

def compute_cost_fminCG( theta, *args ):
	return compute_cost( args[0], args[1], theta, args[2] )

def compute_grad_fminCG( theta, *args ):
	return compute_grad( args[0], args[1], theta, args[2] ).flatten()

class LinearRegressionSolver:
	def train( self_, X, y, **kwArgs ):
		iters = kwArgs.get( "iters", 50 )
		Lambda = kwArgs.get( "Lambda", 0 )
		shaper = mo.DataShaper( X, y, **kwArgs )
		initial_theta = shaper.initial_theta()

		X = shaper.conform( X )

		theta = oa.gradient_descent_fminCG(X, y, initial_theta, iters, Lambda)

		return ma.Solution( theta = theta, shaper = shaper )

	def verify( self_, solution, X, y ):
		y_pred = self_.predict(solution, X)

		X = solution.shaper().conform( X )
		m = np.size(y)

		sqErr = np.sum((y_pred-y)**2) / m
		return sqErr

	def predict( self_, solution, X ):
		X = solution.shaper().conform( X )
		return np.dot(X, solution.theta())
