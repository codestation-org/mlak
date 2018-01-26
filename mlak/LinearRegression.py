from copy import deepcopy
import numpy as np

import mlak.LinearAlgebra as la
import mlak.ModelAnalyzer as mo

import mlak.MathTools as mt
import mlak.ModelAnalyzer as ma
import mlak.OptimizationAlgorithms as oa


# Compute cost value for given theta.
def compute_cost( X, y, theta, lambda_val ):
	m = len( y )
	theta = deepcopy( theta )
	la.columnize( theta )
	hr = np.dot( X, theta )
	sqrErr = ( hr - y ) ** 2
	cost = np.sum( sqrErr ) / ( 2 * m )

	costReg = 0
	if lambda_val > 0:
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
	def __initial_theta( shaper, model = None, **kwArgs ):
		return model if model is not None else np.zeros( shaper.feature_count() + 1 )

	def type( self_ ):
		return ma.SolverType.VALUE_PREDICTOR

	def train( self_, X, y, Lambda = 0, iterations = 50, **kwArgs ):
		shaper = mo.DataShaper( X, y, **kwArgs )
		theta = LinearRegressionSolver.__initial_theta( shaper, **kwArgs )

		X = shaper.conform( X )

		theta = oa.gradient_descent_fminCG( X, y, theta, iterations, Lambda, disp = False )

		return ma.Solution( model = theta, shaper = shaper )

	def verify( self_, solution, X, y ):
		X = solution.shaper().conform( X )
		return compute_cost( X, y, solution.model(), 0 )

	def predict( self_, solution, X ):
		X = solution.shaper().conform( X )
		return np.dot( X, solution.model() )
