from copy import deepcopy
import numpy as np

import mlak.LinearAlgebra as la
import mlak.ModelAnalyzer as ma

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

class LinearRegressionSolver:
	def __initial_theta( shaper, solution, **kwArgs ):
		return solution.model() if solution else np.zeros( shaper.feature_count() + 1 )

	def type( self_ ):
		return ma.SolverType.VALUE_PREDICTOR

	def train( self_, X, y, solution = None, Lambda = 0, iterations = 50, **kwArgs ):
		shaper = solution.shaper() if solution else ma.DataShaper( X, **kwArgs )
		theta = LinearRegressionSolver.__initial_theta( shaper, solution, **kwArgs )

		X = shaper.conform( X )

		theta = oa.gradient_descent_fminCG( oa.Algorithm( compute_cost, compute_grad ), X, y, theta, iterations, Lambda, disp = False )

		return ma.Solution( model = theta, shaper = shaper )

	def verify( self_, solution, X, y ):
		X = solution.shaper().conform( X )
		return compute_cost( X, y, solution.model(), 0 )

	def predict( self_, solution, X ):
		X = solution.shaper().conform( X )
		return np.dot( X, solution.model() )

