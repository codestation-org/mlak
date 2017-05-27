from copy import deepcopy
import numpy as np

import LinearAlgebra as la

# Compute cost value for given theta.
def compute_cost( X, y, theta ):
	m = len( y )
	theta = deepcopy( theta )
	la.columnize( theta )
	hr = np.dot( X, theta )
	sqrErr = ( hr - y ) ** 2
	cost = np.sum( sqrErr ) / ( 2 * m )
	return cost

# Compute gradient \delta for given \theta for optimization algorithms.
def compute_grad( X, y, theta ):
	m = len( y )
	theta = deepcopy( theta )
	la.columnize( theta )
	grad = np.dot( ( np.dot( X, theta ) - y ).T, X ).T / m
	return grad

# Compute both `cost` and `gradient` for given \theta and given equation system.
def compute_cost_grad( X, y, theta ):
	return compute_cost( X, y, theta ), compute_grad( X, y, theta )

def compute_cost_fminCG( theta, *args ):
	return compute_cost( args[0], args[1], theta )

def compute_grad_fminCG( theta, *args ):
	return compute_grad( args[0], args[1], theta ).flatten()

