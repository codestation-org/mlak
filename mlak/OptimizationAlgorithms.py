from copy import deepcopy
import numpy as np
import scipy.optimize as optimize

from collections import namedtuple

Algorithm = namedtuple( "Algorithm", "cost grad" )

def gradient_descent( algo, X, y, theta, alpha, iterations, lambda_val ):
	m = len( y )
	J_history = np.zeros( iterations )
	for i in range( iterations ):
		cost = algo.cost( X, y, theta, lambda_val )
		grad = algo.grad( X, y, theta, lambda_val ).flatten()
		theta = theta - alpha * grad
		J_history[i] = cost
	return theta, J_history

def compute_cost_fminCG( cost ):
	return lambda theta, *args: cost( args[0], args[1], theta, args[2] )

def compute_grad_fminCG( grad ):
	return lambda theta, *args: grad( args[0], args[1], theta, args[2] ).flatten()

def gradient_descent_fminCG( algo, X, y, theta, iterations, lambda_val, **kwArgs ):
	#print( "initial theta = {}".format( theta.flatten() ) )
	return optimize.fmin_cg(
		compute_cost_fminCG( algo.cost ),
		theta,
		fprime = compute_grad_fminCG( algo.grad ),
		args = ( X, y, lambda_val ),
		maxiter = iterations,
		**kwArgs
	)

