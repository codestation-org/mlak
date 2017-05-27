from copy import deepcopy
import numpy as np
import scipy.optimize as optimize
import LinearRegression as linReg

def feature_normalize( X ):
	mu = np.mean( X, axis = 0 )
	sigma = np.std( X, axis = 0, ddof = 0 )
	X = ( X - mu ) / sigma
	return X, mu, sigma

def gradient_descent( X, y, theta, alpha, iters ):
	m = len( y )
	J_history = np.zeros( iters )
	for i in range( iters ):
		cost, grad = linReg.compute_cost_grad( X, y, theta )
		theta = theta - alpha * grad
		J_history[i] = cost
	return theta, J_history

def gradient_descent_fminCG( X, y, theta, iters ):
	print( "initial theta = {}".format( theta.flatten() ) )
	return optimize.fmin_cg( linReg.compute_cost_fminCG, theta, fprime = linReg.compute_grad_fminCG, args = ( X, y ), maxiter = iters )

