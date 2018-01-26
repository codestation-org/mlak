from copy import deepcopy
import numpy as np
import scipy.optimize as optimize

from mlak import LinearRegression as linReg

def gradient_descent( X, y, theta, alpha, iterations, lambda_val ):
	m = len( y )
	J_history = np.zeros( iterations )
	for i in range( iterations ):
		cost, grad = linReg.compute_cost_grad( X, y, theta, lambda_val )
		theta = theta - alpha * grad
		J_history[i] = cost
	return theta, J_history

def gradient_descent_fminCG( X, y, theta, iterations, lambda_val, **kwArgs ):
	#print( "initial theta = {}".format( theta.flatten() ) )
	return optimize.fmin_cg( linReg.compute_cost_fminCG, theta, fprime = linReg.compute_grad_fminCG, args = ( X, y, lambda_val ), maxiter = iterations, **kwArgs )

