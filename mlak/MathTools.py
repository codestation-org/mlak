import numpy as np
from scipy.special import expit

def log( x ):
	epsilon = 0.1 ** 100
	x[x < epsilon] = epsilon
	return np.log( x )

# Sigmoid function.
def sigmoid( x ):
	return 1 / ( 1 + np.exp( -x ) )

# Vectorized sigmoid function.
sigmoid_v = expit

def sigmoid_gradient( z ):
	s = sigmoid_v( z )
	g = s * ( 1 - s )
	return g

