import numpy as np
from scipy.special import expit

def log_v( x ):
	epsilon = 0.1 ** 100
	x[x < epsilon] = epsilon
	return np.log( x )

# Sigmoid function.
def sigmoid( x ):
	return 1 / ( 1 + exp( -x ) )

# Vectorized sigmoid function.
sigmoid_v = expit

def sigmoidGradient(z):
	s = sigmoid(z)
	g = s * (1 - s)
	return g

# Vectorized sigmoidGradient function.
sigmoidGradient_v = expit
