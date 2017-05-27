import numpy as np
from scipy.special import expit

def log_v( x ):
	with np.errstate( divide = "ignore", invalid = "ignore" ):
		return np.log( x )

# Sigmoid function.
def sigmoid( x ):
	return 1 / ( 1 + exp( -x ) )

# Vectorized sigmoid function.
sigmoid_v = expit

