import numpy as np
from math import log, exp

log_v = np.vectorize( log )
exp_v = np.vectorize( exp )

def sigmoid( x ):
	return 1 / ( 1 + exp_v( -x ) )

