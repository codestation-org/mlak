#! /usr/bin/python3

from itertools import product
import numpy as np

def gen_regression_data():
	poly = np.array( [ [1, 2, 3], [4, 5, 6], [7, 8, 9] ] )
	X = []
	y = []
	for x0, x1, x2 in product( np.linspace( -10, 10, 20 ), repeat = 3 ):
		X.append( [ x0, x1, x2 ] )
		y.append(
			x0 ** 2 * poly[0][0] + x0 * poly[0][1] + poly[0][2]
			+ x1 ** 2 * poly[1][0] + x1 * poly[1][1] + poly[1][2]
			+ x2 ** 2 * poly[2][0] + x2 * poly[2][1] + poly[2][2]
		)
	m = len( X )
	X = np.array( X )
	y = np.array( y )
	y.shape = ( m, 1 )
	return ( X, y )

