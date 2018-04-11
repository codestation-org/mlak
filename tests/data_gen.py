#! /usr/bin/python3

from itertools import product
import numpy as np
from math import *
import mlak.Visual as vis

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

def gen_logistic_data():
	def class_drill( x1, x2, p ):
		d = x2 * ( 1 + p[2] ) + p[3]
		return abs( ( x1 * ( 1 + p[0] ) + p[1] ) % ( d if d else 0.001 ) )

	classes = (
		( lambda x1, x2, p: cos( 2 * sqrt( ( x1 * ( 1 + p[0] ) + p[1] ) ** 2 + ( x2 * ( 1 + p[2] ) + p[3] ) ** 2 ) ), "ripple" ),
		( lambda x1, x2, p: abs(  x1 * ( 1 + p[0] ) + p[1] ) + abs( x2 * ( 1 + p[2] ) + p[3] ), "diamond" ),
		( class_drill, "drill" ),
	)
	X = []
	y = []
	for c in classes:
		for p in product( ( 0, 0.2, 0.4, -0.2, -0.4 ), repeat = 4 ):
			x = []
			for x1, x2 in product( np.linspace( -2, 2, 16 ), repeat = 2 ):
				x.append( c[0]( x1, x2, p ) )
			X.append( x )
			y.append( c[1] )
	return np.array( X ), np.array( y )

if __name__ == '__main__':
	X, y = gen_logistic_data()
	se = vis.SampleEditor( X, y, zoom = 10 )
	se.run()
	print( len( y ) )

