#! /usr/bin/python3

import sys
sys.path.extend( [ "./mlak" ] )

from itertools import product
import unittest
import numpy.testing as npt

from FeatureTools import add_features
from LinearAlgebra import columnize, transpose, add_ones_column, normal_equation
import numpy as np

class TestMathTools( unittest.TestCase ):
	def test_columnize( self ):
		y = np.array( [1, 2, 3, 4] )
		self.assertEqual( y.shape, ( 4, ) )
		y = columnize( y )
		npt.assert_equal( y, np.array( [[1],[2],[3],[4]] ) )
		y = np.array( [[1, 2, 3, 4]] )
		self.assertEqual( y.shape, ( 1, 4 ) )
#		y = columnize( y ) # columnize does not support this feature.
#		self.assertEqual( y.shape, ( 4, 1 ) )

	def test_transpose( self ):
		y = np.array( [[1, 2, 3, 4]] )
		self.assertEqual( y.shape, ( 1, 4 ) )
		y = transpose( y )
		npt.assert_equal( y, np.array( [[1],[2],[3],[4]] ) )
		y = transpose( y )
		npt.assert_equal( y, np.array( [[1,2,3,4]] ) )

	def test_add_ones_column( self ):
		X = np.array( [ [1, 2, 3], [4, 5, 6], [7, 8, 9] ] )
		X = add_ones_column( X )
		npt.assert_equal( X, np.array( [ [1, 1, 2, 3], [1, 4, 5, 6], [1, 7, 8, 9] ] ) )

	def test_normal_equation( self ):
		poly = np.array( [ [1, 2, 3], [4, 5, 6], [7, 8, 9] ] )
		linSpace = np.linspace( -10, 10, 20 )
		X = []
		y = []
		for x0, x1, x2 in product( *( linSpace, linSpace, linSpace ) ):
			X.append( [ x0, x1, x2 ] )
			y.append(
				x0 ** 2 * poly[0][0] + x0 * poly[0][1] + poly[0][2]
				+ x1 ** 2 * poly[1][0] + x1 * poly[1][1] + poly[1][2]
				+ x2 ** 2 * poly[2][0] + x2 * poly[2][1] + poly[2][2]
			)
		X = np.array( X )
		y = np.array( y )
		Wlin = normal_equation( X, y, 0.0 )
		ypLin = np.dot( Wlin, X.T )
		X = add_features( X, [ lambda x: x[0] ** 2, lambda x: x[1] ** 2, lambda x: x[2] ** 2 ] )
		Wext = normal_equation( X, y, 0.0 )
		ypExt = np.dot( Wext, X.T )
		ydLin = np.sum( y - ypLin )
		ydExt = np.sum( y - ypExt )
		improvement = ydLin / ydExt
		self.assertAlmostEqual( improvement, 122.1428429027067 )
#		print( f"improvement = {improvement}" )

if __name__ == '__main__':
	unittest.main()

