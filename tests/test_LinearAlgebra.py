#! /usr/bin/python3

import sys
sys.path.extend( [ "./mlak", "./tests" ] )

from data_gen import *
from itertools import product
import unittest
import numpy.testing as npt

from FeatureTools import add_features
from LinearAlgebra import columnize, transpose, add_ones_column, normal_equation
import numpy as np

class TestLinearAlgebra( unittest.TestCase ):
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
		X, y = gen_regression_data()
		Wlin = normal_equation( X, y, 0.0 )
		ypLin = np.dot( X, Wlin )
		X = add_features( X, [ lambda x: x[0] ** 2, lambda x: x[1] ** 2, lambda x: x[2] ** 2 ] )
		Wext = normal_equation( X, y, 0.0 )
		ypExt = np.dot( X, Wext )
		ydLin = np.sum( y - ypLin )
		ydExt = np.sum( y - ypExt )
		improvement = ydLin / ydExt
		self.assertAlmostEqual( improvement, 122.1428429027067 )
#		print( f"improvement = {improvement}" )

if __name__ == '__main__':
	unittest.main()

