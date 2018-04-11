#! /usr/bin/python3

import sys
sys.path.extend( [ "./mlak", "./tests" ] )

from itertools import product
import unittest
import numpy.testing as npt

from FeatureTools import add_features
from OptimizationAlgorithms import *
from LinearRegression import *
from ModelAnalyzer import *
from data_gen import *
import numpy as np

class TestOptimizationAlgorithms( unittest.TestCase ):
	def test_gradient_descent( self ):
		X, y = gen_regression_data()
		shaper = DataShaper( X )
		W = np.zeros( shaper.feature_count() + 1 )
		Xn = shaper.conform( X )
		W, _ = gradient_descent( Algorithm( compute_cost, compute_grad ), Xn, y, W, 0.3, 50, 0.01 )
		cost = compute_cost( Xn, y, W, 0 )
		self.assertAlmostEqual( cost, 35564.36798367374 )
		X = add_features( X, [ lambda x: x[0] ** 2, lambda x: x[1] ** 2, lambda x: x[2] ** 2 ] )
		shaper = DataShaper( X )
		W = np.zeros( shaper.feature_count() + 1 )
		Xn = shaper.conform( X )
		W, _ = gradient_descent( Algorithm( compute_cost, compute_grad ), Xn, y, W, 0.3, 50, 0.01 )
		cost = compute_cost( Xn, y, W, 0 )
		self.assertAlmostEqual( cost, 5.8246025667834156e-08 )
		yp = np.dot( Xn, W )
		npt.assert_almost_equal( yp, y.flatten(), 3 )

if __name__ == '__main__':
	unittest.main()

