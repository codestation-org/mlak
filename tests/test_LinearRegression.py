#! /usr/bin/python3

import sys
sys.path.extend( [ "./mlak", "./tests" ] )

from itertools import product
import unittest
import numpy.testing as npt

from FeatureTools import add_features
from LinearRegression import *
from ModelAnalyzer import *
from data_gen import *
import numpy as np

class TestLinearRegression( unittest.TestCase ):
	def test_compute_cost( self ):
		X, y = gen_regression_data()
		solver = LinearRegressionSolver()
		self.assertEqual( solver.type(), ma.SolverType.VALUE_PREDICTOR )
		solution = solver.train( X, y, Lambda = 0.01 )
		cost = solver.verify( solution, X, y )
		self.assertAlmostEqual( cost, 35564.36798367374 )
		X = add_features( X, [ lambda x: x[0] ** 2, lambda x: x[1] ** 2, lambda x: x[2] ** 2 ] )
		solution = solver.train( X, y, Lambda = 0.01 )
		cost = solver.verify( solution, X, y )
		self.assertAlmostEqual( cost, 5.8246025667834156e-08 )
		yp = solver.predict( solution, X )
		npt.assert_almost_equal( yp, y.flatten(), 3 )

if __name__ == '__main__':
	unittest.main()

