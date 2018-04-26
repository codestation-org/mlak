#! /usr/bin/python3

import sys
sys.path.extend( [ "./mlak", "./tests" ] )

from itertools import product
import unittest
import numpy.testing as npt

from FeatureTools import add_features
from LogisticRegression import *
from ModelAnalyzer import *
from data_gen import *
import numpy as np

class TestLogisticRegression( unittest.TestCase ):
	def setUp( self ):
		mu.fix_random()

	def test( self ):
		X, y = gen_logistic_data()
		solver = LogisticRegressionSolver()
		self.assertEqual( solver.type(), ma.SolverType.CLASSIFIER )
		solution = solver.train( X, y, Lambda = 1 )
		self.assertEqual( solution.shaper().class_count(), 3 )
		cost = solver.verify( solution, X, y )
		self.assertEqual( cost, 0 )
		m = len( y )
		Xt = np.array( [X[0], X[m // 2], X[-1]] )
		yp = solver.predict( solution, Xt )
		npt.assert_equal( yp, ["ripple", "diamond", "drill"] )
		yc = solver.examine( solution, Xt )
		self.assertEqual( len( yc ), 3 )
		self.assertEqual( len( yc[0] ), 3 )
		for y in yc:
			self.assertTrue( mu.is_sorted( y, key = lambda x: x.confidence, reverse = True ) )
		npt.assert_almost_equal( [x.confidence for x in yc[0]], [17.4079843, -20.84046, -26.3580832], 1 )
		npt.assert_equal( [x.label for x in yc[0]], ["ripple", "diamond", "drill"] )
		npt.assert_almost_equal( [x.confidence for x in yc[1]], [32.9144321, -22.929551, -31.8464051], 1 )
		npt.assert_equal( [x.label for x in yc[1]], ["diamond", "drill", "ripple"] )
		npt.assert_almost_equal( [x.confidence for x in yc[2]], [4.2371505, -5.4202037, -11.4355583], 1 )
		npt.assert_equal( [x.label for x in yc[2]], ["drill", "ripple", "diamond"] )

if __name__ == '__main__':
	unittest.main()

