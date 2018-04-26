#! /usr/bin/python3

import sys
sys.path.extend( [ "./mlak", "./tests" ] )

from itertools import product
import unittest
import numpy.testing as npt

from NeuralNetwork import *
from ModelAnalyzer import *
from data_gen import *
import numpy as np

class TestNeuralNetwork( unittest.TestCase ):
	def setUp( self ):
		mu.fix_random()

	def test( self ):
		X, y = gen_logistic_data()
		solver = NeuralNetworkSolver()
		self.assertEqual( solver.type(), ma.SolverType.CLASSIFIER )
		solution = solver.train( X, y, nnTopology = "10", Lambda = 1 )
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
		npt.assert_almost_equal( [x.confidence for x in yc[0]], [0.9969892, 0.0016289, 0.0013147], 5 )
		npt.assert_equal( [x.label for x in yc[0]], ["ripple", "diamond", "drill"] )
		npt.assert_almost_equal( [x.confidence for x in yc[1]], [0.9970075, 0.0014194, 0.0013228], 5 )
		npt.assert_equal( [x.label for x in yc[1]], ["diamond", "drill", "ripple"] )
		npt.assert_almost_equal( [x.confidence for x in yc[2]], [0.98733  , 0.0047969, 0.0024878], 5 )
		npt.assert_equal( [x.label for x in yc[2]], ["drill", "ripple", "diamond"] )

if __name__ == '__main__':
	unittest.main()

