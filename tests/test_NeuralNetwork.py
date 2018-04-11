#! /usr/bin/python3

import sys
sys.path.extend( [ "./mlak", "./tests" ] )

from itertools import product
import unittest
import numpy.testing as npt

from FeatureTools import add_features
from NeuralNetwork import *
from ModelAnalyzer import *
from data_gen import *
import numpy as np

class TestNeuralNetwork( unittest.TestCase ):
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

if __name__ == '__main__':
	unittest.main()

