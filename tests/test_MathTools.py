#! /usr/bin/python3

import sys
sys.path.extend( [ "./mlak" ] )

import unittest
import numpy.testing as npt

from MathTools import sigmoid, sigmoid_gradient, log
import numpy as np

class TestMathTools( unittest.TestCase ):
	def test_sigmoid( self ):
		self.assertEqual( sigmoid( 0 ), 0.5 )
		self.assertEqual(sigmoid(1) + sigmoid(-1), 1)
		self.assertAlmostEqual(sigmoid(2) + sigmoid(-2), 1)
		self.assertEqual(sigmoid(1), 1 / (1 + np.exp(-1)))
		npt.assert_almost_equal(
			sigmoid( np.array( [ -2, -1, -0.5, 0, 0.5, 1., 2. ] ) ),
			np.array( [ 0.1192029, 0.2689414, 0.3775407, 0.5, 0.6224593, 0.7310586, 0.8807971 ] )
		)

	def test_sigmoid_gradient( self ):
		self.assertEqual( sigmoid_gradient( 0 ), 0.25 )
		self.assertAlmostEqual( sigmoid_gradient( 1 ), 0.19661193324148185 )
		npt.assert_almost_equal( sigmoid_gradient( np.array( [ -1, 10 ] ) ), np.array( [ 1.96611933e-01, 4.53958077e-05 ] ) )

	def test_log( self ):
		npt.assert_almost_equal(
			log( np.array( [ 0, 0.5, 1., 2., 10, 100, 1000 ] ) ),
			[ -230.2585093, -0.6931472, 0., 0.6931472, 2.3025851, 4.6051702, 6.9077553 ]
		)

if __name__ == '__main__':
	unittest.main()

