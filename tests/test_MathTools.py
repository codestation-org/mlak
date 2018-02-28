#! /usr/bin/python3

import sys
sys.path.extend( [ "./mlak" ] )

import unittest

from MathTools import sigmoid
import numpy as np

class TestMathTools( unittest.TestCase ):
	def test_sigmoid( self ):
		self.assertEqual( sigmoid( 0 ), 0.5 )
		# self.assertEqual(sigmoid(1) + sigmoid(-1), 1)
		# self.assertEqual(sigmoid(2) + sigmoid(-2), 1)
		# self.assertEqual(sigmoid(1), 1 / (1 + np.exp(-1)))

if __name__ == '__main__':
	unittest.main()

