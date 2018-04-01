#! /usr/bin/python3

import sys
sys.path.extend( [ "./mlak" ] )

import unittest
import numpy.testing as npt

from FeatureTools import find_normalization_params, normalize_features, feature_normalize, add_features
import numpy as np

class TestFeatureTools( unittest.TestCase ):
	def test_10_find_normalization_params( self ):
		TestFeatureTools.normParams = find_normalization_params( np.array( [[1, 2, 3], [1, 4, 9], [1, 10, 100]] ) )
		self.assertEqual( len( self.normParams[0] ), 3 )
		self.assertEqual( len( self.normParams[1] ), 3 )
		npt.assert_almost_equal( self.normParams[0], np.array( [ 1, 5.333333333333, 37.333333333333] ) )
		npt.assert_almost_equal( self.normParams[1], np.array( [ 0, 3.399346342395, 44.379675027602] ) )

	def test_20_normalize_features( self ):
		X = normalize_features( np.array( [[1, 2, 3], [1, 4, 9], [1, 10, 100]] ), self.normParams[0], self.normParams[1] )
		npt.assert_almost_equal(
			X,
			np.array( [
				[ 0., -0.9805807, -0.7736274],
				[ 0., -0.3922323, -0.6384304],
				[ 0.,  1.3728129,  1.4120578]
			] )
		)

	def test_feature_normalize( self ):
		X, mu, sigma = feature_normalize( np.array( [[1, 2, 3], [1, 4, 9], [1, 10, 100]] ) )
		npt.assert_almost_equal(
			X,
			np.array( [
				[ 0., -0.9805807, -0.7736274],
				[ 0., -0.3922323, -0.6384304],
				[ 0.,  1.3728129,  1.4120578]
			] )
		)
		npt.assert_almost_equal( mu, np.array( [ 1, 5.333333333333, 37.333333333333] ) )
		npt.assert_almost_equal( sigma, np.array( [ 0, 3.399346342395, 44.379675027602] ) )

	def test_add_features( self ):
		X = np.array( [[1, 2, 3], [4, 5, 6], [7, 8, 9]] )
		X_ext = add_features( X, [ lambda x: x[0]**2, lambda x: x[1]**3 ] )
		npt.assert_almost_equal(
			X_ext,
			np.array( [
				[1, 2, 3, 1, 8],
				[4, 5, 6, 16, 125],
				[7, 8, 9, 49, 512]
			] )
		)
		X_ext = add_features( X, [] )
		npt.assert_almost_equal( X_ext, X )

if __name__ == '__main__':
	unittest.main()

