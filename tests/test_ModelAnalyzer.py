#! /usr/bin/python3

import sys
sys.path.extend( [ "./mlak" ] )

import unittest
import numpy.testing as npt

from ModelAnalyzer import *
import numpy as np

class TestModelAnalyzer( unittest.TestCase ):
	def test_DataShaper_features( self ):
		X = np.array( [[1, 2, 3], [1.5, 4, 9], [1.7, 10, 100]] )
		ds = DataShaper( X, None )
		self.assertEqual( ds.feature_count(), 3 )
		npt.assert_almost_equal( ds.mu(), np.array( [ 1.4, 5.333333333333, 37.333333333333] ) )
		npt.assert_almost_equal( ds.sigma(), np.array( [ 0.294392, 3.399346342395, 44.379675027602] ) )
		ds = DataShaper( X, [lambda x : x[1] * x[2]] )
		self.assertEqual( ds.is_classifier(), False )
		self.assertEqual( ds.feature_count(), 4 )
		npt.assert_almost_equal( ds.mu(), np.array( [ 1.4, 5.333333333333, 37.333333333333, 347.3333333] ) )
		npt.assert_almost_equal( ds.sigma(), np.array( [ 0.294392, 3.399346342395, 44.379675027602, 461.6675090245023] ) )
		Xnb = np.array( [[7, 10, 19], [45, 49, 91], [23, 37, 29]] )
		Xc = ds.conform( Xnb )
		Xexp = np.array( [
			[1, 19.02225417, 1.37281295, -0.413102018, -0.340793602],
			[1, 148.10183607, 12.8456069,  1.20926227, 8.90612093],
			[1, 73.37155181, 9.31551642, -0.187773645, 1.57183829]
		] )
		npt.assert_almost_equal( Xc, Xexp, 1 )

	def test_DataShaper_labels( self ):
		X = np.array( [[0], [1], [2], [2.5], [1.3], [1.7], [0.1], [0.3], [0.7] ] )
		y = [ "a", "b", "c", "c", "b", "b" , "a", "a", "a" ]
		ds = DataShaper( X )
		self.assertEqual( ds.feature_count(), 1 )
		self.assertEqual( ds.is_classifier(), False )
		ds.learn_labels( y )
		self.assertEqual( ds.is_classifier(), True )
		self.assertEqual( ds.class_count(), 3 )
		yl = ds.labels( np.array( [2, 1, 0] ) )
		npt.assert_equal( yl, ["c", "b", "a"] )
		yc = ds.map_labels( ["b", "c", "a"] )
		npt.assert_equal( yc, [[1], [2], [0]] )
		ds.learn_labels( np.array( [4, 5, 6] ) )
		npt.assert_equal( ds.map_labels( np.array( [[6], [5], [4]] ) ), [[2], [1], [0]] )


	def test_split_data( self ):
		m = 100
		X = np.arange( m )
		y = np.arange( m )
		ds = split_data( X, y, cvFraction = 0.3, testFraction  = 0.3 )
		self.assertEqual( len( ds.trainSet.X ), 40 )
		self.assertEqual( len( ds.trainSet.y ), 40 )
		self.assertEqual( len( ds.crossValidationSet.X ), 30 )
		self.assertEqual( len( ds.crossValidationSet.y ), 30 )
		self.assertEqual( len( ds.testSet.X ), 30 )
		self.assertEqual( len( ds.testSet.y ), 30 )
		npt.assert_equal( ds.trainSet.X, ds.trainSet.y )
		npt.assert_equal( ds.crossValidationSet.X, ds.crossValidationSet.y )
		npt.assert_equal( ds.testSet.X, ds.testSet.y )
		Xd = np.sort( np.concatenate( ( ds.trainSet.X, ds.crossValidationSet.X, ds.testSet.X ) ) )
		yd = np.sort( np.concatenate( ( ds.trainSet.y, ds.crossValidationSet.y, ds.testSet.y ) ) )
		npt.assert_equal( Xd, X )
		npt.assert_equal( yd, y )


if __name__ == '__main__':
	unittest.main()

