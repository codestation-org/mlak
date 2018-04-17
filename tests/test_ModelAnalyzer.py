#! /usr/bin/python3

import sys
sys.path.extend( [ "./mlak" ] )

import unittest
import numpy.testing as npt

from ModelAnalyzer import *
from tests.data_gen import *
from mlak.utils import CapturedStdout, fix_random
import LinearRegression as linReg
import LogisticRegression as logReg
import NeuralNetwork as nn
import KerasSolver as ks
import numpy as np


class TestModelAnalyzer( unittest.TestCase ):
	def setUp( self ):
		fix_random()
	def test_DataShaper_features( self ):
		X = np.array( [[1, 2, 3], [1.5, 4, 9], [1.7, 10, 100]] )
		ds = DataShaper( X, None )
		self.assertEqual( ds.feature_count(), 3 )
		npt.assert_almost_equal( ds.mu(), np.array( [ 1.4, 5.333333333333, 37.333333333333] ) )
		npt.assert_almost_equal( ds.sigma(), np.array( [ 0.294392, 3.399346342395, 44.379675027602] ) )
		ds = DataShaper( X, [
			lambda x : x[1] * x[2]
		] )
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

	def test_find_solution( self ):
		X, y = gen_regression_data()
		solver = linReg.LinearRegressionSolver()
		with CapturedStdout():
			optimizationResults = find_solution(
				solver, X, y,
				showFailureRateTrain = True,
				optimizationParams = {
					"nnTopology": "",
					"Lambda": [0.01, 0.1, 1],
					"functions": [
						[],
						[
							lambda x: x[0] ** 2,
							lambda x: x[1] ** 2
						],
						[
							lambda x: x[0] ** 2,
							lambda x: x[1] ** 2,
							lambda x: x[2] ** 2
						]
					]
				},
				files = [],
				log = {
					"log_dir": "out",
					"log_file_name": "mlak"
				}
			)
		self.assertAlmostEqual( optimizationResults.failureRateTest, 1e-07, 6 )

	def test_analyze_linreg( self ):
		X, y = gen_regression_data()
		solver = linReg.LinearRegressionSolver()
		with CapturedStdout():
			analyzerResults = analyze(
				solver, X, y,
				optimizationParams = {
					"nnTopology": "",
					"Lambda": 0.1,
					"functions": [
						lambda x: x[0] ** 2,
						lambda x: x[1] ** 2,
						lambda x: x[2] ** 2
					]
				},
				iterations = 40,
				bins = 3,
				tries = 4,
				sample_iterations = 40
			)
		npt.assert_equal( analyzerResults.sampleCountAnalyzis.sampleCount, [2133, 4266, 6400] )
		npt.assert_almost_equal( analyzerResults.sampleCountAnalyzis.errorTrain, [8.25e-05, 2.06e-05, 9.19e-06], 5 )
		npt.assert_almost_equal( analyzerResults.sampleCountAnalyzis.errorCV, [8.28e-05, 2.05e-05, 9.04e-06], 5 )
		npt.assert_equal( analyzerResults.iterationCountAnalyzis.iterationCount, [13, 26, 40] )
		npt.assert_almost_equal( analyzerResults.iterationCountAnalyzis.errorTrain, [2.25e-06, 2.25e-06, 2.25e-06], 5 )
		npt.assert_almost_equal( analyzerResults.iterationCountAnalyzis.errorCV, [2.28e-06, 2.28e-06, 2.28e-06], 5 )

	def test_analyze_logreg( self ):
		X, y = gen_logistic_data()
		solver = logReg.LogisticRegressionSolver()
		with CapturedStdout():
			analyzerResults = analyze(
				solver, X, y,
				optimizationParams = {
					"nnTopology": "",
					"Lambda": 1,
					"functions": None
				},
				iterations = 40,
				bins = 3,
				tries = 4,
				sample_iterations = 4
			)
		npt.assert_equal( analyzerResults.sampleCountAnalyzis.sampleCount, [500, 1000, 1500] )
		npt.assert_almost_equal( analyzerResults.sampleCountAnalyzis.errorTrain, [0, 0, 0], 2 )
		npt.assert_almost_equal( analyzerResults.sampleCountAnalyzis.errorCV, [0, 0, 0] )
		npt.assert_equal( analyzerResults.iterationCountAnalyzis.iterationCount, [13, 26, 40] )
		npt.assert_almost_equal( analyzerResults.iterationCountAnalyzis.errorTrain, [0, 0, 0], 5 )
		npt.assert_almost_equal( analyzerResults.iterationCountAnalyzis.errorCV, [0, 0, 0], 5 )

	def test_analyze_neuralnetwork( self ):
		X, y = gen_logistic_data()
		solver = nn.NeuralNetworkSolver()
		with CapturedStdout():
			analyzerResults = analyze(
				solver, X, y,
				optimizationParams = {
					"nnTopology": "10",
					"Lambda": 1,
					"functions": None
				},
				iterations = 40,
				bins = 3,
				tries = 4,
				sample_iterations = 10
			)
		npt.assert_equal( analyzerResults.sampleCountAnalyzis.sampleCount, [500, 1000, 1500] )
		npt.assert_almost_equal( analyzerResults.sampleCountAnalyzis.errorTrain, [0., 0, 0] )
		npt.assert_almost_equal( analyzerResults.sampleCountAnalyzis.errorCV, [0., 0., 0.] )
		npt.assert_equal( analyzerResults.iterationCountAnalyzis.iterationCount, [13, 26, 40] )
		npt.assert_almost_equal( analyzerResults.iterationCountAnalyzis.errorTrain, [0, 0, 0], 5 )
		npt.assert_almost_equal( analyzerResults.iterationCountAnalyzis.errorCV, [0, 0, 0], 5 )

	def test_analyze_keras( self ):
		X, y = gen_logistic_data()
		solver = ks.KerasSolver()
		with CapturedStdout():
			analyzerResults = analyze(
				solver, X, y,
				optimizationParams = {
					"nnTopology": "N(10)",
					"Lambda": 1,
					"functions": None
				},
				iterations = 40,
				bins = 3,
				tries = 4,
				sample_iterations = 4
			)
		npt.assert_equal( analyzerResults.sampleCountAnalyzis.sampleCount, [500, 1000, 1500] )
		npt.assert_almost_equal( analyzerResults.sampleCountAnalyzis.errorTrain, [0.02, 0, 0], 2 )
		npt.assert_almost_equal( analyzerResults.sampleCountAnalyzis.errorCV, [0.024, 0, 0], 2 )
		npt.assert_equal( analyzerResults.iterationCountAnalyzis.iterationCount, [13, 26, 40] )
		npt.assert_almost_equal( analyzerResults.iterationCountAnalyzis.errorTrain, [0, 0, 0], 5 )
		npt.assert_almost_equal( analyzerResults.iterationCountAnalyzis.errorCV, [0, 0, 0], 5 )

if __name__ == '__main__':
	unittest.main()

