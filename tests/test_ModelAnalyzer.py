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
		y = [ "a", "b", "c", "c", "b", "b", "a", "a", "a" ]
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
		npt.assert_equal( analyzerResults.sampleCountAnalyzis.sampleCount, [1, 4, 7, 10, 19, 28, 54, 80, 159, 238, 475, 712, 1423, 2134, 4267, 6400] )
		npt.assert_almost_equal( analyzerResults.sampleCountAnalyzis.errorTrain, [
			0.00000000e+00, 2.33847837e+01, 3.00222512e+01, 7.96243961e+00,
			1.37056787e+00, 5.14152946e-01, 1.26362331e-01, 6.45051925e-02,
			1.46205243e-02, 7.25704802e-03, 1.67555343e-03, 7.51420424e-04,
			1.82472270e-04, 8.16512666e-05, 2.04713359e-05, 9.08089772e-06
		], 5 )
		npt.assert_almost_equal( analyzerResults.sampleCountAnalyzis.errorCV, [
			4.99885725e+04, 1.75878006e+04, 4.13273401e+03, 4.01030109e+01,
			3.82169238e+00, 6.81074172e-01, 1.43006176e-01, 7.01556101e-02,
			1.44597141e-02, 8.57240292e-03, 1.67551231e-03, 7.42103605e-04,
			1.73911141e-04, 7.83993282e-05, 1.98093851e-05, 8.65960541e-06
		], 4 )
		npt.assert_equal( analyzerResults.iterationCountAnalyzis.iterationCount, [2, 4, 6, 10, 14, 27, 40] )
		npt.assert_almost_equal( analyzerResults.iterationCountAnalyzis.errorTrain, [
			2.18325422e-01, 1.04552670e-05, 2.27461723e-06, 2.27461723e-06,
			2.27461723e-06, 2.27461723e-06, 2.27461723e-06
		], 5 )
		npt.assert_almost_equal( analyzerResults.iterationCountAnalyzis.errorCV, [
			2.14481838e-01, 1.01920583e-05, 2.16874107e-06, 2.16874107e-06,
			2.16874107e-06, 2.16874107e-06, 2.16874107e-06
		], 5 )

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
		npt.assert_equal( analyzerResults.sampleCountAnalyzis.sampleCount, [4, 6, 8, 14, 20, 38, 56, 112, 168, 334, 500, 1000, 1500] )
		npt.assert_almost_equal( analyzerResults.sampleCountAnalyzis.errorTrain, [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], 2 )
		npt.assert_almost_equal( analyzerResults.sampleCountAnalyzis.errorCV, [0.0646667, 0.104, 0.1846667, 0.0126667, 0.0166667, 0.0053333, 0.0006667, 0.0006667, 0., 0.0006667, 0., 0., 0.] )
		npt.assert_equal( analyzerResults.iterationCountAnalyzis.iterationCount, [2, 4, 6, 10, 14, 27, 40] )
		npt.assert_almost_equal( analyzerResults.iterationCountAnalyzis.errorTrain, [0., 0.0005, 0., 0., 0., 0., 0.], 5 )
		npt.assert_almost_equal( analyzerResults.iterationCountAnalyzis.errorCV, [0, 0, 0, 0, 0, 0, 0], 5 )

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
		npt.assert_equal( analyzerResults.sampleCountAnalyzis.sampleCount, [4, 6, 8, 14, 20, 38, 56, 112, 168, 334, 500, 1000, 1500] )
		npt.assert_almost_equal( analyzerResults.sampleCountAnalyzis.errorTrain, [0., 0.08, 0.03, 0., 0.04, 0., 0., 0., 0., 0., 0., 0., 0.], 2 )
		npt.assert_almost_equal( analyzerResults.sampleCountAnalyzis.errorCV, [0.3426667, 0.3333333, 0.3, 0.0206667, 0.0886667, 0.004, 0., 0.0013333, 0., 0., 0., 0., 0.] )
		npt.assert_equal( analyzerResults.iterationCountAnalyzis.iterationCount, [2, 4, 6, 10, 14, 27, 40] )
		npt.assert_almost_equal( analyzerResults.iterationCountAnalyzis.errorTrain, [0.0135, 0.00083, 0., 0., 0., 0., 0.], 5 )
		npt.assert_almost_equal( analyzerResults.iterationCountAnalyzis.errorCV, [0.012, 0.00133, 0., 0., 0., 0., 0.], 5 )

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
		npt.assert_equal( analyzerResults.sampleCountAnalyzis.sampleCount, [4, 6, 8, 14, 20, 38, 56, 112, 168, 334, 500, 1000, 1500] )
		npt.assert_almost_equal( analyzerResults.sampleCountAnalyzis.errorTrain, [0.19, 0.46, 0.34, 0.18, 0.31, 0.3, 0.45, 0.21, 0.13, 0.03, 0.02, 0., 0.], 0 )
		npt.assert_almost_equal( analyzerResults.sampleCountAnalyzis.errorCV, [0.48, 0.51, 0.27, 0.25, 0.4, 0.26, 0.46, 0.23, 0.13, 0.04, 0.02, 0., 0.], 0 )
		npt.assert_equal( analyzerResults.iterationCountAnalyzis.iterationCount, [2, 4, 6, 10, 14, 27, 40] )
		npt.assert_almost_equal( analyzerResults.iterationCountAnalyzis.errorTrain, [0., 0., 0., 0., 0., 0., 0.], 5 )
		npt.assert_almost_equal( analyzerResults.iterationCountAnalyzis.errorCV, [0., 0., 0., 0., 0., 0., 0.], 5 )

if __name__ == '__main__':
	unittest.main()

