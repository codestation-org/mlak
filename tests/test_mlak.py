#! /usr/bin/python3

import sys
sys.path.extend( [ "./mlak" ] )

import unittest
import numpy.testing as npt
from tests.data_gen import *

from collections import namedtuple

import mlak.mlak as mlak

from mlak.utils import *
import DataIO as dio
import ModelAnalyzer as mo
import LinearRegression as linReg
import LogisticRegression as logReg
import NeuralNetwork as nn
import KerasSolver as ks

class Args( dict ):
	def __init__( self_,
		data_set = None,
		solution = None,
		topology = None,
		Lambda = None,
		iterations = None,
		functions = [],
		engine = None,
		cvFraction = 0.1,
		testFraction = 0.1,
		verbose = False,
		debug = False
	):
		self_.data_set = data_set
		self_.solution = solution
		self_.topology = topology
		self_.Lambda = Lambda
		self_.iterations = iterations
		self_.functions = functions
		self_.engine = engine
		self_.cvFraction = cvFraction
		self_.testFraction = testFraction
		self_.verbose = verbose
		self_.debug = debug

class TestMLAK( unittest.TestCase ):
	def setUp( self ):
		fix_random()

	def test_set_preferred_engine( self ):
		mlak.set_preferred_engine( "linreg" )
		self.assertEqual( type( mlak.preferredEngine ), type( linReg.LinearRegressionSolver ) )
		mlak.set_preferred_engine( "logreg" )
		self.assertEqual( type( mlak.preferredEngine ), type( logReg.LogisticRegressionSolver ) )
		mlak.set_preferred_engine( "struggle" )
		self.assertEqual( type( mlak.preferredEngine ), type( nn.NeuralNetworkSolver ) )
		mlak.set_preferred_engine( "keras" )
		self.assertEqual( type( mlak.preferredEngine ), type( ks.KerasSolver ) )
		with self.assertRaises( Exception ):
			mlak.set_preferred_engine( "xxx" )

	def test_linreg_10_train( self ):
		X, y = gen_regression_data()
		dio.save( "./out/lin.p", { "X": X, "y": y } )
		args = Args(
			data_set = "./out/lin.p",
			solution = "./out/lin-sol.p",
			topology = [],
			Lambda = "0.1",
			iterations = 10,
			functions = [
				"lambda x: x[0] ** 2",
				"lambda x: x[1] ** 2",
				"lambda x: x[2] ** 2"
			],
			engine = "linreg"
		)
		mlak.set_preferred_engine( args.engine )
		with CapturedStdout():
			mlak.train( args )
		solution = dio.load( "./out/lin-sol.p" )
		npt.assert_almost_equal( solution.model(), [460.3618418, 12.1335894, 30.351926, 48.5490816, 32.8447929, 131.4717615, 229.648099] )
		npt.assert_almost_equal( solution.shaper().mu(), [-1.43274854e-02, 2.85087719e-02, 2.80701754e-02, 3.68073253e+01, 3.68513389e+01, 3.68301016e+01] )
		npt.assert_almost_equal( solution.shaper().sigma(), [6.06688718, 6.07046342, 6.06871598, 32.84525096, 32.86839366, 32.80732591] )

	def test_linreg_20_test( self ):
		X, y = gen_regression_data()
		args = Args(
			data_set = "./out/lin.p",
			solution = "./out/lin-sol.p",
			engine = "linreg"
		)
		mlak.set_preferred_engine( args.engine )
		with CapturedStdout() as out:
			mlak.test( args )
		self.assertAlmostEqual( float( out.getvalue() ), 7.117094744250965e-06 )

	def test_logreg_10_train( self ):
		X, y = gen_logistic_data()
		dio.save( "./out/log.p", { "X": X, "y": y } )
		args = Args(
			data_set = "./out/log.p",
			solution = "./out/log-sol.p",
			topology = [],
			Lambda = "0.1",
			iterations = 1,
			engine = "logreg"
		)
		mlak.set_preferred_engine( args.engine )
		with CapturedStdout():
			mlak.train( args )

		with CapturedStdout() as out:
			mlak.test( args )
		self.assertEqual( out.getvalue(), "0.2666666666666667\n" )
		solution = dio.load( "./out/log-sol.p" )
		npt.assert_equal( solution.shaper()._classesIdToLabel, ["diamond", "drill", "ripple"] )
		self.assertEqual( len( solution.model() ), 3 )
		npt.assert_almost_equal( np.sum( solution.model(), axis = 1 ), [15.9547132, -0.9575446, -4.9745893] )
		self.assertAlmostEqual( np.sum( solution.shaper().mu() ), 205.85431033692913 )
		self.assertAlmostEqual( np.sum( solution.shaper().sigma() ), 296.5719143203778 )

		fix_random()
		args.iterations = 9
		with CapturedStdout():
			mlak.train( args )

		with CapturedStdout() as out:
			mlak.test( args )
		self.assertEqual( out.getvalue(), "0.0\n" )
		solution = dio.load( "./out/log-sol.p" )
		npt.assert_equal( solution.shaper()._classesIdToLabel, ["diamond", "drill", "ripple"] )
		self.assertEqual( len( solution.model() ), 3 )
		npt.assert_almost_equal( np.sum( solution.model(), axis = 1 ), [20.5073757, -13.1428356, -10.8690069] )
		self.assertAlmostEqual( np.sum( solution.shaper().mu() ), 205.85431033692913 )
		self.assertAlmostEqual( np.sum( solution.shaper().sigma() ), 296.5719143203778 )

	def test_logreg_20_test( self ):
		X, y = gen_regression_data()
		args = Args(
			data_set = "./out/log.p",
			solution = "./out/log-sol.p",
			engine = "logreg"
		)
		mlak.set_preferred_engine( args.engine )
		with CapturedStdout() as out:
			mlak.test( args )
		self.assertEqual( out.getvalue(), "0.0\n" )

	def test_struggle_10_train( self ):
		X, y = gen_logistic_data()
		dio.save( "./out/struggle.p", { "X": X, "y": y } )
		args = Args(
			data_set = "./out/struggle.p",
			solution = "./out/struggle-sol.p",
			topology = ["10"],
			Lambda = "0.1",
			iterations = 1,
			engine = "struggle"
		)
		mlak.set_preferred_engine( args.engine )
		with CapturedStdout():
			mlak.train( args )
		with CapturedStdout() as out:
			mlak.test( args )
		self.assertEqual( out.getvalue(), "0.5706666666666667\n" )
		solution = dio.load( "./out/struggle-sol.p" )
		npt.assert_equal( solution.shaper()._classesIdToLabel, ["diamond", "drill", "ripple"] )
		npt.assert_equal( solution.model()[0], [256, 10, 3] )
		t = solution.model()[1];
		npt.assert_almost_equal( [np.sum( t[0] ), np.sum( t[1] ), np.sum( t[2] )], [0.14382  ,  0.0245429, -0.132043] )
		self.assertAlmostEqual( np.sum( solution.shaper().mu() ), 205.85431033692913 )
		self.assertAlmostEqual( np.sum( solution.shaper().sigma() ), 296.5719143203778 )
		fix_random()
		args.iterations = 9
		with CapturedStdout():
			mlak.train( args )
		with CapturedStdout() as out:
			mlak.test( args )
		self.assertEqual( out.getvalue(), "0.0\n" )
		solution = dio.load( "./out/struggle-sol.p" )
		npt.assert_equal( solution.shaper()._classesIdToLabel, ["diamond", "drill", "ripple"] )
		npt.assert_equal( solution.model()[0], [256, 10, 3] )
		t = solution.model()[1];
		npt.assert_almost_equal( [np.sum( t[0] ), np.sum( t[1] ), np.sum( t[2] )], [0.3095417, -0.0072933, -0.1675103] )
		self.assertAlmostEqual( np.sum( solution.shaper().mu() ), 205.85431033692913 )
		self.assertAlmostEqual( np.sum( solution.shaper().sigma() ), 296.5719143203778 )

	def test_struggle_20_test( self ):
		X, y = gen_regression_data()
		args = Args(
			data_set = "./out/struggle.p",
			solution = "./out/struggle-sol.p",
			engine = "struggle"
		)
		mlak.set_preferred_engine( args.engine )
		with CapturedStdout() as out:
			mlak.test( args )
		self.assertEqual( out.getvalue(), "0.0\n" )

	def test_keras_10_train( self ):
		X, y = gen_logistic_data()
		dio.save( "./out/keras.p", { "X": X, "y": y } )
		args = Args(
			data_set = "./out/keras.p",
			solution = "./out/keras-sol.p",
			topology = ["N(10)"],
			Lambda = "0.1",
			iterations = 10,
			engine = "keras"
		)
		mlak.set_preferred_engine( args.engine )
		with CapturedStdout():
			mlak.train( args )
		solution = dio.load( "./out/keras-sol.p" )

	def test_keras_20_test( self ):
		X, y = gen_regression_data()
		args = Args(
			data_set = "./out/keras.p",
			solution = "./out/keras-sol.p",
			engine = "keras"
		)
		mlak.set_preferred_engine( args.engine )
		with CapturedStdout() as out:
			mlak.test( args )
		self.assertEqual( out.getvalue(), "0.0\n" )

if __name__ == '__main__':
	unittest.main()

