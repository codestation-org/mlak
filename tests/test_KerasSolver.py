#! /usr/bin/python3

import sys
sys.path.extend( [ "./mlak", "./tests" ] )

from itertools import product
import unittest
import numpy.testing as npt

from KerasSolver import *
from ModelAnalyzer import *
import DataIO as dio
from data_gen import *
import numpy as np

def desc( conf, debug = False ):
	if debug:
		print()
	d = ""
	for i in conf:
		if debug:
			print( i )
		if d:
			d += " "
		cn = i["class_name"]
		c = i["config"]
		d += cn
		if cn == "Flatten" and "batch_input_shape" in c:
			d += ":" + str( c["batch_input_shape"] )
		elif cn == "Dense":
			d += ":" + "({}, {})".format( c['units'], c['activation'] )
		elif cn == "Conv2D":
			d += ":" + "({}, {})".format( c["filters"], c["kernel_size"] )
		elif cn == "Dropout":
			d += ":" + str( c["rate"] )
		elif cn == "MaxPooling2D":
			d += ":" + str( c["pool_size"] )
	return d

class TestKerasSolver( unittest.TestCase ):
	X, y = gen_logistic_data()
	def test_prepare_model( self ):
		solver = KerasSolver()
		shaper = DataShaper( self.X )
		model = KerasSolver._KerasSolver__prepare_model( shaper, self.y, nnTopology = "N(10)" )
		self.assertEqual( desc( model.get_config() ), "Flatten:(None, 16, 16, 1) Dense:(10, linear) Dense:(3, softmax)" )
		model = KerasSolver._KerasSolver__prepare_model( shaper, self.y, nnTopology = "C(4, 2, 2),N(8)" )
		self.assertEqual( desc( model.get_config() ), "Conv2D:(4, (2, 2)) Flatten Dense:(8, linear) Dense:(3, softmax)" )
		model = KerasSolver._KerasSolver__prepare_model( shaper, self.y, nnTopology = "C(4, 2, 2),D(0.5),N(8)" )
		self.assertEqual( desc( model.get_config() ), "Conv2D:(4, (2, 2)) Dropout:0.5 Flatten Dense:(8, linear) Dense:(3, softmax)" )
		model = KerasSolver._KerasSolver__prepare_model( shaper, self.y, nnTopology = "C(4, 2, 2),MP(2,2),D(0.5),N(8,activation=relu)" )
		self.assertEqual( desc( model.get_config() ), "Conv2D:(4, (2, 2)) MaxPooling2D:(2, 2) Dropout:0.5 Flatten Dense:(8, relu) Dense:(3, softmax)" )
		model = KerasSolver._KerasSolver__prepare_model( shaper, self.y, nnTopology = "C(4, 2, 2),MP(2,2),D(0.5),F,N(8)" )
		self.assertEqual( desc( model.get_config() ), "Conv2D:(4, (2, 2)) MaxPooling2D:(2, 2) Dropout:0.5 Flatten Dense:(8, linear) Dense:(3, softmax)" )
		model = KerasSolver._KerasSolver__prepare_model( shaper, self.y, nnTopology = "" )
		self.assertEqual( desc( model.get_config() ), "Flatten:(None, 16, 16, 1) Dense:(3, softmax)" )

	def test_make_keras_pickable( self ):
		make_keras_picklable()
		solver = KerasSolver()
		self.assertEqual( solver.type(), ma.SolverType.CLASSIFIER )
		dio.save( "./out/sol.p", solver.train( self.X, self.y, nnTopology = "N(10)", Lambda = 1, kerasArgs = { "verbose": 0 } ) )
		solution = dio.load( "./out/sol.p" )
		self.assertEqual( solution.shaper().class_count(), 3 )
		cost = solver.verify( solution, self.X, self.y )
		self.assertEqual( cost, 0 )
		m = len( self.y )
		Xt = np.array( [self.X[0], self.X[m // 2], self.X[-1]] )
		yp = solver.predict( solution, Xt )
		npt.assert_equal( yp, ["ripple", "diamond", "drill"] )

if __name__ == '__main__':
	unittest.main()

