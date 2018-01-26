#! /usr/bin/python3
import sys
import os

import traceback

import argparse
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

import mlak.LinearRegression as linReg
import mlak.LogisticRegression as logReg
import mlak.NeuralNetwork as nn
import mlak.ModelAnalyzer as ma

import mlak.DataIO as dio
import mlak.Terminal as term
import mlak.Visual as visual

preferredEngine = None
def set_preferred_engine( engine ):
	global preferredEngine
	if engine == "keras":
		import mlak.KerasSolver as ks
		preferredEngine = ks.KerasSolver
		ks.make_keras_picklable()
	elif engine == "linreg":
		preferredEngine = linReg.LinearRegressionSolver
	elif engine == "logreg":
		preferredEngine = logReg.LogisticRegressionSolver
	elif engine == "struggle":
		preferredEngine = nn.NeuralNetworkSolver
	else:
		raise Exception( "Bad engine!" )
	return engine

def predict( solver, solution, X, y, n ):
	x = X[n]
	y = y[n]
	yp = solver.predict( solution, np.array( [ x ] ) )[0]
	if yp != y:
		term.plot( x, art = True, label = "n:{} p:{} e:{}".format( n, yp, y ) )
		return input( "Press `Enter` to continue or `q` followed by `Enter` to quit: " ) != "q"
	return True

def create( args ):
	labels = list( args.classes.split( "," ) )
	lifeDemo = visual.DrawingPad( labels = labels, repeat = args.samples, size = args.size )
	lifeDemo.run()
	dio.save( args.data_set, lifeDemo._data )

def train( args ):
	solver = preferredEngine()
	topology = args.topology
	Lambda = list( map( float, args.Lambda.split( "," ) ) )
	rawData = dio.load( args.data_set )
	X_orig = rawData["X"]
	y_orig = rawData["y"]
	args.engine = solver.__class__.__name__
	optimizationResults = ma.find_solution(
		solver, X_orig, y_orig,
		showFailureRateTrain = True,
		optimizationParams = {
			"nnTopology": topology,
			"Lambda": Lambda,
			"functions": [
			]
		},
		files = [ args.data_set ],
		logFileName = "mlak.log",
		**vars( args )
	)
	solution = optimizationResults.solution
	dio.save( args.solution, solution )
	if args.debug:
		print( "solution = {}".format( solution ) )
	if solution.shaper().is_classifier():
		for i in range( len( y_orig ) ):
			if not predict( solver, solution, X_orig, y_orig, i ):
				return

def analyze( args ):
	solver = preferredEngine()
	rawData = dio.load( args.data_set )
	X_orig = rawData["X"]
	y_orig = rawData["y"]
	args.engine = solver.__class__.__name__
	analyzerResults = ma.analyze(
		solver, X_orig, y_orig,
		optimizationParams = {
			"nnTopology": args.topology,
			"Lambda": args.Lambda,
			"functions": None
		},
		files = [ args.data_set ],
		**vars( args )
	)
	fig = plt.figure( 1 )
	plt.rc( 'grid', linestyle = ":", color = 'gray' )
	plt.subplot( 211 )
	plt.title( "data: {}\nlambda: {}\n\n".format( args.data_set, args.Lambda ), loc = "left" )
	plt.title( "Sample count test\n" )
	plt.xlabel( "Sample count" )
	plt.ylabel( "Error rate" )
	plt.plot( analyzerResults.sampleCountAnalyzis.sampleCount, analyzerResults.sampleCountAnalyzis.errorTrain, 'b-', label = "train" )
	plt.plot( analyzerResults.sampleCountAnalyzis.sampleCount, analyzerResults.sampleCountAnalyzis.errorCV, 'g-', label = "CV" )
	plt.grid()
	plt.legend()
	plt.subplot( 212 )
	plt.title( "Iteration count test" )
	plt.xlabel( "Iteration count" )
	plt.ylabel( "Error rate" )
	plt.plot( analyzerResults.iterationCountAnalyzis.iterationCount, analyzerResults.iterationCountAnalyzis.errorTrain, 'b-', label = "train" )
	plt.plot( analyzerResults.iterationCountAnalyzis.iterationCount, analyzerResults.iterationCountAnalyzis.errorCV, 'g-', label = "CV" )
	plt.grid()
	plt.legend()
	def on_resize( event ):
		plt.tight_layout()
		plt.subplots_adjust( right = 0.95 )
	cid = fig.canvas.mpl_connect( "resize_event", on_resize )
	plt.show()

def test( args ):
	solver = preferredEngine()
	solution = dio.load( args.solution )
	rawData = dio.load( args.data_set )
	X_orig = rawData["X"]
	y_orig = rawData["y"]
	for i in range( len( y_orig ) ):
		if not predict( solver, solution, X_orig, y_orig, i ):
			return

def show( args ):
	solver = preferredEngine()
	solution = None
	yp = None

	rawData = dio.load( args.data_set )
	X_orig = rawData["X"]
	y_orig = rawData["y"]

	if args.solution:
		solution = dio.load( args.solution )
		yp = solver.predict( solution, X_orig )

	se = visual.SampleEditor( X_orig, y_orig, zoom = args.zoom, yPredict = yp )
	se.run()

def live( args ):
	solver = preferredEngine()
	solution = dio.load( args.solution )
	lifeDemo = visual.DrawingPad( solver = solver, solution = solution, speech = args.speech )
	lifeDemo.run()

def main():
	parser = argparse.ArgumentParser(
		description = "Machine Learning Army Knife - an experimentation tool\n"
			"\n"
			"Valid engines are:\n"
			"keras    - Keras based engine (default)\n"
			"struggle - native Neural Network implementation with fully connected layers\n"
			"logreg   - Logisic Regression\n"
			"linreg   - Linear Regression\n"
			"\n"
			"Example invocations:\n"
			"  {0} create -c classA,classB,classC -n 30 -r 28 -d train_data_ABC.p\n"
			"  {0} train -d train_data.p -s solution_ABC.p -i 100 -t 40,25 -l 1,3,10\n"
			"  {0} test -d train_data.p -s solution_ABC.p\n"
			"  {0} analyze -d train_data.p -r 10 -s 1.5 -l 0.1\n"
			"  {0} show -d train_data.p -z 3 -s solution_ABC.p\n"
			"  {0} live -s solution_ABC.p".format( os.path.basename( sys.argv[0] ) ),
		formatter_class = argparse.RawTextHelpFormatter
	)
	subparsers = parser.add_subparsers()

	parserCreate = subparsers.add_parser( "create", help = "Prepare new batch of training samples." )
	parserCreate.add_argument( "-d", "--data-set", metavar = "path", type = str, required = True, help = "Output file for created data set." )
	parserCreate.add_argument( "-c", "--classes", metavar = "labels", type = str, required = True, help = "List of classes to be added to new data set." )
	parserCreate.add_argument( "-n", "--samples", metavar = "num", type = int, required = True, help = "Number of samples per class." )
	parserCreate.add_argument( "-r", "--size", metavar = "size", type = int, required = True, help = "Single sample resolution." )
	parserCreate.set_defaults( func = create )

	parserTrain = subparsers.add_parser( "train", help = "Train given model on given data." )
	parserTrain.add_argument( "-d", "--data-set", metavar = "path", type = str, required = True, help = "Dataset for training." )
	parserTrain.add_argument( "-s", "--solution", metavar = "path", type = str, required = True, help = "Store solution path." )
	parserTrain.add_argument( "-t", "--topology", metavar = "topo", type = str, action = "append", help = "NeuralNetwork topologies to test." )
	parserTrain.add_argument( "-l", "--Lambda", metavar = "lambda", type = str, required = True, help = "Values of regularization parameter to test." )
	parserTrain.add_argument( "-i", "--iterations", metavar = "num", type = int, help = "Maximum number of iterations." )
	parserTrain.set_defaults( iterations = 50 )
	parserTrain.set_defaults( func = train )

	parserAnalyze = subparsers.add_parser( "analyze", help = "Analyze given model architecture." )
	parserAnalyze.add_argument( "-d", "--data-set", metavar = "path", type = str, required = True, help = "Dataset for training." )
	parserAnalyze.add_argument( "-t", "--topology", metavar = "topo", type = str, help = "NeuralNetwork topology to test." )
	parserAnalyze.add_argument( "-l", "--Lambda", metavar = "lambda", type = float, required = True, help = "Value of regularization parameter to test." )
	parserAnalyze.add_argument( "-r", "--tries", metavar = "count", type = int, help = "Average out experiment results over that many retries." )
	parserAnalyze.add_argument( "-S", "--step", metavar = "factor", type = float, help = "Sample count increment factor." )
	parserAnalyze.add_argument( "-I", "--sample-iterations", metavar = "num", type = int, help = "Number of iterations in sample count trial." )
	parserAnalyze.add_argument( "-i", "--iterations", metavar = "num", type = int, help = "Maximum number of iterations." )
	parserAnalyze.set_defaults( func = analyze )

	parserTest = subparsers.add_parser( "test", help = "Test trained model against given data." )
	parserTest.add_argument( "-d", "--data-set", metavar = "path", type = str, required = True, help = "Dataset to test a model against." )
	parserTest.add_argument( "-s", "--solution", metavar = "path", type = str, required = True, help = "Path to solution to test." )
	parserTest.set_defaults( func = test )

	parserShow = subparsers.add_parser( "show", help = "Show sample data, optionally with invalid predictions from model." )
	parserShow.add_argument( "-d", "--data-set", metavar = "path", type = str, required = True, help = "Dataset for training." )
	parserShow.add_argument( "-s", "--solution", metavar = "path", type = str, help = "Load solution path." )
	parserShow.add_argument( "-z", "--zoom", metavar = "level", type = int, help = "Zoom level." )
	parserShow.set_defaults( zoom = 1 )
	parserShow.set_defaults( func = show )

	parserLive = subparsers.add_parser( "live", help = "Run live test for given solution." )
	parserLive.add_argument( "-s", "--solution", metavar = "path", type = str, required = True, help = "Solution to use for live test." )
	parserLive.set_defaults( func = live )


	parser.add_argument(
		"-e", "--engine", metavar = "name", choices = [ "struggle", "keras", "linreg", "logreg" ], type = set_preferred_engine,
		action = "store",
		help = "Choose machine learning engine."
	)
	parser.add_argument( "-s", "--speech", help = "Provide voice feedback for predictions.", action = 'store_true' )
	parser.add_argument( "-v", "--verbose", help = "Increase program verbosity level.", action = 'store_true' )
	parser.add_argument( "-D", "--debug", help = "Print all debuging information.", action = 'store_true' )
	parser.set_defaults( engine = "keras" )
	parser.set_defaults( speech = False )
	parser.set_defaults( verbose = False )
	parser.set_defaults( debug = False )
	parser.set_defaults( func = lambda x : parser.print_help() )
	args = parser.parse_args()
	args.func( args )
	return

if __name__ == "__main__":
	try:
		main()
	except Exception:
		traceback.print_exc( file = sys.stdout )
		sys.exit( 1 )

