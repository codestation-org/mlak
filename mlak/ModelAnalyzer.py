import numpy as np
from collections import namedtuple
import OptimizationAlgorithms as oa

Observations = namedtuple( "Observations", "X y" )
DataSet = namedtuple( "DataSet", "trainSet crossValidationSet testSet" )
Solution = namedtuple( "Solution", "theta accuracy Lambda" )

# Split data to train set, cross validation set and test set.
# *Fraction tells how much of the data shall go into given set.
def split_data( X, y, cvFraction, testFraction ):
	assert cvFraction + testFraction < 1
	m = np.size( X, 0 )
	cvSize = int( m * cvFraction )
	testSize = int( m * testFraction )
	trainSize = m - ( cvSize + testSize )

	perm = np.random.permutation( m )

	XTrain = X[perm[ : trainSize]]
	yTrain = y[perm[ : trainSize]]

	XCV = X[perm[trainSize : trainSize + cvSize]]
	yCV = y[perm[trainSize : trainSize + cvSize]]

	XTest = X[perm[trainSize + cvSize : ]]
	yTest = y[perm[trainSize + cvSize : ]]

	return DataSet( Observations( XTrain, yTrain ), Observations( XCV, yCV ), Observations( XTest, yTest ) )

class SolverBase:
	def __init__( self_, X, y, **kwArgs ):
		cvFrequency = kwArgs.get( "cvFrequency", 0.2 )
		testFrequency = kwArgs.get( "testFrequency", 0.2 )

		self_._iterations = kwArgs.get( "iters", 50 )
		self_._X = X
		self_._y = y

		self_._XNormalized, self_._mu, self_._sigma = oa.feature_normalize( self_._X )

		m = np.size( self_._X, 0 )
		self_._XWithOnes = np.c_[ np.ones( m ), self_._XNormalized ]
		self_._dataSet = split_data( self_._XWithOnes, self_._y, cvFrequency, testFrequency )

	def solve( self_, Lambda, **kwArgs ):
		raise Exception( "Method `solve` is not implemented!" )

	def test( self_, solution ):
		raise Exception( "Method `test` is not implemented!" )

def find_solution( solver, **kwArgs ):
	lambdaRange = kwArgs.get( "lambdaRange", ( 0, 100 ) )
	lambdaQuality = kwArgs.get( "lambdaQuality", 1 )
	lo = lambdaRange[0]
	hi = lambdaRange[1]
	Lambda = lo + ( hi - lo ) / 2
	accuracy = {}
	get_solution = lambda x : accuracy.get( x ) if x in accuracy else accuracy.setdefault( x, solver.solve( x, **kwArgs ) )
	solution = None
	old = []
	while hi - lo > lambdaQuality:
		sLo = get_solution( lo )
		sHyper = get_solution( Lambda )
		sHi = get_solution( hi )
		hMax = np.argmin( [sLo.accuracy, sHyper.accuracy, sHi.accuracy] )
		hMin = np.argmax( [sLo.accuracy, sHyper.accuracy, sHi.accuracy] )
		if old == [lo, Lambda, hi]: #unstable
			Lambda, solution = ( lo, sLo ) if hMin == 0 else ( hi, sHi )
			break
		old = [lo, Lambda, hi]
		if hMin == 0:
			hi = Lambda
		elif hMin == 2:
			lo = Lambda
		else:
			loT = lo
			hiT = hi
			while hiT - loT > 0.01:
				loT = np.mean( [loT, Lambda] )
				hiT = np.mean( [Lambda, hiT] )
				sLoT = get_solution( loT )
				sHiT = get_solution( hiT )
				hMin = np.argmax( [sLoT.accuracy, sHyper.accuracy, sHiT.accuracy] )
				if hMin == 0:
					hi = Lambda
					break
				elif hMin == 2:
					lo = Lambda
					break
		Lambda = np.mean( [lo, hi] )
		solution = sHyper
	solution = Solution( solution.theta, solver.test( solution ), solution.Lambda )
	return solution
