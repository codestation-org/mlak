import math
import numpy as np
import scipy.optimize as optimize

import LinearAlgebra as la
import MathTools as mt
import ModelAnalyzer as ma

def apply_topology( topology, theta_ ):
	theta = []
	offset = 0
	for i in range( 1, len( topology ) ):
		rows = topology[i]
		cols = topology[i - 1] + 1
		s = rows * cols
		t = np.array( theta_[offset:offset + s] )
		t.shape = ( rows, cols )
		theta.append( t )
		offset += s
	return theta

def flatten_nn( theta ):
	return np.concatenate( list( map( lambda x : x.flatten(), theta ) ) )

def randomize_weights( topology, theta ):
	offset = 0
	ss = math.sqrt( 6 )
	for i in range( 1, len( topology ) ):
		rows = topology[i]
		cols = topology[i - 1] + 1
		s = rows * cols
		epsilon = ss / math.sqrt( rows + cols )
		theta[offset:offset + s] = np.random.rand( s ) * epsilon * 2 - epsilon
		offset += s

def compute_cost( theta, *args ):
	X, y, topology, classCount, Lambda = args
	theta = apply_topology( topology, theta )

	for l in range( len( theta ) ):
		X = la.add_ones_column( X )
		X = np.dot( X, theta[l].T )
		X = mt.sigmoid( X )

	m = len( y )
	yp = np.zeros( ( m, classCount ) )
	yp[np.arange( m ), y.flatten()] = 1
	cost = np.sum( -( yp * mt.log_v( X ) + ( 1 - yp ) * mt.log_v( 1 - X ) ) )
	r = 0
	for t in theta:
		r += np.sum( t[:,1:] ** 2 )
	cost /= m
	cost += Lambda * r / ( m * 2 )
	if "debug" in args:
		print( "cost = {}                                \r".format( cost ), end = "" )
	return cost

def activation_derivative( a ):
	return a * ( 1 - a )

def compute_grad( theta, *args ):
	X, y, topology, classCount, Lambda = args
	theta = apply_topology( topology, theta )
	thetaT = []
	for t in theta:
		thetaT.append( t.T )
	gradient = []
	for t in theta:
		gradient.append( np.zeros( t.shape ) )
	activation = []
	for t in thetaT:
		X = la.add_ones_column( X )
		activation.append( X )
		X = np.dot( X, t )
		X = mt.sigmoid( X )
	m = len( y )
	yp = np.zeros( ( m, classCount ) )
	yp[np.arange( m ), y.flatten()] = 1
	delta = ( X - yp ).T
	for g, t, a in zip( reversed( gradient ), reversed( thetaT ), reversed( activation ) ):
		g += np.dot( delta, a )
		delta = np.dot( t, delta ) * activation_derivative( a.T )
		delta = delta[1:]
	for g, t in zip( gradient, theta ):
		t[:, 0] = 0
		g += t * Lambda
		g /= m
	return flatten_nn( gradient )

def predict_one_vs_all( X, topoTheta ):
	topology, theta = topoTheta
	theta = apply_topology( topology, theta )
	y = []
	for i, x in enumerate( X ):
		for l in range( len( theta ) ):
			x = np.concatenate( ( [1], x ) )
			x = np.dot( theta[l], x )
			x = mt.sigmoid( x )
		y.append( np.argmax( x ) )
	return np.array( y )

class NeuralNetworkSolver:
	def __initial_theta( shaper, **kwArgs ):
		model = kwArgs.get( "model", None )
		if model is None:
			topology = kwArgs.get( "nnTopology", None )
			topology = list( map( int, topology.split( "," ) ) ) if topology else []
			topology = [shaper.feature_count()] + topology + [shaper.class_count()]
			s = 0
			for i in range( 1, len( topology ) ):
				s += topology[i] * ( topology[i - 1] + 1 )
			theta = np.zeros( s )
			randomize_weights( topology, theta )
			return topology, theta
		return model[0], model[1]

	def train( self_, X, y, **kwArgs ):
		shaper = ma.DataShaper( X, y, **kwArgs )
		iterations = kwArgs.get( "iterations", 50 )
		Lambda = kwArgs.get( "Lambda", 0 )
		y = shaper.map_labels( y )
		topology, theta = kwArgs.get( "theta", NeuralNetworkSolver.__initial_theta( shaper, **kwArgs ) )
		X = shaper.conform( X, addOnes = False )
		theta = optimize.fmin_cg(
			compute_cost,
			theta, fprime = compute_grad,
			args = ( X, y, topology, shaper.class_count(), Lambda ),
			maxiter = iterations,
			disp = False
		)
		return ma.Solution( model = ( topology, theta ), shaper = shaper )

	def verify( self_, solution, X, y ):
		X = solution.shaper().conform( X, addOnes = False )
		yp = predict_one_vs_all( X, solution.model() )
		accuracy = np.mean( 1.0 * ( y.flatten() == solution.shaper().labels( yp ) ) )
		return 1 - accuracy

	def predict( self_, solution, X ):
		X = solution.shaper().conform( X, addOnes = False )
		yp = predict_one_vs_all( X, solution.model() )
		return solution.shaper().labels( yp )


