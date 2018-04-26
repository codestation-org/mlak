def bins( count, top ):
	step = top // count
	b = list( range( top, 0, -step ) )[::-1]
	while ( b[0] > count ) or ( ( len( b ) > 1 ) and ( ( b[1] - b[0] ) > count ) ):
		if b[0] > count:
			b = bins( count, b[0] ) + b
		else:
			b = bins( count, b[1] ) + b[1:]
	return sorted( list( set( b ) ) )

def is_sorted( l, key = lambda x: x, reverse = False ):
	s = False
	if reverse:
		s = all( [ key( l[i] ) >= key( l[i + 1] ) for i in range( len( l ) - 1 ) ] )
	else:
		s = all( [ key( l[i] ) <= key( l[i + 1] ) for i in range( len( l ) - 1 ) ] )
	return s

import warnings

class NoWarnings( object ):
	def __init__( self_ ):
		self_._warnSave = warnings.warn
	def __enter__( self_ ):
		warnings.warn = lambda *arga, **kwArgs: None
	def __exit__( self_, type_, value_, traceback_ ):
		warnings.warn = self_._warnSave

import io
import sys

class CapturedStdout( object ):
	def __init__( self_ ):
		self_._stream = io.StringIO()
	def __enter__( self_ ):
		sys.stdout = self_._stream
		return self_._stream
	def __exit__( self_, type_, value_, traceback_ ):
		sys.stdout = sys.__stdout__

import inspect

def func_to_str( f ):
	return inspect.getsource( f ).replace( "\t", "" ).replace( "\n", "" )

with NoWarnings():
	import numpy as np
	import random as rn
	import tensorflow as tf

import os

def fix_random():
	np.random.seed( 0 )
	rn.seed( 0 )
	tf.set_random_seed( 0 )
	os.environ['PYTHONHASHSEED'] = '0'
	sessionConf = tf.ConfigProto( intra_op_parallelism_threads = 1, inter_op_parallelism_threads = 1, allow_soft_placement = True, device_count = { 'CPU': 1 } )
	from keras import backend as K
	sess = tf.Session( graph = tf.get_default_graph(), config = sessionConf )
	K.set_session( sess )
	np.random.seed( 0 )
	rn.seed( 0 )
	tf.set_random_seed( 0 )

