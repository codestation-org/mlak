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

import types
from copy import deepcopy

def __stringify( data ):
	if type( data ) == list:
		for i in range( len( data ) ):
			data[i] = __stringify( data[i] )
	elif type( data ) == dict:
		for k in data:
			data[k] = __stringify( data[k] )
	elif isinstance( data, types.FunctionType ):
		data = func_to_str( data )
	return data

def stringify( data ):
	return __stringify( deepcopy( data ) )
