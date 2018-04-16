import numpy as np
import math
from collections import Counter

def plot( x_, label = None, art = False, **kwArgs ):
	width = x_.shape[0]
	x = np.array( x_.flatten(), dtype = float )
	if width == len( x ):
		width = int( math.sqrt( width ) )
	width = kwArgs.get( "width", width )
	sup = max( x )
	inf = min( x )
	r = sup - inf
	x *= 99
	x /= r
	x = np.array( x, dtype = int )
	neutral = Counter( x ).most_common()[0][0]
	pixelWidth = 3
	fmt = lambda x : "{: 3d}".format( x )
	def artForm( x ):
		x = abs( x )
		if x < 3:
			x = " "
		elif x < 10:
			x = "."
		elif x < 20:
			x = "+"
		elif x < 25:
			x = "*"
		elif x < 30:
			x = "Q"
		elif x < 40:
			x = "#"
		else:
			x = "@"
		return x
	if art:
		pixelWidth = 1
		fmt = artForm
	barLen = ( width * pixelWidth + 1 )
	print( "+" + "-" * barLen + "+" )
	txt = []
	if label:
		txt.append( "Label: {}".format( label ) )
	if art:
		txt.append( "neutral = {}".format( neutral ) )
		txt.append( "range = {}".format( round( r, 8 ) ) )
		txt.append( "min = {}".format( round( inf, 10 ) ) )
		txt.append( "max = {}".format( round( sup, 10 ) ) )
	else:
		txt.append( "neutral = {}, range = {}".format( neutral, r ) )
		txt.append( "min = {}, max = {}".format( inf, sup ) )
	for s in txt:
		print( "| " + s + " " * ( barLen - len( s ) - 1 ) + "|" )
	print( "+" + "-" * barLen + "+" )
	for i in range( len( x ) ):
		if i % width == 0:
			print( "|", end = "" )
		print( fmt ( x[i] ), end = "" )
		if ( i + 1 ) % width == 0:
			print( " |" )
	print( "+" + "-" * barLen + "+" )

class Progress( object ):
	def __init__( self_, n_, label_ = "" ):
		self_._max = n_
		self_._label = label_
		self_._cur = 0
		self_._display = 0
		self_.cll = " " * ( len( self_._label ) + 7 ) + "\r"
	def next( self_ ):
		if self_._cur < self_._max:
			p = int( self_._cur * 10000 / self_._max )
			if p != self_._display:
				print( "{}{:6.2f}%{}".format( self_._label, p / 100, self_.cll ), end = "" )
				self_._display = p
			self_._cur += 1
		else:
			self_.done( end = "" )
	def done( self_, end = None ):
		print( "{}100%{}".format( self_._label, self_.cll ), end = end )

