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

def Progress( n, label = "" ):
	p = 0
	op = 0
	cll = " " * ( len( label ) + 7 ) + "\r"
	for i in range( n ):
		p = int( i * 10000 / n )
		if p != op:
			print( "{}{:6.2f}%{}".format( label, p / 100, cll ), end = "" )
			op = p
		yield
	print( "{}100%{}".format( label, cll ) )
	yield
	while True:
		yield

