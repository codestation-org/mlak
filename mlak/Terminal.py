import numpy as np
import math
from collections import Counter

def plot( x_, width_ = None ):
	x = np.array( x_, dtype = float )
	if width_ == None:
		width_ = int( math.sqrt( len( x ) ) )
	sup = max( x )
	inf = min( x )
	r = sup - inf
	x *= 99
	x /= r
	x = np.array( x, dtype = int )
	neutral = Counter( x ).most_common()[0][0]
	print( "range = {}, neutral = {}, min = {}, max = {}".format( r, neutral, inf, sup ) )
	for i in range( len( x ) ):
		print( "{: 3d}".format( x[i] ), end = "" )
		if ( i + 1 ) % width_ == 0:
			print( "" )

def Progress( n ):
	p = 0
	op = 0
	for i in range( n ):
		p = int( i * 10000 / n )
		if p != op:
			print( "{:6.2f}%        \r".format( p / 100 ), end = "" )
			op = p
		yield
