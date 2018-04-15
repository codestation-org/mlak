#! /usr/bin/python3

import sys
sys.path.extend( [ "./mlak" ] )

import unittest
from mlak.utils import CapturedStdout

from Terminal import *

class TestTerminal( unittest.TestCase ):
	def test_plot( self ):
		X = np.array( [
			0,  0,  0,  0,  1,  0,  0,  0,
			0,  0,  0,  1, 18,  0,  0,  0,
			0,  0,  0, 14,  1, 13,  0,  0,
			0,  0,  1, 12,  1, 11,  0,  0,
			0,  0, 10,  0,  0,  9,  1,  0,
			1,  2,  3,  8,  7,  3,  2,  1,
			0,  5,  1,  0,  0,  1,  4,  0,
			2,  1,  0,  0,  0,  0,  2,  1
		] )
		with CapturedStdout() as numerical:
			plot( X, label = "A" )
		with CapturedStdout() as art:
			plot( X, art = True, label = "A" )

		numericalExp = """+-------------------------+
| Label: A                |
| neutral = 0, range = 18.0|
| min = 0.0, max = 18.0   |
+-------------------------+
|  0  0  0  0  5  0  0  0 |
|  0  0  0  5 99  0  0  0 |
|  0  0  0 77  5 71  0  0 |
|  0  0  5 66  5 60  0  0 |
|  0  0 55  0  0 49  5  0 |
|  5 11 16 44 38 16 11  5 |
|  0 27  5  0  0  5 22  0 |
| 11  5  0  0  0  0 11  5 |
+-------------------------+
"""
		artExp = """+---------+
| Label: A|
| neutral = 0|
| range = 18.0|
| min = 0.0|
| max = 18.0|
+---------+
|    .    |
|   .@    |
|   @.@   |
|  .@.@   |
|  @  @.  |
|.++@#++. |
| Q.  .*  |
|+.    +. |
+---------+
"""
		numerical = numerical.getvalue()
		art = art.getvalue()
		self.assertEqual( numerical, numericalExp )
		self.assertEqual( art, artExp )

	def test_progress( self ):
		with CapturedStdout() as out:
			p = Progress( 5, "test: " )
			next( p )
			next( p )
			next( p )
			next( p )
			next( p )
			next( p )
			next( p )
			next( p )
			next( p )
		out = out.getvalue()
		exp = "test:  20.00%             \rtest:  40.00%             \rtest:  60.00%             \rtest:  80.00%             \rtest: 100%             \r\n"
		self.assertEqual( out, exp )


if __name__ == '__main__':
	unittest.main()

