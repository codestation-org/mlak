#! /usr/bin/python3

import sys
sys.path.extend( [ "./mlak" ] )

import unittest

from Logger import Logger
from re import sub
from glob import glob
from os import remove

class TestLogger( unittest.TestCase ):
	def setUp( self ):
		for f in glob( "out/*.log" ):
			remove( f )

	def tearDown( self ):
		self.setUp()

	def do_test_line( self, line ):
		cuts = [ "date", "changes", "hash" ]
		for cut in cuts:
			line = str( sub( "(?<={}\": )\"[^\"]*\"".format( cut ), "\"\"", line ) )
		self.assertEqual(
			line,
			'{"date": "", "git": {"currenthash": "", "changes": ""}, "data": {"data": "value"}, "files": [{"filename": "out/data.p", "hash": ""}, {"filename": "out/data.mat", "hash": ""}]}\n'
		)

	def test_Logger_log_split_false( self ):
		Logger.log( { "data": "value" }, ["out/data.p", "out/data.mat"], log_dir = "out", log_file_name = "test" )
		Logger.log( { "data": "value" }, ["out/data.p", "out/data.mat"], log_dir = "out", log_file_name = "test" )
		with open( "out/test.log" ) as f:
			line = f.readline()
			l2 = f.readline()
			self.assertEqual( line, l2 )
			self.do_test_line( line )

	def test_Logger_log_split_true( self ):
		Logger.log( { "data": "value" }, ["out/data.p", "out/data.mat"], log_dir = "out", log_file_name = "test", split = True )
		Logger.log( { "data": "value" }, ["out/data.p", "out/data.mat"], log_dir = "out", log_file_name = "test", split = True )
		dt = Logger.get_unique_log_filename()[:14]
		files = glob( "out/" + dt + "-*.log" )
		d = [ None, None ]
		for i, p in enumerate( files ):
			with open( p ) as f:
				d[i] = f.read()
		self.assertEqual( d[0], d[1] )
		self.do_test_line( d[0] )

if __name__ == '__main__':
	unittest.main()

