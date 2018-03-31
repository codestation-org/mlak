#! /usr/bin/python3

import sys
sys.path.extend( [ "./mlak" ] )

import unittest
import numpy.testing as npt

from DataIO import load, save, write_text_to_file, append_text_to_file, get_file_md5
import numpy as np

class TestDataIO( unittest.TestCase ):
	def test_10_write_text_to_file( self ):
		write_text_to_file( "out/data.txt", "1, 2, 3, .4\n" )

	def test_20_append_text_to_file( self ):
		append_text_to_file( "out/data.txt", "4, 5, 6, .3\n" )
		append_text_to_file( "out/data.txt", "7, 8, 9, .2\n" )

	def test_30_get_file_md5( self ):
		self.assertEqual( get_file_md5( "out/data.txt" ), "d7406c51cd65275203fec6355a296fcd" )

	X = np.array( [ [1, 2, 3], [4, 5, 6], [7, 8, 9] ] )
	y = np.array( [ [.4], [.3], [.2] ] )
	def test_40_save( self ):
		data = { "X": TestDataIO.X, "y": TestDataIO.y }
		save( "out/data.p", data )
		save( "out/data.mat", data )
		with self.assertRaises( Exception ):
			save( "out/data.xx", data )

	def test_50_load( self ):
		data = load( "out/data.p" )
		self.assertIn( "X", data )
		self.assertIn( "y", data )
		npt.assert_almost_equal( data["X"], TestDataIO.X )
		npt.assert_almost_equal( data["y"], TestDataIO.y )
		data = load( "out/data.mat" )
		self.assertIn( "X", data )
		self.assertIn( "y", data )
		npt.assert_almost_equal( data["X"], TestDataIO.X )
		npt.assert_almost_equal( data["y"], TestDataIO.y )
		data = load( "out/data.txt" )
		self.assertIn( "X", data )
		self.assertIn( "y", data )
		npt.assert_almost_equal( data["X"], TestDataIO.X )
		npt.assert_almost_equal( data["y"], TestDataIO.y )
		with self.assertRaises( Exception ):
			load( "out/data.xx" )

if __name__ == '__main__':
	unittest.main()

