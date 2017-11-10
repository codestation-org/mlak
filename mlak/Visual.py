from math import sqrt
import numpy as np

from tkinter import *
from PIL import Image, ImageDraw, ImageTk

import Terminal as term

class DrawingPad:
	imgSize = 480
	width = 40
	def __init__( self_, **kwArgs ):
		self_._b1 = "up"
		self_._xold = None
		self_._yold = None
		self_._img = None
		self_._imgTk = None
		self_._solver = kwArgs.get( "solver", None )
		self_._solution = kwArgs.get( "solution", None )
		self_._size = kwArgs.get( "size", int( sqrt( self_._solution.theta()[0][0] ) ) if self_._solution else None )
		self_._labels = kwArgs.get( "labels", None )
		self_._repeat = kwArgs.get( "repeat", 0 )
		self_._i = 0

	def run( self_ ):
		self_._root = Tk()
		drawing_area = Canvas( self_._root, width = DrawingPad.imgSize, height = DrawingPad.imgSize, bg = 'white' )
		drawing_area.pack()
		self_._img = Image.new( "L", ( DrawingPad.imgSize, DrawingPad.imgSize ), 255 )
		drawing_area.pack()
		drawing_area.bind( "<Motion>", self_.motion )
		drawing_area.bind( "<ButtonPress-1>", self_.b1down )
		drawing_area.bind( "<ButtonRelease-1>", self_.b1up )
		drawing_area.bind( "<ButtonPress-3>", self_.b3down )
		if self_._labels:
			self_._data = { "X": [], "y": [] }
			print( "{}".format( self_.current_label() ) )

		self_._root.mainloop()

	def b1down( self_, event ):
		self_._b1 = "down"	# you only want to draw when the button is down
								# because "Motion" events happen -all the time-

	def b3down( self_, event ):
		if self_._solution:
			self_.predict()
		elif self_._labels:
			self_.make_sample()
		event.widget.delete( "all" )
		self_._img = Image.new( "L", ( DrawingPad.imgSize, DrawingPad.imgSize ), 255 )

	def b1up( self_, event ):
		self_._b1 = "up"
		self_._xold = None		   # reset the line when you let go of the button
		self_._yold = None

	def motion( self_, event ):
		if self_._b1 == "down":
			if self_._xold is not None and self_._yold is not None:
				draw = ImageDraw.Draw( self_._img )
				draw.line( [ self_._xold, self_._yold, event.x, event.y ], 0, width = DrawingPad.width )
				draw.ellipse(
					( self_._xold - DrawingPad.width / 2, self_._yold - DrawingPad.width / 2, self_._xold + DrawingPad.width / 2, self_._yold + DrawingPad.width / 2 ),
					fill = 0
				)
				draw.ellipse( ( event.x - DrawingPad.width / 2, event.y - DrawingPad.width / 2, event.x + DrawingPad.width / 2, event.y + DrawingPad.width / 2 ), fill = 0 )
				self_._imgTk = ImageTk.PhotoImage( self_._img )
				event.widget.create_image( 0, 0, anchor = "nw", image = self_._imgTk )
				event.widget.pack()
			self_._xold = event.x
			self_._yold = event.y

	def get_drawing( self_ ):
		return 255 - np.asarray( self_._img.resize( ( self_._size, self_._size ), Image.ANTIALIAS ) ).flatten()

	def predict( self_ ):
		d = self_.get_drawing()
		term.plot( d, art = True, label = "{}".format( self_._solver.predict( self_._solution, np.array( [ d ] ) )[0] ) )

	def make_sample( self_ ):
		d = self_.get_drawing()
		term.plot( d, art = True )
		self_._data["X"].append( d )
		self_._data["y"].append( self_.current_label() )
		self_._i += 1
		if self_._i >= ( len( self_._labels ) * self_._repeat ):
			self_._root.quit()
			self_._data["X"] = np.array( self_._data["X"] )
			self_._data["y"] = np.array( self_._data["y"] )
			return
		print( "{}".format( self_.current_label() ) )

	def current_label( self_ ):
		l = self_._labels[self_._i // self_._repeat]
		return l
