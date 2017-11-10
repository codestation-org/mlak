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

class SampleEditor:
	def __init__( self_, X, y, **kwArgs ):
		self_._X = X
		self_._y = y
		self_._yPredict = kwArgs.get( "yPredict", y )
		self_._winWidth = 480
		self_._winHeight = self_._winWidth
		self_._sampleSize = kwArgs.get( "size", int( sqrt( len( self_._X[0] ) ) ) )
		self_._zoom = kwArgs.get( "zoom", 1 )
		print( "sampleSize = {}".format( self_._sampleSize ) )

	def run( self_ ):
		self_._root = Tk()
		self_._canvas = Canvas( self_._root, width = self_._winWidth, height = self_._winHeight, bg = 'white' )
		vbar = Scrollbar( self_._root, orient = VERTICAL )
		vbar.pack( side = RIGHT, fill = Y )
		vbar.config( command = self_._canvas.yview )
		self_._canvas.pack( side = LEFT, expand = YES, fill = BOTH )
		#self_._img = Image.new( "L", ( self_._winWidth, self_._winWidth ), 255 )
		self_._canvas.bind( "<Configure>", self_.on_resize )
		self_._canvas.bind( "<Motion>", self_.on_motion )
		self_._canvas.bind( "<ButtonPress-1>", self_.on_b1down )
		self_._canvas.bind( "<MouseWheel>", self_.on_mousewheel )
		self_._canvas.bind( "<ButtonPress-4>", self_.on_mousewheel )
		self_._canvas.bind( "<ButtonPress-5>", self_.on_mousewheel )
		self_._canvas.config( yscrollcommand = vbar.set, yscrollincrement = self_._sampleSize / 2 * self_._zoom )
		self_._imgs = []
		for x in self_._X:
			x = self_.normalize( x )
			img = ImageTk.PhotoImage(
				image = Image.fromarray( x, mode = "L" ).resize( ( self_._sampleSize * self_._zoom, self_._sampleSize * self_._zoom ), Image.ANTIALIAS )
			)
			self_._imgs.append( img )
		self_.paint()
		self_._root.mainloop()

	def paint( self_ ):
		borderWidth = 2
		ss = self_._sampleSize * self_._zoom + borderWidth
		ssy = ss
		noSamples = len( self_._y )
		columns = self_._winWidth // ss
		rows = noSamples // columns
		realHeight = rows * ssy
		self_._canvas.delete( "all" )
		self_._canvas.config( scrollregion = ( 0, 0, columns * ss, realHeight ) )
		for c in range( columns ):
			self_._canvas.create_line( c * ss, 0, c * ss, realHeight, fill = 'gray', width = borderWidth )
		for r in range( rows ):
			self_._canvas.create_line( 0, r * ssy, ss * columns, r * ssy, fill = 'gray', width = borderWidth )
		for i in range( noSamples ):
			c = ( i % columns )
			r = ( i // columns )
			self_._canvas.create_text(
				ss * c,
				ssy * r,
				anchor = "nw",
				text = "{}".format( self_._y[i][0] if type( self_._y[i] ) == list else self_._y[i] )
			)
			self_._canvas.create_image(
				ss * c,
				ssy * r,
				anchor = "nw",
				image = self_._imgs[i]
			)
			if self_._yPredict is not None and self_._y[i] != self_._yPredict[i]:
				self_._canvas.create_rectangle(
					c * ss - borderWidth / 2,
					r * ssy - borderWidth / 2,
					( c + 1 ) * ss - borderWidth / 2,
					( r + 1 ) * ssy - borderWidth / 2,
					outline = 'red', width = 2
				)

	def normalize( self_, x ):
		x = np.array( x, dtype = np.float32 )
		sup = max( x )
		inf = min( x )
		r = sup - inf
		x *= 255
		x /= r
		inf = min( x )
		if inf < 0:
			x -= inf
		x = np.array( x, dtype = np.int8 )
		x.shape = ( self_._sampleSize, self_._sampleSize )
		return x

	def on_resize( self_, event ):
		self_._winWidth = self_._root.winfo_width()
		self_._winHeight = self_._root.winfo_height()
		self_.paint()

	def on_motion( self_, event ):
		pass

	def on_b1down( self_, event ):
		print( "on_b1down" )

	def on_mousewheel( self_, event ):
		self_._canvas.yview_scroll( -1 if event.num == 4 else 1, "units" )
