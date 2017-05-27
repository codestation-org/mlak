import os
import pickle
import numpy as np
import scipy.io as scipy_io

def get_filename_extension( path_ ):
	_, fileExtension = os.path.splitext( path_ )
	return fileExtension[1:]

def load( path_ ):
	ext = get_filename_extension( path_ )
	data = {}
	if ext == "p" or ext == "pickle" or ext == "":
		with open( path_, 'rb' ) as f:
			data = pickle.load( f )
	elif ext == "txt":
		dataRaw = np.genfromtxt( path_, delimiter = "," )
		n = dataRaw.shape[1] - 1
		data["X"] = dataRaw[:,0:n]
		data["y"] = dataRaw[:,n:n + 1]
	elif ext == "mat":
		data = scipy_io.loadmat( "ex3data1.mat" )
	else:
		raise Exception( "Unknown file extension {}".format( ext ) )
	return data

def save( path_, data_ ):
	ext = get_filename_extension( path_ )
	data = {}
	if ext == "p" or ext == "pickle" or ext == "":
		with open( path_, 'wb' ) as f:
			pickle.dump( data_, f )
	elif ext == "mat":
		data = scipy_io.savemat( path_, data_ )
	else:
		raise Exception( "Unknown file extension {}".format( ext ) )

