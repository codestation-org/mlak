import os
import pickle
import numpy as np
from io import StringIO
import scipy.io as scipy_io
import hashlib

def get_filename_extension( path_ ):
	_, fileExtension = os.path.splitext( path_ )
	return fileExtension[1:]

# Load data from known file types.
# Supported file types are: CSV, Python Pickle, Matlab.
def load( path_ ):
	ext = get_filename_extension( path_ )
	data = {}
	if ext == "p" or ext == "pickle" or ext == "":
		with open( path_, 'rb' ) as f:
			data = pickle.load( f )
	elif ext == "txt":
		data["X"] = np.genfromtxt( path_, delimiter = "," )[:,:-1]
		data["y"] = np.genfromtxt( path_, dtype = str, usecols = -1, autostrip = True, delimiter = "," )
	elif ext == "mat":
		data = scipy_io.loadmat( path_ )
	else:
		raise Exception( "Unknown file extension {}".format( ext ) )
	return data

# Save data to either Python Pickle or Matlab file.
# Desired file type is recognized from suppied file name extension.
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


def write_text_to_file(path, content):
	obj = open(path, 'wb')
	obj.write(content.encode())
	obj.close()

def append_text_to_file(path, content):
	obj = open(path, 'ab')
	obj.write(content.encode())
	obj.close()

def get_file_md5(path):
	md5 = hashlib.md5()
	with open(path,'rb') as f:
		for chunk in iter(lambda: f.read(8192), b''):
			md5.update(chunk)
	return md5.hexdigest()
