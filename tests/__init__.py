import os
import shutil

try:
	shutil.rmtree( "out" )
except:
	pass

try:
	os.mkdir( "out" )
except:
	pass
