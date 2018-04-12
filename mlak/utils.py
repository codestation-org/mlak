import warnings

class NoWarnings( object ):
	def __init__( self_ ):
		self_._warnSave = warnings.warn
	def __enter__( self_ ):
		warnings.warn = lambda *arga, **kwArgs: None
	def __exit__( self_, type_, value_, traceback_ ):
		warnings.warn = self_._warnSave

