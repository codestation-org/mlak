import numpy as np

# Turn whatever vector into a column vector.
def columnize( v ):
	v.shape = ( v.shape[0], 1 )

# Inplace transpose.
def transpose( v ):
	s = v.shape
	a1 = s[0]
	a2 = s[1] if len( s ) > 1 else 1
	v.shape = ( a2, a1 )

# Solve Normal Equation.
# Find \theta that gives best approximation for solution of given equation system.
def normal_equation( X, y, lambda_val ):
	n = np.size(X, axis=1)
	L = np.eye(n)
	L[0,0] = 0
	return np.dot( np.dot( np.linalg.pinv( np.dot( X.T, X ) + lambda_val * L), X.T ), y )

def add_ones_column( X ):
	m = np.size(X, axis=0)
	X_ext = np.c_[np.ones((m, 1)), X]
	return X_ext