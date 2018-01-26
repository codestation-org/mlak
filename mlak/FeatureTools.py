import numpy as np
from scipy.special import expit

import mlak.Terminal as term


def find_normalization_params( X ):
	mu = np.mean( X, axis = 0 )
	sigma = np.std( X, axis = 0, ddof = 0 )
	epsilon = 0.1 ** 100
	sigma[sigma < epsilon] = epsilon
	return mu, sigma


def normalize_features( X, mu, sigma ):
	return ( X - mu ) / sigma


def feature_normalize(X):
	mu, sigma = find_normalization_params(X)
	return normalize_features(X, mu, sigma), mu, sigma


def add_features( X, functions ):
	if not functions:
		return X
	m = np.size(X, axis=0)
	features_cnt_old = np.size(X, axis=1)
	features_cnt_new = np.size(functions)

	X_ext = np.zeros((m, features_cnt_old + features_cnt_new))
	#p = term.Progress( m, "Adding features: " )
	for i in range( m ):
		#next( p )
		val = add_features_single(X[i], functions)
		#print("old X_ext[i] {}".format(X_ext[i]))
		#print("new X_ext[i] {}".format(val))
		X_ext[i] = val
	return X_ext


def add_features_single( x, functions ):

	xe = []

	for f in functions:
		#val = np.array(f(x)).flatten()
		xe.append(f(x))

	return x.tolist() + xe
