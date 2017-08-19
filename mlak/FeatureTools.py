import numpy as np
from scipy.special import expit


def add_features( X, functions ):

	m = np.size(X, axis=0)
	features_cnt_old = np.size(X, axis=1)
	features_cnt_new = np.size(functions)

	X_ext = np.zeros((m, features_cnt_old + features_cnt_new))

	for i in range(m):
		X_ext[i] = add_features_single(X[i], functions)

	return X_ext


def add_features_single( x, functions ):

	xe = []

	for f in functions:
		xe.append(f(x))

	return x.tolist() + xe
