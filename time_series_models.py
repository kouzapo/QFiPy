# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize

class LeastSquares:
	def __init__(self):
		self.coefs = None

	def fit(self, X, y):
		I = np.ones(X.shape[0])
		I = I.reshape((len(I), 1))

		X = np.hstack((I, X))

		#minFunc = lambda b: np.sqrt(((X.dot(b) - y) ** 2).sum() / X.shape[0])
		#minFunc = lambda b: ((y - X.dot(b)) ** 2).sum()

		#res = minimize(minFunc, X.shape[1] * [1], method = 'SLSQP')

		#self.coefs = res.x

		self.coefs = (np.linalg.inv((X.T).dot(X))).dot((X.T).dot(y))

		#self.coefs = np.linalg.lstsq(X, y, rcond = None)[0]

	def predict(self, X):
		if self.coefs is not None:
			return self.coefs[1] * X + self.coefs[0]
		else:
			return None

class AR:
	def __init__(self, p, asset):
		self.p = p
		self.asset = asset

		self.params = None

	def fit(self):
		p = self.p
		log_returns = self.asset.calcLogReturns()

		X = np.array([log_returns[i:-(p - i)] for i in range(p)]).T
		X = np.flip(X, axis = 1)

		y = log_returns[p:]

		regressor = LeastSquares()
		regressor.fit(X, y)

		self.params = regressor.coefs