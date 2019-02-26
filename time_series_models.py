# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize

class OLS:
	def __init__(self):
		self.coefs = None

	def fit(self, y, X):
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

		self.coefs = None

	def estimateCoefs(self):
		R = self.asset.calcLogReturns()
		p = self.p

		X = np.array([R[i:-(p - i)] for i in range(p)]).T
		y = R[p:]

		reggresor = OLS()
		reggresor.fit(y, X)

		self.coefs = reggresor.coefs





