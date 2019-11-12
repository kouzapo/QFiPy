#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains classes that perform time series analysis, such as
least squares estimator, AR and MA models.
"""

import numpy as np
from scipy.optimize import minimize

__author__ = "Apostolos Kouzoukos"
__license__ = "MIT"
__email__ = "kouzoukos97@gmail.com"
__status__ = "Development"

class LeastSquares:
	"""
	This class fits a model with the OLS method. It is equilevant
	to the sklearn LinearRegression class.
	"""

	def __init__(self):
		self.coefs = None

	def fit(self, X, y):
		"""
		This method fits the model using the normal equations.

		Parameters:
		----------
			X: ndarray.
			y: 1-d ndarray.
		"""

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
		"""
		This method makes predictions given a set of new instances.
		
		Parameters:
		----------
			X: ndarray.
		Returns:
		-------
			y_hat: 1-d ndarray, the predictions.

		"""

		if self.coefs is not None:
			y_hat = self.coefs[1] * X + self.coefs[0]

			return y_hat
		else:
			return None

class AR:
	"""
	This class implements the Autoregressive (AR) model.
	"""

	def __init__(self, p, asset):
		"""
		Parameters:
		----------
			p: int, the order of the AR.
			asset: a Stock/Index object.
		"""
		self.p = p
		self.asset = asset

		self.params = None

	def fit(self):
		"""
		This method fits the AR model.
		"""

		p = self.p
		log_returns = self.asset.calcLogReturns()

		X = np.array([log_returns[i:-(p - i)] for i in range(p)]).T
		X = np.flip(X, axis = 1)

		y = log_returns[p:]

		regressor = LeastSquares()
		regressor.fit(X, y)

		self.params = regressor.coefs
