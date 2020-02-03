#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module provides classes for modeling portfolios of stocks and bonds.
"""

import threading as thrd

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import matplotlib.pyplot as plt
from matplotlib import style

from equities import Stock, Index
from fixed_income import get_yields

__author__ = "Apostolos Kouzoukos"
__license__ = "MIT"
__email__ = "kouzoukos97@gmail.com"
__status__ = "Development"

style.use('ggplot')

class StockPortfolio:
	"""
	This class models a porfolio and the main objective is to optimize it's allocation
	based on the Markowitz framework. Several optimization options are available.
	"""

	def __init__(self, stocks):
		"""
		Parameters:
		----------
			stocks: a list of Stock objects.
		"""

		self.stocks = stocks

	def add_stock(self, stock):
		"""
		Adds a stock in the portfolio.

		Parameters:
		----------
			stock: a Stock object.
		"""

		self.stocks.append(stock)

	def get_stocks_weights(self):
		"""
		Returns the weights of the stocks after the allocation.

		Returns:
		-------
			w: a ndarray of weights.
		"""

		w = np.array([stock.weight for stock in self.stocks])

		return w

	def __calc_cov_matrix(self):
		"""
		This private method calculates the covariance matrix of the stocks.

		Returns:
		-------
			ret: a DataFrame object with the logarithmic returns of each stock.
			cov_matrix: a DataFrame object with the covariance of the stocks. 
		"""

		ret = {}

		for stock in self.stocks:
			ret[stock.quote] = stock.calc_log_returns()

		ret = pd.DataFrame(ret)
		cov_matrix = ret.cov()

		return ret, cov_matrix

	def calc_min_var_alloc(self, save = True, allow_short = True):
		"""
		This method allocates the stocks in the portfolio in order to achieve
		the minimum variance. Two optimization methods are implemented.

		Parameters:
		----------
			save: boolean, whether or not to save the weights.
			allow_short: boolean, whether or not to allow short selling.

		Returns:
		-------
			weights: a ndarray with the weight of each stock.
		"""

		n = len(self.stocks)
		rets, cov_matrix = self.__calc_cov_matrix()

		def __formula_calculation():
			C_inv = np.linalg.inv(cov_matrix.values)
			e = np.ones(len(self.stocks))

			weights = np.dot(e, C_inv) / np.dot(e, np.dot(C_inv, e))

			return weights

		def __optimization_solution():
			minFun = lambda weights: np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))

			cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
			bnds = tuple((0, 1) for _ in range(n))

			res = minimize(minFun, n * [1 / n], method = 'SLSQP', bounds = bnds, constraints = cons)
			weights = res.get('x')

			return weights

		if allow_short:
			weights =  __formula_calculation()
		else:
			weights = __optimization_solution()

		i = 0

		if save:
			for stock in self.stocks:
				stock.set_weight(weights[i])
				i += 1

		return weights

	def calc_min_var_line(self, mv, save = True, allow_short = True):
		"""
		This method allocates the stocks in the portfolio in order to achieve
		the minimum variance the desirable return. Two optimization methods are implemented.

		Parameters:
		----------
			mv: float, the return we want to achieve.
			save: boolean, whether or not to save the weights.
			allow_short: boolean, whether or not to allow short selling.

		Returns:
		-------
			weights: a ndarray with the weight of each stock.
		"""

		n = len(self.stocks)
		rets, cov_matrix = self.__calc_cov_matrix()

		def __formula_calculation():
			m = rets.mean() * 252
			C_inv = np.linalg.inv(cov_matrix.values)
			e = np.ones(len(self.stocks))

			eC_invM = np.dot(np.dot(e, C_inv), m)
			mC_invM = np.dot(np.dot(m, C_inv), m)
			mC_invE = np.dot(np.dot(m, C_inv), e)
			eC_invE = np.dot(np.dot(e, C_inv), e)

			eC_inv = np.dot(e, C_inv)
			mC_inv = np.dot(m, C_inv)

			A = np.linalg.det([[1, eC_invM], [mv, mC_invM]])
			B = np.linalg.det([[eC_invE, 1], [mC_invE, mv]])
			C = np.linalg.det([[eC_invE, eC_invM], [mC_invE, mC_invM]])

			weights = (A * eC_inv + B * mC_inv) / C

			return weights

		def __optimization_solution():
			minFun = lambda weights: np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))

			cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, {'type': 'eq', 'fun': lambda x: np.sum(rets.mean() * x) * 252 - mv})
			bnds = tuple((0, 1) for _ in range(n))

			res = minimize(minFun, n * [1 / n], method = 'SLSQP', bounds = bnds, constraints = cons)
			weights = res.get('x')

			return weights

		if allow_short:
			weights =  __formula_calculation()
		else:
			weights = __optimization_solution()

		i = 0

		if save:
			for stock in self.stocks:
				stock.set_weight(weights[i])
				i += 1

		return weights

	def maximize_sharpe_ratio(self, rf, save = True, allow_short = False):
		"""
		This method allocates the stocks in the portfolio in order to maximize
		the Sharpe Ratio.

		Parameters:
		----------
			rf: float, the risk free rate.
			save: boolean, whether or not to save the weights.
			allow_short: boolean, whether or not to allow short selling.

		Returns:
		-------
			weights: a ndarray with the weight of each stock.
		"""

		n = len(self.__stocks)
		rets, cov_matrix = self.__calc_cov_matrix()

		def __min_func(weights):
			weights = np.array(weights)
			portExpRet = np.sum(rets.mean() * weights) * 252
			portStd = np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))

			return -((portExpRet - rf) / portStd)

		cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

		if allow_short:
			bnds = tuple((-1, 1) for _ in range(n))
		else:
			bnds = tuple((0, 1) for _ in range(n))

		res = minimize(__min_func, n * [1 / n], method = 'SLSQP', bounds = bnds, constraints = cons)
		weights = res.get('x')

		i = 0

		if save:
			for stock in self.__stocks:
				stock.setWeight(weights[i])
				i += 1

		return weights

	def calc_expected_return(self):
		"""
		Calculates the expected return of the portfolio.

		Returns:
		-------
			exp_ret: float, the expected return.
		"""

		rets, cov_matrix = self.__calc_cov_matrix()
		weights = self.get_stocks_weights()

		exp_ret = np.dot(rets.mean(), weights) * 252

		return exp_ret

	def calc_standard_deviation(self):
		"""
		Calculates the standard deviation of the portfolio.

		Returns:
		-------
			std_dev: float, the standard deviation.
		"""

		rets, cov_matrix = self.__calc_cov_matrix()
		weights = self.get_stocks_weights()

		std_dev = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights))) * np.sqrt(252)

		return std_dev

	def calc_performance(self, *args):
		if len(args) == 2:
			rets = args[0]
			cov_matrix = args[1]

		elif len(args) == 3:
			rets = args[1]
			cov_matrix = args[2]
			
		else:
			rets, cov_matrix = self.__calc_cov_matrix()

		weights = self.get_stocks_weights()

		exRet = np.dot(rets.mean(), weights) * 252
		std = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights))) * np.sqrt(252)

		if len(args) == 1:
			rf = args[0]
			benchmark = Index('^GSPC')

			sharpeRatio = (exRet - rf) / std
			stocksBetas = np.array([s.calc_beta_alpha(benchmark)['beta'] for s in self.__stocks])
			stocksAlphas = np.array([s.calc_beta_alpha(benchmark)['alpha'] for s in self.__stocks])

			beta = stocksBetas.dot(weights)
			alpha = stocksAlphas.dot(weights)

			res = {'return': round(exRet, 5), 'std': round(std, 5), 'sharpe': round(sharpeRatio, 5), 'beta': round(beta, 5), 'alpha': round(alpha, 5)}
		else:
			res = {'return': round(exRet, 5), 'std': round(std, 5)}

		return res

	def print_summary(self, res):
		D = pd.DataFrame([self.get_stocks_weights()], index = ['Allocation'])
		D.columns = [stock.quote for stock in self.get_stocks()]

		print('---------------------Portfolio Summary---------------------')
		print(D,'\n')
		print('Expected Return:', res['return'])
		print('Standard Deviation:', res['std'])

		if len(res) == 5:
			print('Sharpe Ratio:', res['sharpe'])
			print('Beta:', res['beta'])
			print('Alpha:', res['alpha'])

		print()

	def plotEfficientFrontier(self, graph = True):
		R = np.linspace(0.05, 0.35, 50)
		rets, cov_matrix = self.__calc_cov_matrix()

		portExpRet = []
		portStd = []

		for i in R:
			self.calc_min_var_line(i, allow_short = True)
			res = self.calc_performance(rets, cov_matrix)
			m = res['return']
			s = res['std']

			portExpRet.append(m)
			portStd.append(s)

		stockExpRet = []
		stockStd = []

		for stock in self.__stocks:
			stockExpRet.append(stock.calc_exp_return())
			stockStd.append(stock.calc_std())

		if graph:
			plt.plot(portStd, portExpRet, color = 'blue', linewidth = 2, label = "Efficient Frontier")
			plt.scatter(stockStd, stockExpRet, s = 30, color = 'red', label = "Asset")
			plt.ylabel("Expected return")
			plt.xlabel("Standard deviation")
			plt.title("Efficient Frontier")
			plt.legend(loc = 2)
			plt.show()

		return portExpRet, portStd

	def __generate_random_portfolios(self, N):
		results = []

		for i in range(N):
			weights = np.random.random(len(self.__stocks))
			weights /= np.sum(weights)

			quotes = [s.quote for s in self.__stocks]
			stocks = []
			j = 0

			for q in quotes:
				stocks.append(Stock(q, weights[j]))
				j += 1

			results.append(StockPortfolio(stocks))

		return results

	def plotSimulatedRandomProtfolios(self, N):
		simulated_portfolios = self.__generate_random_portfolios(N)
		rets, cov_matrix = self.__calc_cov_matrix()

		'''A = np.arange(0, N, N / 10, dtype = int)
		S = [simulated_portfolios[A[i]:A[i + 1]] for i in range(len(A) - 1)]
		S.append(simulated_portfolios[A[-1]:])'''

		rf = get_yields()['10 yr']

		M = []
		S = []

		for p in simulated_portfolios:
			res = p.calc_performance(rets, cov_matrix)
			m = res['return']
			s = res['std']

			M.append(m)
			S.append(s)

		M = np.array(M)
		S = np.array(S)

		plt.scatter(S, M, s = 12, c = (M - rf) / S, alpha = 1, label = "Portfolio")
		plt.ylabel("Expected return")
		plt.xlabel("Standard deviation")
		plt.title("Simulated Random Portfolios")
		plt.colorbar(label = "Sharpe Ratio")
		plt.legend(loc = 2)
		plt.show()
