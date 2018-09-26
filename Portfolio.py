import numpy as np
import pandas as pd

from scipy.optimize import minimize

from Stock import *
from Utilities import getRiskFreeRate

import matplotlib.pyplot as plt

class Portfolio:
	def __init__(self, stocks):
		self.stocks = stocks

	def getStocks(self):
		return self.stocks

	def addStock(self, stock):
		self.getStocks().append(stock)

	def getStocksWeights(self):
		return np.array([stock.getWeight() for stock in self.getStocks()])

	def calcCovMatrix(self):
		ret = {}

		for stock in self.getStocks():
			ret[stock.getQuote()] = stock.calcLogReturns()

		ret = pd.DataFrame(ret)
		covMatrix = ret.cov()

		return ret, covMatrix

	def calcMinVarAlloc(self):
		ret, covMatrix = self.calcCovMatrix()

		C_inv = np.linalg.inv(covMatrix.values)
		e = np.ones(len(self.getStocks()))

		weights = np.dot(e, C_inv) / np.dot(e, np.dot(C_inv, e))
		i = 0

		for stock in self.getStocks():
			stock.setWeight(weights[i])
			i += 1

	def calcMinVarLine(self, mv):
		ret, covMatrix = self.calcCovMatrix()
		m = ret.mean() * 252
		C_inv = np.linalg.inv(covMatrix.values)
		e = np.ones(len(self.getStocks()))

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
		i = 0

		for stock in self.getStocks():
			stock.setWeight(weights[i])
			i += 1

	def calcPerformance(self):
		ret, covMatrix = self.calcCovMatrix()

		weights = self.getStocksWeights()

		exRet = np.dot(ret.mean(), weights) * 252
		std = np.sqrt(np.dot(weights, np.dot(covMatrix, weights))) * np.sqrt(252)

		return exRet, std

	def calcSharpeRatio(self, rf):
		exRet, std = self.calcPerformance()

		return (exRet - rf) / std

	def genRandomPortfolios(self, n):
		results = []

		for i in range(n):
			weights = np.random.random(len(self.getStocks()))
			weights /= np.sum(weights)

			quotes = [s.getQuote() for s in self.getStocks()]
			stocks = []
			j = 0

			for s in quotes:
				stocks.append(Stock(s, weights[j]))
				j += 1

			results.append(Portfolio(stocks))

		return results

	def maximizeSharpeRatio(self, I):
		P = self.genRandomPortfolios(I)

		M = []
		S = []
		i = 0

		for p in P:
			m, s = p.calcPerformance()
			print(i)

			i += 1
			M.append(m)
			S.append(s)

		M = np.array(M)
		S = np.array(S)

		minFunc = -(M - getRiskFreeRate()) / S
		cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
		bnds = tuple((0,1) for x in range(len(self.getStocks())))
		w = len(self.getStocks()) * [1 / len(self.getStocks())]

		opts = minimize(minFunc, w, method = 'SLSQP', bounds = bnds, constraints = cons)

		print(opts)

	def graphEfficientFrontier(self):
		self.calcMinVarAlloc()
		m, s = self.calcPerformance()

		R = np.arange(m, 0.4, 0.01)

		portExpRet = []
		portStd = []

		for i in R:
			self.calcMinVarLine(i)
			m, s = self.calcPerformance()

			portExpRet.append(m)
			portStd.append(s)

		stockExpRet = []
		stockStd = []

		for stock in self.getStocks():
			stockExpRet.append(stock.calcExpReturn())
			stockStd.append(stock.calcStd())

		plt.plot(portStd, portExpRet, color = 'blue', linewidth = 2, label = "Efficient Frontier")
		plt.scatter(stockStd, stockExpRet, s = 30, color = 'red', label = "Asset")
		plt.ylabel("Expected return")
		plt.xlabel("Standard deviation")
		plt.title("Efficient Frontier with individual assets")
		plt.legend(loc = 2)
		plt.show()

	def graphSimulatedEfficientFrontier(self, I):
		P = self.genRandomPortfolios(I)

		M = []
		S = []
		i = 0

		for p in P:
			m, s = p.calcPerformance()
			print(i)

			i += 1
			M.append(m)
			S.append(s)

		M = np.array(M)
		S = np.array(S)

		plt.scatter(S, M, s = 12, c = (M - getRiskFreeRate()) / S, alpha = 1, label = "Portfolio")
		plt.ylabel("Expected return")
		plt.xlabel("Standard deviation")
		plt.title("Simulated Efficient Frontier")
		plt.colorbar(label = "Sharpe Ratio")
		plt.legend(loc = 2)
		plt.show()