import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import style

style.use('ggplot')

class Stock:

	def __init__(self, quote, weight = 0, beta = 0):
		self.quote = quote
		self.weight = weight

	def getQuote(self):
		return self.quote

	def getWeight(self):
		return self.weight

	def setWeight(self, weight):
		self.weight = weight

	def getBeta(self):
		return self.beta

	def getIncomeStatement(self):
		return pd.read_csv('financial_statements/inc_' + self.quote + '.dat')

	def getBalanceSheet(self):
		return pd.read_csv('financial_statements/bal_' + self.quote + '.dat')

	def getEPS(self):
		return float(pd.read_html('https://finance.yahoo.com/quote/' + self.quote + '?p=' + self.quote)[1][1][3])

	def getPE(self):
		return float(pd.read_html('https://finance.yahoo.com/quote/' + self.quote + '?p=' + self.quote)[1][1][2])

	def calcIndicators(self):
		income_statement = self.getIncomeStatement()
		balance_sheet = self.getBalanceSheet()

		equity = float(balance_sheet.iloc[36])

		gross_profit = float(income_statement.iloc[2])
		net_income = float(income_statement.iloc[24])

		ROE = net_income / equity

		print(ROE)

	def getPrices(self, return_dates = False):
		DF = pd.read_csv('hist_data/' + self.quote + '.dat')

		closeDF = DF['Adj Close']
		dates = DF['Date']

		if return_dates:
			return np.array(closeDF), np.array(dates)
		else:
			return np.array(closeDF)

	def getVolume(self):
		return np.array(pd.read_csv('hist_data/' + self.quote + '.dat')['Volume'])

	def calcLogReturns(self):
		closeDF = pd.read_csv('hist_data/' + self.quote + '.dat')['Adj Close']
		logReturns = np.log(closeDF / closeDF.shift(1)).dropna()

		return np.array(logReturns)

	def calcBeta(self, benchmark, graph = False):
		stockReturns = self.calcLogReturns()
		benchmarkReturns = benchmark.calcLogReturns()

		stockReturns = np.reshape(stockReturns, (len(stockReturns), 1))
		benchmarkReturns = np.reshape(benchmarkReturns, (len(benchmarkReturns), 1))

		regressor = LinearRegression()
		regressor.fit(benchmarkReturns, stockReturns)

		self.beta = regressor.coef_[0][0]

		if graph:
			plt.scatter(benchmarkReturns, stockReturns, color = 'blue', s = 23, alpha = 0.6, label = "Returns")
			plt.plot(benchmarkReturns, regressor.coef_ * benchmarkReturns + regressor.intercept_, color = 'red', linewidth = 2, label = "Fitting line")
			plt.ylabel(self.quote + " Log Returns")
			plt.xlabel(benchmark.getQuote() + " Log Returns")
			plt.legend(loc = 2)
			plt.title(self.quote + ' aganinst ' + benchmark.getQuote() + ' Log returns, β = ' + str(round(self.beta, 3)))
			plt.show()

			return self.beta

		#print(regressor.coef_[0][0], regressor.intercept_[0])
		return self.beta

	def graphPrices(self):
		closeDF, dates = self.getPrices(return_dates = True)
		dates = pd.to_datetime(dates)

		fig, ax = plt.subplots(1)
		fig.autofmt_xdate()
		plt.plot(dates, closeDF, color = 'blue', linewidth = 1.4, label = "Price")
		plt.title(self.getQuote())
		plt.xlabel("Date")
		plt.ylabel("Price")
		plt.grid(True)
		plt.legend(loc = 2)

		xfmt = mdates.DateFormatter('%Y-%m-%d')
		ax.xaxis.set_major_formatter(xfmt)
		plt.show()