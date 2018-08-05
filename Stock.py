import numpy as np
import pandas as pd

#from scipy import stats
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import style

style.use('ggplot')

class Stock:

	def __init__(self, quote, weight = 0):
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

	def getPrices(self):
		DF = pd.read_csv('hist_data/' + self.quote + '.dat')

		closeDF = DF['Adj Close']
		dates = DF['Date']

		return np.array(closeDF), np.array(dates)

	def calcLogReturns(self):
		closeDF = pd.read_csv('hist_data/' + self.quote + '.dat')['Adj Close']
		logReturns = np.log(closeDF / closeDF.shift(1)).dropna()

		return np.array(logReturns)

	def calcBeta(self, benchmark):
		stock_returns = self.calcLogReturns()
		benchmark_returns = benchmark.calcLogReturns()

		stock_returns = np.reshape(stock_returns, (len(stock_returns), 1))
		benchmark_returns = np.reshape(benchmark_returns, (len(benchmark_returns), 1))

		regressor = LinearRegression()
		regressor.fit(benchmark_returns, stock_returns)

		#print(regressor.coef_[0][0], regressor.intercept_[0])

		self.beta = regressor.coef_[0][0]

		print(self.beta)

	def graphPrices(self):
		closeDF, dates = self.getPrices()
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