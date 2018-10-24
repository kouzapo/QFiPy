# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy import stats

from bs4 import BeautifulSoup
import urllib3

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import style

style.use('ggplot')

class Stock:
	def __init__(self, quote, weight = 0):
		self.http = urllib3.PoolManager()
		urllib3.disable_warnings()

		self.quote = quote
		self.weight = weight

	def getQuote(self):
		return self.quote

	def getWeight(self):
		return self.weight

	def setWeight(self, weight):
		self.weight = weight

	def getIncomeStatement(self):
		return pd.read_csv('financial_statements/inc_' + self.quote + '.dat')

	def getBalanceSheet(self):
		return pd.read_csv('financial_statements/bal_' + self.quote + '.dat')

	def getCurrentPrice(self):
		S = self.http.request('GET', 'https://finance.yahoo.com/quote/' + self.quote + '?p=' + self.quote)
		soup = BeautifulSoup(S.data, 'lxml')
		J = soup.find('span', class_ = 'Trsdu(0.3s) Fw(b) Fz(36px) Mb(-4px) D(ib)')

		return float(J.text.replace(',', ''))

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

	def calcExpReturn(self, annualized = True):
		logReturns = self.calcLogReturns()

		if annualized:
			return logReturns.mean() * 252
		else:
			return logReturns.mean()

	def calcStd(self, annualized = True):
		logReturns = self.calcLogReturns()

		if annualized:
			return logReturns.std() * np.sqrt(252)
		else:
			return logReturns.std()

	def calcCorrCoef(self, asset):
		return np.corrcoef(self.calcLogReturns(), asset.calcLogReturns())[0][1]

	def calcBeta(self, benchmark):
		stockReturns = self.calcLogReturns()
		benchmarkReturns = benchmark.calcLogReturns()

		stockReturns = np.reshape(stockReturns, (len(stockReturns), 1))
		benchmarkReturns = np.reshape(benchmarkReturns, (len(benchmarkReturns), 1))

		regressor = LinearRegression()
		regressor.fit(benchmarkReturns, stockReturns)
		R_sq = regressor.score(benchmarkReturns, stockReturns)

		return {'beta': regressor.coef_[0][0], 'alpha': regressor.intercept_[0], 'R-squared': R_sq}

	def calcSharpeRatio(self, rf):
		return (self.calcExpReturn() - rf) / self.calcStd()

	def calcVaR(self, c = 0.95):
		logReturns = self.calcLogReturns()
		logReturns.sort()
		a = round(len(logReturns) * (1 - c))

		return {'VaR': logReturns[a - 1], 'CVaR': logReturns[0:a - 1].mean()}

	def normalTest(self):
		return stats.normaltest(self.calcLogReturns())[1]

	def graphPrices(self):
		closeDF, dates = self.getPrices(return_dates = True)
		rollingMean = pd.DataFrame(closeDF).rolling(window = 60, min_periods = 0).mean()
		dates = pd.to_datetime(dates)
		volume = self.getVolume()

		fig, (ax1, ax2) = plt.subplots(2, sharex = True, gridspec_kw = {'height_ratios': [4, 1]})
		fig.autofmt_xdate()

		ax1.plot(dates, closeDF, color = 'blue', linewidth = 1.8, label = "Price")
		ax1.plot(dates, rollingMean, color = 'red', linewidth = 1.0, label = "Rolling Mean")

		ax2.bar(dates, volume, width = 2, color = 'blue', label = "Volume")

		plt.suptitle(str(self.getQuote()) + " price movement and Volume", fontsize = 20)
		ax1.set_ylabel("Price", fontsize = 12)
		ax2.set_ylabel("Volume", fontsize = 12)
		ax1.legend(loc = 2)
		xfmt = mdates.DateFormatter('%Y-%m-%d')
		ax1.xaxis.set_major_formatter(xfmt)

		plt.show()

	def graphLogReturns(self):
		logReturns = self.calcLogReturns()

		fig, (ax1, ax2) = plt.subplots(1, 2)

		ax1.plot(logReturns, color = 'blue', lw = 0.4)
		ax2.hist(logReturns, bins = 40, color = 'blue')

		ax1.set_ylabel("% Change", fontsize = 12)

		ax2.set_ylabel("Density", fontsize = 12)
		ax2.set_xlabel("% Change", fontsize = 15)
		plt.suptitle(str(self.getQuote()) + " Log Returns," + " μ = " + str(round(self.calcExpReturn(), 3)) + " σ = " + str(round(self.calcStd(), 3)), fontsize = 15)

		plt.show()

	def graphCorrelation(self, benchmark):
		stockReturns = self.calcLogReturns()
		benchmarkReturns = benchmark.calcLogReturns()
		B = self.calcBeta(benchmark)
		corrcoef = self.calcCorrCoef(benchmark)

		plt.scatter(benchmarkReturns, stockReturns, color = 'blue', s = 23, alpha = 0.5, label = "Returns")
		plt.plot(benchmarkReturns, B['beta'] * benchmarkReturns + B['alpha'], color = 'red', linewidth = 2, label = "Fitting line")

		plt.ylabel(self.quote + " Log Returns", fontsize = 12)
		plt.xlabel(benchmark.getQuote() + " Log Returns", fontsize = 15)
		plt.legend(loc = 2)
		plt.title(self.quote + " aganinst " + benchmark.getQuote() + " Log returns," + " ρ = " + str(round(corrcoef, 3)) + ", β = " + str(round(B['beta'], 3)), fontsize = 18)

		plt.show()