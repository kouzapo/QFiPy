#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains classes that model the behavior of equities (stocks)
and stock market indices. Many methods that give usefull insights about the stocks
and indices behavior are implemented, ranging from fundamental and technical analysis
to time series analysis. The data used to conduct the analysis is historical date-to-date
historical data updated manualy by the user, using the classes in update_data.py module.
Financial statements (income statement and balance sheet) are also provided for the stocks.
The stock objects can be used to construct portfolios (see the portfolio.py module) and as the
underlying asset for derivatives such as options and futures contracts (see the derivatives.py module).
"""

import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
from scipy import stats

from bs4 import BeautifulSoup
import urllib3

#from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import style

from time_series_models import LeastSquares

__author__ = "Apostolos Anastasios Kouzoukos"
__email__ = "kouzoukos97@gmail.com"
__status__ = "Development"

style.use('ggplot')
register_matplotlib_converters()

class Index:
	"""

	"""

	def __init__(self, quote):
		"""
		This is the constructor of the Index base class
		"""

		#self.__http = urllib3.PoolManager()
		urllib3.disable_warnings()

		self.quote = quote

	def getCurrentPrice(self):
		"""
		This method returns the current price of an asset based on the price 
		indicated by Yahoo Finance. It makes an http request.
		"""

		http = urllib3.PoolManager()
		urllib3.disable_warnings()

		S = http.request('GET', 'https://finance.yahoo.com/quote/' + self.quote + '?p=^' + self.quote)
		soup = BeautifulSoup(S.data, 'lxml')
		J = soup.find('span', class_ = 'Trsdu(0.3s) Fw(b) Fz(36px) Mb(-4px) D(ib)')

		return float(J.text.replace(',', ''))

	def getPrices(self, return_dates = False):
		"""
		This method opens the historical price data file of the asset 
		and returns a time series of the adjusting closing prices.

		Parameters:
		----------
			return_dates: bool, defaults to False.

		Returns:
		-------
			close_prices: ndarray, a numpy array containing the closing prices.

			dates: ndarray, a numpy array containing the dates.
		"""

		#df = pd.read_csv('data/historical_data/' + self.quote + '.dat')
		df = pd.read_csv('data/historical_data/{}.dat'.format(self.quote))

		close_prices = np.array(df['Adj Close'])
		dates = np.array(df['Date'])

		if return_dates:
			return close_prices, dates
		else:
			return close_prices

	def getVolume(self):
		"""
		This method returns a time series of the daily volume.

		Returns:
		-------
			volume: ndarray, a numpy array containing the daily volume.
		"""

		volume = np.array(pd.read_csv('data/historical_data/{}.dat'.format(self.quote))['Volume'])

		return volume

		#return np.array(pd.read_csv('data/historical_data/' + self.quote + '.dat')['Volume'])

	def calcLogReturns(self):
		"""
		This method calculates the log returns of the asset based on daily historical
		closing prices. Many other methods and calculations are based on this time series.

		Returns:
		-------
			log_returns: ndarray, a numpy array containing the log returns.
		"""

		closeDF = pd.read_csv('data/historical_data/' + self.quote + '.dat')['Adj Close']
		log_returns = np.array(np.log(closeDF / closeDF.shift(1)).dropna())

		return log_returns

	def calcExpReturn(self, annualized = True):
		"""
		This method calculates the daily expected return of the asset based 
		on the historical returns. The annualized return is also possible to calculate.

		Paramaters:
		----------
			annualized: bool, if set to True, the method returns the annualized return.
			It defaults to True.

		Returns:
		-------
			exp_return: float, the expected return (daily or annualized).
		"""

		log_returns = self.calcLogReturns()

		if annualized:
			exp_return = log_returns.mean() * 252
		else:
			exp_return = log_returns.mean()

		return exp_return

	def calcStd(self, annualized = True):
		"""
		This method calculates the daily standard deviation of an asset based on the 
		historical returns. The annualized return is also possible to calculate.

		Paramaters:
		----------
			annualized: bool, if set to True, the method returns the annualized standard deviation.
			It defaults to True.

		Returns:
		-------
			standard_dev: float, the standard deviation (daily or annualized).
		"""

		log_returns = self.calcLogReturns()

		if annualized:
			standard_dev = log_returns.std() * np.sqrt(252)
		else:
			standard_dev = log_returns.std()

		return standard_dev

	def calcSkewness(self):
		"""
		This method calculates the skewness of the asset based on the historical returns.

		Returns:
		-------
			skewness: float, the skewness.
		"""

		skewness = stats.skew(self.calcLogReturns())

		return skewness

	def calcKurtosis(self):
		"""
		This method calculates the kurtosis of the asset based on the historical returns.

		Returns:
		-------
			kurtosis: float, the kurtosis.
		"""

		kurtosis = stats.kurtosis(self.calcLogReturns())

		return kurtosis

	def calcCorrCoef(self, asset):
		"""
		This method calculates the correlation coefficient between two assets. 
		Both assets must have a calcLogReturns method, thus the asset object
		passed as parameter mey either be a Stock object or an Index object.

		Parameters:
		----------
			asset: Stock or Index object, the asset of which the log returns are used to calculate
			the correlation coefficient between the two assets.

		Returns:
		-------
			corr_coef: float, the correlation coefficient between the two assets.
		"""

		corr_coef = np.corrcoef(self.calcLogReturns(), asset.calcLogReturns())[0][1]

		return corr_coef

	def calcACF(self, lags):
		"""
		This method calculates the autocorreation of the asset up to a predefined lag.

		Parameters:
		----------
			lags: int, the max lag of the autocorrelations to be calculated.

		Returns:
		-------
			acf: ndarray, a numpy array of the autocorrelations.
		"""

		log_returns = self.calcLogReturns()

		acf = np.array([np.corrcoef(log_returns[lag:], log_returns[:-lag])[0][1] for lag in lags])

		return acf

	def calcPACF(self, lags):
		"""
		This method calculates the partial autocorreation of the asset up to a predefined lag.

		Parameters:
		----------
			lags: int, the max lag of the partial autocorrelations to be calculated.

		Returns:
		-------
			pacf: ndarray, a numpy array of the partial autocorrelations.
		"""

		log_returns = self.calcLogReturns()
		regressor = LeastSquares()

		pacf = []

		for lag in lags:
			X = np.array([log_returns[i:-(lag - i)] for i in range(lag)]).T
			y = log_returns[lag:]

			regressor.fit(X, y)

			pacf.append(regressor.coefs[1])

		pacf = np.array(pacf)

		return pacf

	def testNormality(self):
		"""
		This method returns the t-statistic and the p-value of the normality test of the
		asset's returns. 

		Returns:
		-------
			results: ndarray, a numpy array of the normality test results.
		"""

		results = stats.normaltest(self.calcLogReturns())

		return results

	def testAutocorrelation(self, lags):
		"""

		"""

		ACF = self.calcACF(lags)
		n = len(self.calcLogReturns())

		Q = []
		p_values = []

		for lag in lags:
			autocorrs = ACF[:lag]
			k = np.arange(1, len(lags[:lag]) + 1)

			q = n * (n + 2) * ((autocorrs ** 2) / (n - k)).sum()
			p = 1 - stats.chi2.cdf(q, lag)

			Q.append(q)
			p_values.append(p)

		return (np.array(Q), np.array(p_values))

	def testPartialAutocorrelation(self, lags):
		PACF = self.calcPACF(lags)
		n = len(self.calcLogReturns())

		Q = []
		p_values = []

		for lag in lags:
			partial_autocorrs = PACF[:lag]
			k = np.arange(1, len(lags[:lag]) + 1)

			q = n * (n + 2) * ((partial_autocorrs ** 2) / (n - k)).sum()
			p = 1 - stats.chi2.cdf(q, lag)

			Q.append(q)
			p_values.append(p)

		return (np.array(Q), np.array(p_values))

	def testStationarity(self, number_of_subsamples):
		log_returns = self.calcLogReturns()
		n = len(log_returns)

		A = np.arange(0, n, n / number_of_subsamples)
		A = np.array([int(i) for i in A])

		subsamples = [log_returns[A[i]:A[i + 1]] for i in range(len(A) - 1)]
		subsamples.append(log_returns[A[-1]:])

		results = [{'mean': round(subsample.mean(), 5), 'std': round(subsample.std(), 5)} for subsample in subsamples]

		for i in results:
			print(i)

	def calcSharpeRatio(self, rf):
		return (self.calcExpReturn() - rf) / self.calcStd()

	def descriptiveStats(self):
		closeDF = pd.read_csv('data/historical_data/' + self.quote + '.dat')['Adj Close']
		log_returns = np.log(closeDF / closeDF.shift(1)).dropna()

		desc = log_returns.describe()
		skewness = self.calcSkewness()
		kurtosis = self.calcKurtosis()

		print('-----Descriptive Statistics for ' + self.quote + '-----')
		print('count\t', desc['count'])
		print('mean\t', round(desc['mean'], 6))
		print('std\t', round(desc['std'], 6))
		print('skew\t', round(skewness, 6))
		print('kurt\t', round(kurtosis, 6))
		print('min\t', round(desc['min'], 6))
		print('max\t', round(desc['max'], 6))
		print('25%\t', round(desc['25%'], 6))
		print('50%\t', round(desc['50%'], 6))
		print('75%\t', round(desc['75%'], 6))

	def plotPrice(self):
		closeDF, dates = self.getPrices(return_dates = True)
		rolling_mean = pd.DataFrame(closeDF).rolling(window = 60, min_periods = 0).mean()
		dates = pd.to_datetime(dates)
		volume = self.getVolume()

		fig, (ax1, ax2) = plt.subplots(2, sharex = True, gridspec_kw = {'height_ratios': [4, 1]})
		fig.autofmt_xdate()

		ax1.plot(dates, closeDF, color = 'blue', linewidth = 1.8, label = "Price")
		ax1.plot(dates, rolling_mean, color = 'red', linewidth = 1.0, label = "Rolling Mean")

		ax2.bar(dates, volume, width = 2, color = 'blue', label = "Volume")

		plt.suptitle(str(self.quote) + " value movement and Volume", fontsize = 20)
		ax1.set_ylabel("Price", fontsize = 12)
		ax2.set_ylabel("Volume", fontsize = 12)
		ax1.legend(loc = 2)
		xfmt = mdates.DateFormatter('%Y-%m-%d')
		ax1.xaxis.set_major_formatter(xfmt)

		plt.show()

	def plotLogReturns(self):
		log_returns = self.calcLogReturns()

		fig, (ax1, ax2) = plt.subplots(1, 2)

		ax1.plot(log_returns, color = 'blue', lw = 0.4)
		ax2.hist(log_returns, bins = 40, color = 'blue')

		ax1.set_ylabel("% Change", fontsize = 12)

		ax2.set_ylabel("Density", fontsize = 12)
		ax2.set_xlabel("% Change", fontsize = 15)
		plt.suptitle(str(self.quote) + " Log Returns", fontsize = 18)
		plt.show()

	def plotQQPlot(self):
		log_returns = self.calcLogReturns()
		R = np.arange(-3.3, 3.3, 0.1)

		quantiles, LSFit = stats.probplot(log_returns, dist = "norm")

		plt.scatter(quantiles[0], quantiles[1], color = 'blue', alpha = 0.5, label = 'Quantiles')
		plt.plot(R, LSFit[0] * R + LSFit[1], color = 'red', label = 'Best Fit Line')

		plt.ylabel('Ordered Values')
		plt.xlabel('Theoretical quantiles')
		plt.legend(loc = 2)
		plt.title('Q-Q plot for ' + self.quote, fontsize = 18)

		plt.show()

	def plotACF(self, max_lag, confidence = 0.05):
		lags = np.arange(1, max_lag + 1, 1)
		ACF = self.calcACF(lags)

		confidence_interval = stats.norm.ppf(1 - confidence / 2) / np.sqrt(len(self.calcLogReturns()))

		plt.bar(lags, ACF, width = 0.7, color = 'blue', label = 'ACF')

		plt.plot([0, max_lag], [confidence_interval, confidence_interval], color = 'red', linestyle = ':')
		plt.plot([0, max_lag], [-confidence_interval, -confidence_interval], color = 'red', linestyle = ':')

		plt.ylabel('ACF')
		plt.xlabel('Lag')
		plt.legend(loc = 2)
		plt.title('Autocorrelation Function for ' + self.quote, fontsize = 15)

		plt.show()

	def plotPACF(self, max_lag, confidence = 0.05):
		lags = np.arange(1, max_lag + 1, 1)
		PACF = self.calcPACF(lags)

		confidence_interval = stats.norm.ppf(1 - confidence / 2) / np.sqrt(len(self.calcLogReturns()))

		plt.bar(lags, PACF, width = 0.7, color = 'blue', label = 'PACF')

		plt.plot([0, max_lag], [confidence_interval, confidence_interval], color = 'red', linestyle = ':')
		plt.plot([0, max_lag], [-confidence_interval, -confidence_interval], color = 'red', linestyle = ':')

		plt.ylabel('PACF')
		plt.xlabel('Lag')
		plt.legend(loc = 2)
		plt.title('Partial Autocorrelation Function for ' + self.quote, fontsize = 15)

		plt.show()

class Stock(Index):
	def __init__(self, quote, weight = 0):
		Index.__init__(self, quote)

		#self.http = urllib3.PoolManager()
		urllib3.disable_warnings()

		self.quote = quote
		self.weight = weight

	def setWeight(self, weight):
		self.weight = weight

	def getIncomeStatement(self):
		return pd.read_csv('financial_statements/inc_' + self.quote + '.dat')

	def getBalanceSheet(self):
		return pd.read_csv('financial_statements/bal_' + self.quote + '.dat')

	def calcIndicators(self):
		income_statement = self.getIncomeStatement()
		balance_sheet = self.getBalanceSheet()

		print(balance_sheet.iloc[7])

	def calcBetaAlpha(self, benchmark):
		stock_returns = self.calcLogReturns()
		benchmark_returns = benchmark.calcLogReturns()

		benchmark_returns = np.reshape(benchmark_returns, (len(benchmark_returns), 1))

		'''stock_returns = np.reshape(stock_returns, (len(stock_returns), 1))
		benchmark_returns = np.reshape(benchmark_returns, (len(benchmark_returns), 1))

		regressor = LinearRegression()
		regressor.fit(benchmark_returns, stock_returns)

		return {'alpha': regressor.intercept_[0], 'beta': regressor.coef_[0][0]}'''

		regressor = LeastSquares()
		regressor.fit(benchmark_returns, stock_returns)

		return {'alpha': regressor.coefs[0], 'beta': regressor.coefs[1]}

	def calcVaR(self, c = 0.95):
		log_returns = self.calcLogReturns()
		log_returns.sort()
		a = round(len(log_returns) * (1 - c))

		return {'VaR': log_returns[a - 1], 'CVaR': log_returns[0:a - 1].mean()}

	def plotCorrelation(self, benchmark):
		stock_returns = self.calcLogReturns()
		benchmark_returns = benchmark.calcLogReturns()

		B = self.calcBetaAlpha(benchmark)

		plt.scatter(benchmark_returns, stock_returns, color = 'blue', s = 23, alpha = 0.5, label = "Returns")
		plt.plot(benchmark_returns, B['beta'] * benchmark_returns + B['alpha'], color = 'red', linewidth = 2, label = "Fitting line")

		plt.ylabel(self.quote + " Log Returns", fontsize = 12)
		plt.xlabel(benchmark.quote + " Log Returns", fontsize = 12)
		plt.legend(loc = 2)
		plt.title(self.quote + " aganinst " + benchmark.quote, fontsize = 18)

		plt.show()
