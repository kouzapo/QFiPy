# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy import stats

from bs4 import BeautifulSoup
import urllib3

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import style

style.use('ggplot')

class Index:
	def __init__(self, quote):
		self.http = urllib3.PoolManager()
		urllib3.disable_warnings()

		self.quote = quote

	def getQuote(self):
		return self.quote

	def getCurrentPrice(self):
		http = urllib3.PoolManager()
		urllib3.disable_warnings()

		S = http.request('GET', 'https://finance.yahoo.com/quote/' + self.quote + '?p=^' + self.quote)
		soup = BeautifulSoup(S.data, 'lxml')
		J = soup.find('span', class_ = 'Trsdu(0.3s) Fw(b) Fz(36px) Mb(-4px) D(ib)')

		return float(J.text.replace(',', ''))

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

	def calcSkewness(self):
		return stats.skew(self.calcLogReturns())

	def calcKurtosis(self):
		return stats.kurtosis(self.calcLogReturns())

	def calcCorrCoef(self, asset):
		return np.corrcoef(self.calcLogReturns(), asset.calcLogReturns())[0][1]

	def calcAutocorr(self, lag):
		logReturns = self.calcLogReturns()

		return np.corrcoef(logReturns[lag:], logReturns[:-lag])[0][1]

	def normalTest(self):
		return stats.normaltest(self.calcLogReturns())[1]

	def descriptiveStats(self):
		closeDF = pd.read_csv('hist_data/' + self.quote + '.dat')['Adj Close']
		logReturns = np.log(closeDF / closeDF.shift(1)).dropna()

		desc = logReturns.describe()
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

		plt.suptitle(str(self.getQuote()) + " value movement and Volume", fontsize = 20)
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
		plt.suptitle(str(self.getQuote()) + " Log Returns", fontsize = 18)
		plt.show()

	def graphQQPlot(self):
		logReturns = self.calcLogReturns()
		R = np.arange(-3.3, 3.3, 0.1)

		quantiles, LSFit = stats.probplot(logReturns, dist="norm")

		plt.scatter(quantiles[0], quantiles[1], color = 'blue', alpha = 0.5, label = 'Quantiles')
		plt.plot(R, LSFit[0] * R + LSFit[1], color = 'red', label = 'Best Fit Line')

		plt.ylabel('Ordered Values')
		plt.xlabel('Theoretical quantiles')
		plt.legend(loc = 2)
		plt.title('Q-Q plot for ' + self.quote, fontsize = 18)

		plt.show()