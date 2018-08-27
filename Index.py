import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import style

style.use('ggplot')

class Index:
	def __init__(self, quote):
		self.quote = quote

	def getQuote(self):
		return self.quote

	def getPrices(self, return_dates = False):
		DF = pd.read_csv('hist_data/' + self.quote + '.dat')

		closeDF = DF['Adj Close']
		dates = DF['Date']

		if return_dates:
			return np.array(closeDF), np.array(dates)
		else:
			return np.array(closeDF)

	def calcLogReturns(self):
		closeDF = pd.read_csv('hist_data/' + self.quote + '.dat')['Adj Close']
		logReturns = np.log(closeDF / closeDF.shift(1)).dropna()

		return np.array(logReturns)

	def graphPrices(self):
		closeDF, dates = self.getPrices(return_dates = True)
		dates = pd.to_datetime(dates)

		fig, ax = plt.subplots(1)
		fig.autofmt_xdate()
		plt.plot(dates, closeDF, color = 'blue', label = "Price")
		plt.title(self.getQuote())
		plt.xlabel("Date")
		plt.ylabel("Price")
		plt.legend(loc = 2)

		xfmt = mdates.DateFormatter('%Y-%m-%d')
		ax.xaxis.set_major_formatter(xfmt)
		ax.set_facecolor('tab:black')
		plt.show()