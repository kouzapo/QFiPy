import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import style

from sklearn.linear_model import LinearRegression

#style.use('ggplot')

def graphSimulatedEfficientFrontier(ex, std):
	plt.scatter(std, ex, s = 10, alpha = 0.4)
	plt.grid(True)
	plt.ylabel("Expected return")
	plt.xlabel("Standard deviation")
	plt.title("Simulated Efficient Frontier")
	plt.show()

def graphMinVarLine(ex, std):
	plt.plot(std, ex)
	plt.grid(True)
	plt.ylabel("Expected return")
	plt.xlabel("Standard deviation")
	plt.title("Minimum Variance Line")
	plt.show()

def graphLinearRegressionOfStock(quote):
	closeDF = np.array(pd.read_csv('hist_data/' + quote + '.dat')['Adj Close'])
	dates = np.arange(0, len(closeDF), 1)

	closeDF = np.reshape(closeDF, (len(closeDF), 1))
	dates = np.reshape(dates, (len(dates), 1))

	reg = LinearRegression()
	reg.fit(dates, closeDF)

	plt.scatter(dates, closeDF, color = 'blue', s = 15, alpha = 1, label = "Price")
	plt.plot(dates, reg.coef_ * dates + reg.intercept_, color = 'red', linewidth = 2, label = "Fitting line")
	plt.legend(loc = 2)
	plt.ylabel("Price")
	plt.xlabel("Time")
	plt.title("Linear Regression" + ' | ' + quote)
	plt.show()
