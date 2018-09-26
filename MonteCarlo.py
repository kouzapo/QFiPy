import numpy as np
import pandas as pd
from scipy import stats

from Stock import *
from Index import *
from Portfolio import *

def genRandomPortfolios(symbols, n):
	results = []

	for i in range(n):
		weights = np.random.random(len(symbols))
		weights /= np.sum(weights)

		stocks = []
		j = 0

		for s in symbols:
			stocks.append(Stock(s, weights[j]))
			j += 1

		results.append(Portfolio(stocks))

	return results

def simulateRandomWalk(N):
	R = np.random.binomial(1, 0.5, N)
	R = np.array([-1 if i == 0 else i for i in R])
	R[0] = 0

	return R.cumsum()

def simulateWienerProcess(N):
	dt = 1 / N

	for i in range(N):
		dW = np.sqrt(dt) * np.random.normal(0, 1, N)

	dW[0] = 0
	return dW.cumsum()

def simulatePrices(S0, std, rf, T, M, I):
	dt = T / M

	S = np.zeros((M + 1, I))
	S[0] = S0

	for t in range(1, M + 1):
		z = np.random.standard_normal(I)
		S[t] = S[t - 1] * np.exp((rf - 0.5 * std ** 2) * dt + std * np.sqrt(dt) * z)

	return S

def main():
	pass

if __name__ == '__main__':
	main()