import numpy as np
import pandas as pd
from scipy import stats

from Utilities import getRiskFreeRate

class Option:
	def __init__(self, stock):
		self.stock = stock

	def getStock(self):
		return self.stock

	def calcCallPrice(self, K, T):
		S0 = self.stock.getCurrentPrice()
		s = self.stock.calcStd()
		rf = getRiskFreeRate()

		d1 = (np.log(S0 / K) + (rf + 0.5 * s ** 2) * T) /(s * np.sqrt(T))
		d2 = (np.log(S0 / K) + (rf - 0.5 * s ** 2) * T) /(s * np.sqrt(T))

		#C0 = np.exp(-r * T) * np.sum(np.maximum(S[-1] - K, 0)) / I

		return (S0 * stats.norm.cdf(d1, 0, 1) - K * np.exp(-rf * T) * stats.norm.cdf(d2, 0, 1))


'''
S0 = s1.getCurrentPrice()
K = 150
T = 1/12
r = getRiskFreeRate()
s = s1.calcStd()
M = 50
dt = T / M
I = 250000

S = np.zeros((M + 1, I))
S[0] = S0

for t in range(1, M + 1):
	z = np.random.standard_normal(I)
	S[t] = S[t - 1] * np.exp((r - 0.5 * s ** 2) * dt + s * np.sqrt(dt) * z)

C0 = np.exp(-r * T) * np.sum(np.maximum(S[-1] - K, 0)) / I
print(C0)
'''