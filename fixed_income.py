# -*- coding: utf-8 -*-

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import style

from utilities import getRiskFreeRate

style.use('ggplot')

def FV(P, i, n, m, cont_comp = False):
	if cont_comp:
		return P * np.exp(n * i)
	else:
		return P * ((1 + i/m) ** (m * n))

def PV(F, i, n, m, cont_comp = False):
	if cont_comp:
		return F * (1 / np.exp(n * i))
	else:
		return F * (1 / ((1 + i/m) ** (m * n)))

def PVOfAnnuity(C, i, n, m):
	return C * ((1 - (1 + i/m) ** -(n * m)) / i/m)

def graphYieldCurve():
	maturities = [1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]
	yields = list(getRiskFreeRate().values())

	plt.plot(maturities, yields, linewidth = 2.0, color = 'blue', label = 'Yield Curve')
	plt.scatter(maturities, yields, color = 'red', s = 50, label = 'Maturity')
	plt.xlabel('Maturitiy(Years)')
	plt.ylabel('Yield')
	plt.title('Yield Curve of US Treasury securities')
	plt.legend(loc = 2)
	plt.show()

class USTreasurySecurity():
	def __init__(self, maturity):
		self.maturity = maturity

	def getMaturity(self):
		return self.maturity