# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.optimize import brentq

import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

def getRiskFreeRate():
	RF = {}
	D = pd.read_html('https://www.treasury.gov/resource-center/data-chart-center/interest-rates/Pages/TextView.aspx?data=yield')[1]

	RF['1M'] = round(float(np.array(D[1])[-1]) / 100, 4)
	RF['2M'] = round(float(np.array(D[2])[-1]) / 100, 4)
	RF['3M'] = round(float(np.array(D[3])[-1]) / 100, 4)
	RF['6M'] = round(float(np.array(D[4])[-1]) / 100, 4)
	RF['1Y'] = round(float(np.array(D[5])[-1]) / 100, 4)
	RF['2Y'] = round(float(np.array(D[6])[-1]) / 100, 4)
	RF['3Y'] = round(float(np.array(D[7])[-1]) / 100, 4)
	RF['5Y'] = round(float(np.array(D[8])[-1]) / 100, 4)
	RF['7Y'] = round(float(np.array(D[9])[-1]) / 100, 4)
	RF['10Y'] = round(float(np.array(D[10])[-1]) / 100, 4)
	RF['20Y'] = round(float(np.array(D[11])[-1]) / 100, 4)
	RF['30Y'] = round(float(np.array(D[12])[-1]) / 100, 4)

	return RF

def FV(P, i, n, m, contComp = False):
	if contComp:
		return P * np.exp(n * i)
	else:
		return P * ((1 + i/m) ** (m * n))

def PV(F, i, n, m, contComp = False):
	if contComp:
		return F * (1 / np.exp(n * i))
	else:
		return F * (1 / ((1 + i/m) ** (m * n)))

def PVOfAnnuity(C, i, n, m):
	return C * ((1 - (1 + i/m) ** -(n * m)) / i/m)

def graphYieldCurve():
	maturities = [1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]
	yields = list(getRiskFreeRate().values())

	plt.plot(maturities, yields, linewidth = 2.0, color = 'blue', label = 'Yield Curve')
	plt.scatter(maturities, yields, color = 'red', s = 50, label = 'Security')
	plt.xlabel('Maturitiy(Years)')
	plt.ylabel('Yield')
	plt.title('Yield Curve of US Treasury securities')
	plt.legend(loc = 2)
	plt.show()

class USTreasurySecurity():
	def __init__(self, parValue, maturity):
		self.parValue = parValue
		self.maturity = maturity

	def getParValue(self):
		return self.parValue

	def getMaturity(self):
		return self.maturity

class ZeroCouponBond(USTreasurySecurity):
	def __init__(self, parValue, maturity):
		super().__init__(parValue, maturity)

	def calcDiscountYield(self, purchasePrice, contComp = False):
		if contComp:
			discountYield = np.log(self.parValue / purchasePrice) * (365 / self.maturity)
		else:
			discountYield = ((self.parValue - purchasePrice) / self.parValue) * (365 / self.maturity)

		return discountYield

class CouponBond(USTreasurySecurity):
	def __init__(self, parValue, i, maturity, m = 2):
		super().__init__(parValue, maturity)
		self.i = i
		self.m = m

		self.coupon = self.parValue * (self.i / self.m)

	def getCoupon(self):
		return self.coupon

	def calcYieldToMaturity(self, P):
		n = self.maturity * self.m
		C = self.coupon
		F = self.parValue

		fun = lambda y: C * ((1 - (1 + y) ** (-n)) / y) + (F / (1 + y) ** n) - P

		return round(brentq(fun, 0.0001, 1), 5)

	def calcPrice(self, y):
		n = self.maturity * self.m
		C = self.coupon
		F = self.parValue

		P = C * ((1 - (1 + y) ** (-n)) / y) + (F / (1 + y) ** n)

		return P