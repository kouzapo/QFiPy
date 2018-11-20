# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.misc import derivative

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

def FV(P, i, n, contComp = False):
	if contComp:
		return P * np.exp(n * i)
	else:
		return P * ((1 + i) ** n)

def calcDiscountFactor(i, N, contComp = False):
	if contComp:
		return (1 / np.exp(N * i))
	else:
		return (1 / ((1 + i) ** N))

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
	def __init__(self, parValue, c, maturity, m = 2):
		super().__init__(parValue, maturity)
		self.c = c
		self.m = m

		self.periods = self.maturity * self.m
		self.coupon = self.parValue * (self.c / self.m)

	def getCoupon(self):
		return self.coupon

	def calcYieldToMaturity(self, purchasePrice):
		n = self.maturity * self.m
		C = self.coupon
		F = self.parValue

		N = np.array([t for t in range(1, n + 1)])

		fun = lambda y: (C * calcDiscountFactor(y / self.m, N)).sum() + (F / (1 + y / self.m) ** n) - purchasePrice
		YTM = round(brentq(fun, 0.0001, 1), 5)

		return YTM

	def calcPrice(self, y):
		n = self.periods
		C = self.coupon
		F = self.parValue

		N = np.array([t for t in range(1, n + 1)])

		P = (C * calcDiscountFactor(y / self.m, N)).sum() + F / (1 + y / self.m) ** n

		return round(P, 5)

	def calcMacaulayDuration(self, y):
		n = self.periods
		C = self.coupon
		F = self.parValue

		N = np.array([t for t in range(1, n + 1)])
		P = lambda y: (C * calcDiscountFactor(y / self.m, N)).sum() + F / (1 + y / self.m) ** n
		deriv = derivative(P, x0 = y, dx = 1e-6, n = 1)

		MacD = -(1 + y/self.m) * deriv / self.calcPrice(y)

		return round(MacD, 4)

	def calcModifiedDuration(self, y):
		MacD = self.calcMacaulayDuration(y)

		return round(MacD / ((1 + y / self.m)), 4)

	def calcConvexity(self, y):
		n = self.periods
		C = self.coupon
		F = self.parValue

		N = np.array([t for t in range(1, n + 1)])
		P = lambda y: (C * calcDiscountFactor(y / self.m, N)).sum() + F / (1 + y / self.m) ** n
		deriv = derivative(P, x0 = y, dx = 1e-6, n = 2)

		convexity = deriv / self.calcPrice(y)

		return round(convexity, 4)

	def approximatePriceChange(self, y, dx = 0.01):
		ModD = self.calcModifiedDuration(y)
		convexity = self.calcConvexity(y)
		P = self.calcPrice(y)

		percentChange = -ModD * dx + 0.5 * convexity * dx ** 2
		priceChange = P * percentChange

		return {'percent_change': round(percentChange, 4), 'price_change': round(priceChange, 4)}

	def graphPriceBehavior(self):
		yields = np.arange(0.01, 0.5, 0.01)
		prices = np.array([self.calcPrice(y) for y in yields])

		plt.plot(yields, prices, color = 'blue', linewidth = 2.0, label = 'Price vs Yield')
		plt.xlabel('Yield')
		plt.ylabel('Price')
		plt.title('Price Behavior')
		plt.legend(loc = 1)
		plt.show()