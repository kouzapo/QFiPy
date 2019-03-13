# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.misc import derivative

import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

def get_risk_free_rate():
	RF = {}
	D = pd.read_html('https://www.treasury.gov/resource-center/data-chart-center/interest-rates/Pages/TextView.aspx?data=yield')[1]

	RF['1M'] = round(float(D['1 mo'].iloc[-1]) / 100, 4)
	RF['2M'] = round(float(D['2 mo'].iloc[-1]) / 100, 4)
	RF['3M'] = round(float(D['3 mo'].iloc[-1]) / 100, 4)
	RF['6M'] = round(float(D['6 mo'].iloc[-1]) / 100, 4)
	RF['1Y'] = round(float(D['1 yr'].iloc[-1]) / 100, 4)
	RF['2Y'] = round(float(D['2 yr'].iloc[-1]) / 100, 4)
	RF['3Y'] = round(float(D['3 yr'].iloc[-1]) / 100, 4)
	RF['5Y'] = round(float(D['5 yr'].iloc[-1]) / 100, 4)
	RF['7Y'] = round(float(D['7 yr'].iloc[-1]) / 100, 4)
	RF['10Y'] = round(float(D['10 yr'].iloc[-1]) / 100, 4)
	RF['20Y'] = round(float(D['20 yr'].iloc[-1]) / 100, 4)
	RF['30Y'] = round(float(D['30 yr'].iloc[-1]) / 100, 4)

	return RF

def FV(P, i, n, contComp = False):
	if contComp:
		return P * np.exp(n * i)
	else:
		return P * ((1 + i) ** n)

def calc_discount_factor(i, N, contComp = False):
	if contComp:
		return (1 / np.exp(N * i))
	else:
		return (1 / ((1 + i) ** N))

def plot_yield_curve():
	maturities = [1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]
	yields = list(get_risk_free_rate().values())

	plt.plot(maturities, yields, linewidth = 1.5, color = 'blue', marker = 'o', label = 'Yield Curve')
	plt.xlabel('Maturitiy (Years)')
	plt.ylabel('Yield')
	plt.title('Yield Curve of US Treasury securities')
	plt.legend(loc = 2)
	plt.show()

class USTreasurySecurity():
	def __init__(self, par_value, maturity):
		self.par_value = par_value
		self.maturity = maturity

	def getParValue(self):
		return self.par_value

	def getMaturity(self):
		return self.maturity

class ZeroCouponBond(USTreasurySecurity):
	def __init__(self, par_value, maturity):
		super().__init__(par_value, maturity)

	def calcDiscountYield(self, purchase_price):
		F = self.par_value
		n = self.maturity

		y = (F / purchase_price) ** (1 / n) - 1

		return round(y, 5)

	def calcPrice(self, y):
		pass

class CouponBond(USTreasurySecurity):
	def __init__(self, par_value, c, maturity, m = 2):
		super().__init__(par_value, maturity)
		self.c = c
		self.m = m

		self.periods = self.maturity * self.m
		self.coupon = self.par_value * (self.c / self.m)

	def getCoupon(self):
		return self.coupon

	def calcYieldToMaturity(self, purchase_price):
		n = self.maturity * self.m
		C = self.coupon
		F = self.par_value

		N = np.array([t for t in range(1, n + 1)])

		fun = lambda y: (C * calc_discount_factor(y / self.m, N)).sum() + (F / (1 + y / self.m) ** n) - purchase_price
		YTM = round(brentq(fun, 0.0001, 1), 5)

		return YTM

	def calcPrice(self, y):
		n = self.periods
		C = self.coupon
		F = self.par_value

		N = np.array([t for t in range(1, n + 1)])

		P = (C * calc_discount_factor(y / self.m, N)).sum() + F / (1 + y / self.m) ** n

		return round(P, 5)

	def calcMacaulayDuration(self, y):
		n = self.periods
		C = self.coupon
		F = self.par_value

		N = np.array([t for t in range(1, n + 1)])
		P = lambda y: (C * calc_discount_factor(y / self.m, N)).sum() + F / (1 + y / self.m) ** n
		deriv = derivative(P, x0 = y, dx = 1e-6, n = 1)

		MacD = -(1 + y/self.m) * deriv / self.calcPrice(y)

		return round(MacD, 5)

	def calcModifiedDuration(self, y):
		MacD = self.calcMacaulayDuration(y)

		return round(MacD / ((1 + y / self.m)), 5)

	def calcConvexity(self, y):
		n = self.periods
		C = self.coupon
		F = self.par_value

		N = np.array([t for t in range(1, n + 1)])
		P = lambda y: (C * calc_discount_factor(y / self.m, N)).sum() + F / (1 + y / self.m) ** n
		deriv = derivative(P, x0 = y, dx = 1e-6, n = 2)

		convexity = deriv / self.calcPrice(y)

		return round(convexity, 5)

	def approximatePriceChange(self, y, dx = 0.01):
		ModD = self.calcModifiedDuration(y)
		convexity = self.calcConvexity(y)
		P = self.calcPrice(y)

		percent_change = -ModD * dx + 0.5 * convexity * dx ** 2
		price_change = P * percent_change

		return {'percent_change': round(percent_change, 5), 'price_change': round(price_change, 5)}

	def plotPriceBehavior(self):
		yields = np.arange(0.01, 0.5, 0.01)
		prices = np.array([self.calcPrice(y) for y in yields])

		plt.plot(yields, prices, color = 'blue', linewidth = 2.0, label = 'Price vs Yield')
		plt.xlabel('Yield')
		plt.ylabel('Price')
		plt.title('Price Behavior')
		plt.legend(loc = 1)

		plt.show()