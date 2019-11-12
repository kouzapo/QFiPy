#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module implements several classes and functions that model 
the behavior of fixed income instruments such as US Treasury securities. 
Functions that construct and plot the yield curve are also implemented.
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.misc import derivative

import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

__author__ = "Apostolos Kouzoukos"
__license__ = "MIT"
__email__ = "kouzoukos97@gmail.com"
__status__ = "Development"

def get_yields():
	"""
	This function fetches the current yields of US Treasury securities
	and returns a dict with the values and the maturities as keys.
	Returns:
	-------
		yields: dict, a dictitonary object containing the yield for each maturiy.
	"""

	D = pd.read_html('https://www.treasury.gov/resource-center/data-chart-center/interest-rates/Pages/TextView.aspx?data=yield')[1]

	maturities_keys = ['1 mo', '2 mo', '3 mo', '6 mo', '1 yr', '2 yr', '3 yr', '5 yr', '7 yr', '10 yr', '20 yr', '30 yr']
	yields = {m: round(float(D[m].iloc[-1]) / 100, 4) for m in maturities_keys}

	return yields

def plot_yield_curve():
	"""
	This function plots the yield curve of US Treasury securities based on the
	current yields. 
	"""

	maturities_keys = ['1 mo', '2 mo', '3 mo', '6 mo', '1 yr', '2 yr', '3 yr', '5 yr', '7 yr', '10 yr', '20 yr', '30 yr']
	Y = get_yields()
	
	yields = [Y[m] for m in maturities_keys]
	maturities = [1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]

	plt.plot(maturities, yields, linewidth = 1.5, color = 'blue', marker = 'o', label = 'Yield Curve')
	plt.xlabel('Maturitiy (Years)')
	plt.ylabel('Yield')
	plt.title('Yield Curve of US Treasury securities')
	plt.legend(loc = 2)
	plt.show()

def calc_discount_factor(i, N, cont_comp = False):
	"""
	This function calculates the discount factor (present value) based on interest rate i and
	N number of years. The calculation based on continuous compound is also possible.

	Parameters:
	----------
		i: float, the interest rate.
		N: int, the number of years.
		cont_comp: bool, whether or not to use continuous compound. Defaults to False.
	Returns:
	-------
		discount_factor: float, the discount_factor calculated based on i and N.
	"""

	if cont_comp:
		discount_factor = (1 / np.exp(N * i))
	else:
		discount_factor = (1 / ((1 + i) ** N))

	return discount_factor

class USTreasurySecurity():
	"""
	This base class models the US Treasury security object and has two subclasses,
	the zero coupon bond and the coupon bond. 
	"""

	def __init__(self, par_value, maturity):
		"""
		The constructor of the USTreasurySecurity base class. 
		Parameters:
		----------
			par_value: float, the par value of the security.
			maturity: float, the maturity of the security.
		"""

		self.par_value = par_value
		self.maturity = maturity

class ZeroCouponBond(USTreasurySecurity):
	"""
	This subclass models the zero coupon bond and inherits the attributes of 
	par value and maturity from the USTreasurySecurity class.
	"""

	def __init__(self, par_value, maturity):
		"""
		The constructor of the ZeroCouponBond class. 

		Parameters:
		----------
			par_value: float, the par value of the security.
			maturity: float, the maturity of the security.
		"""

		super().__init__(par_value, maturity)

	def calcDiscountYield(self, purchase_price):
		"""
		This method calculates the discount yield of a zero coupon bond based
		on the purchase price.

		Parameters:
		----------
			purchase_price: float, the purchase price of the security.
		Returns:
		-------
			discount_yield: float, the discount yield of the security 
			based on the purchase price.
		"""

		F = self.par_value
		n = self.maturity

		discount_yield = round((F / purchase_price) ** (1 / n) - 1, 5)

		return discount_yield

class CouponBond(USTreasurySecurity):
	"""
	This subclass models the coupon bond and inherits the attributes of 
	par value and maturity from the USTreasurySecurity class. Common operations
	of coupond bonds are implemented such as the calculation on durations and convexity,
	the calculation of yield to maturity etc.
	"""

	def __init__(self, par_value, c, maturity, m = 2):
		"""
		The constructor of CouponBond. Additional attributes are needed like the coupon rate
		and the number of coupon payments per annum.

		Parameters:
		----------
			par_value: float, the par value of the security.
			c: float, the coupon rate.
			maturity: float, the maturity of the security.
			m: number of coupon payments per annum. 
		"""

		super().__init__(par_value, maturity)
		self.c = c
		self.m = m

		self.periods = self.maturity * self.m
		self.coupon = self.par_value * (self.c / self.m)

	def calcYieldToMaturity(self, purchase_price):
		"""
		This method calculates the yield to maturity given the purchase price.
		This is an optimization problem solved with the brentq function of scipy.optimize.

		Parameters:
		----------
			purchase_price: float, the purchase price of the coupon bond.
		Returns:
		-------
			YTM: float, the yield to maturity.
		"""

		n = self.maturity * self.m
		C = self.coupon
		F = self.par_value

		N = np.array([t for t in range(1, n + 1)])

		fun = lambda y: (C * calc_discount_factor(y / self.m, N)).sum() + (F / (1 + y / self.m) ** n) - purchase_price
		YTM = round(brentq(fun, 0.0001, 1), 5)

		return YTM

	def calcPrice(self, y):
		"""
		This method calculates the price of a coupon bond based on a yield.

		Parameters:
		----------
			y: float, the current yield.
		Returns:
		-------
			price: float, the price based on the given yield.
		"""

		n = self.periods
		C = self.coupon
		F = self.par_value

		N = np.array([t for t in range(1, n + 1)])

		price = (C * calc_discount_factor(y / self.m, N)).sum() + F / (1 + y / self.m) ** n
		price = round(price, 5)

		return price

	def calcMacaulayDuration(self, y):
		"""
		This method calculates the Macaulay duration of a coupon bond based on a yield.

		Parameters:
		----------
			y: float, the current yield.
		Returns:
		-------
			MacD: float, the Macaulay duration based on the given yield.
		"""

		n = self.periods
		C = self.coupon
		F = self.par_value

		N = np.array([t for t in range(1, n + 1)])
		P = lambda y: (C * calc_discount_factor(y / self.m, N)).sum() + F / (1 + y / self.m) ** n
		deriv = derivative(P, x0 = y, dx = 1e-6, n = 1)

		MacD = -(1 + y/self.m) * deriv / self.calcPrice(y)
		MacD = round(MacD, 5)

		return MacD

	def calcModifiedDuration(self, y):
		"""
		This method calculates the modified duration of a coupon bond based on a yield.

		Parameters:
		----------
			y: float, the current yield.
		Returns:
		-------
			modified_duration: float, the modified duration based on the given yield.
		"""

		MacD = self.calcMacaulayDuration(y)
		modified_duration = round(MacD / ((1 + y / self.m)), 5)

		return modified_duration

	def calcConvexity(self, y):
		"""
		This method calculates the convexity of a coupon bond based on a yield.
		
		Parameters:
		----------
			y: float, the current yield.
		Returns:
		-------
			convexity: float, the convexity based on the given yield.
		"""

		n = self.periods
		C = self.coupon
		F = self.par_value

		N = np.array([t for t in range(1, n + 1)])
		P = lambda y: (C * calc_discount_factor(y / self.m, N)).sum() + F / (1 + y / self.m) ** n
		deriv = derivative(P, x0 = y, dx = 1e-6, n = 2)

		convexity = deriv / self.calcPrice(y)
		convexity = round(convexity, 5)

		return convexity

	'''def approximatePriceChange(self, y, dx = 0.01):
		modified_duration = self.calcModifiedDuration(y)
		convexity = self.calcConvexity(y)
		P = self.calcPrice(y)

		percent_change = -modified_duration * dx + 0.5 * convexity * dx ** 2
		price_change = P * percent_change

		return {'percent_change': round(percent_change, 5), 'price_change': round(price_change, 5)}'''

	def plotPriceBehavior(self):
		"""
		This method plots the price behavior of the security based on random yields.
		"""

		yields = np.arange(0.01, 0.5, 0.01)
		prices = np.array([self.calcPrice(y) for y in yields])

		plt.plot(yields, prices, color = 'blue', linewidth = 2.0, label = 'Price vs Yield')
		plt.xlabel('Yield')
		plt.ylabel('Price')
		plt.title('Price Behavior')
		plt.legend(loc = 1)

		plt.show()
