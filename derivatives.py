# -*- coding: utf-8 -*-

import numpy as np

class Option:
	def __init__(self, asset, X):
		self.asset = asset

		self.strike_price = X

class CallOption(Option):
	def __init__(self, asset, X):
		Option.__init__(self, asset, X)

	def calcIntrinsicValue(self):
		S = self.asset.getCurrentPrice()
		X = self.strike_price

		return round(max(0, S - X), 5)

class PutOption(Option):
	def __init__(self, asset, X):
		Option.__init__(self, asset, X)

	def calcIntrinsicValue(self):
		S = self.asset.getCurrentPrice()
		X = self.strike_price

		return round(max(0, X - S), 5)