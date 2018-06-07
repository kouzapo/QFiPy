import numpy as np
import pandas as pd

class Stock:

	def __init__(self, quote):
		self.quote = quote
		self.weight = 0

	def getQuote(self):
		return self.quote

	def getWeight(self):
		return self.weight

	def setWeight(self, weight):
		self.weight = weight

	def getIncomeStatement(self):
		return pd.read_csv('financial_statements/inc_' + self.quote + '.dat')

	def getBalanceSheet(self):
		return pd.read_csv('financial_statements/bal_' + self.quote + '.dat')

	def getEPS(self):
		return float(pd.read_html('https://finance.yahoo.com/quote/' + self.quote + '?p=' + self.quote)[1][1][3])

	def getPE(self):
		return float(pd.read_html('https://finance.yahoo.com/quote/' + self.quote + '?p=' + self.quote)[1][1][2])

	def calcIndicators(self):
		income_statement = self.getIncomeStatement()
		balance_sheet = self.getBalanceSheet()

		inventory = float(balance_sheet.iloc[4])
		total_current_assets = float(balance_sheet.iloc[6])
		total_assets = float(balance_sheet.iloc[14])
		total_current_liabilities = float(balance_sheet.iloc[18])
		total_liabilities = float(balance_sheet.iloc[24])
		equity = float(balance_sheet.iloc[33])

		gross_profit = float(income_statement.iloc[2])
		net_income = float(income_statement.iloc[20])

		#Liquidity
		current_ratio = total_current_assets / total_current_liabilities
		quick_ratio = (total_current_assets - inventory) / total_current_liabilities
		net_working_capital = total_current_assets - total_current_liabilities

		print(quick_ratio)

		#Dept
		dept_ratio = total_liabilities / total_assets

	def calcLogReturns(self):
		close_df = pd.read_csv('hist_data/' + self.quote + '.dat')['Adj Close']
		log_returns = np.log(close_df / close_df.shift(1))

		return log_returns