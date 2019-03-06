#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import datetime as dt
from threading import Thread, Lock

import numpy as np
import pandas as pd
import pandas_datareader.data as pdr

from utilities import open_symbols_file, get_directory_size, progress_bar

class DataUpdater:
	def __getDates(self, years):
		end = str(dt.datetime.now().year) + '-' + str(dt.datetime.now().month) + '-' + str(dt.datetime.now().day)
		start = dt.datetime.now() - dt.timedelta(days = years * 365)
		start = str(start.year) + '-' + str(start.month) + '-' + str(start.day)

		return start, end

	def __removeData(self, directory):
		if not os.path.isdir(directory):
			os.mkdir(directory)

		for f in os.listdir(directory):
			path = os.path.join(directory, f)

			if os.path.isfile(path):
				os.remove(path)

	def __getHistoricalData(self, symList, start, end):
		for sym in symList:
			while True:
				try:
					histDF = pdr.DataReader(sym, 'yahoo', start, end)
					histDF.to_csv('hist_data/' + sym + '.dat')

					break

				except Exception:
					#print(sym)
					pass

	def __getFinancialStatements(self, symList):
		for sym in symList:
			try:
				income_statement = pd.read_html('https://finance.yahoo.com/quote/' + sym + '/financials?p=' + sym)[0][1]
				balance_sheet = pd.DataFrame(pd.read_html('https://finance.yahoo.com/quote/' + sym + '/balance-sheet?p=' + sym)[0][1])

				income_statement.to_csv('financial_statements/inc_' + sym + '.dat', index = False)
				balance_sheet.to_csv('financial_statements/bal_' + sym + '.dat', index = False)

			except Exception:
				pass

	def runStockDataUpdate(self, index, remove = True):
		if remove:
			self.__removeData('hist_data/')

		start, end = self.__getDates(2)

		stockSymbols = open_symbols_file(index)
		indicesSymbols = open_symbols_file('indices')

		n = len(stockSymbols)
		A = np.arange(0, n, n / 5, dtype = int)

		S = [stockSymbols[A[i]:A[i + 1]] for i in range(len(A) - 1)]
		S.append(stockSymbols[A[-1]:])

		T = [Thread(target = self.__getHistoricalData, args = (s, start, end)) for s in S]
		T.append(Thread(target = self.__getHistoricalData, args = (indicesSymbols, start, end)))

		print("Downloading historical data...")

		n += len(indicesSymbols)
		progress_bar(0, n, prefix = 'Progress:', length = 50)

		start = time.time()

		for t in T:
			t.start()

		'''for t in T:
			t.join()'''

		while len(os.listdir('hist_data')) != n:
			progress_bar(len(os.listdir('hist_data')), n, prefix = 'Progress:', length = 50)
			time.sleep(0.5)

		progress_bar(n, n, prefix = 'Progress:', length = 50)
		print()

		total_time = str(round(time.time() - start, 1))
		files_count = str(len(os.listdir('hist_data')))
		files_size = str(round(get_directory_size('hist_data'), 2))

		print("Total " + files_count + " files in " + total_time + ' sec (' + files_size + ' MB)')

	def runFinancialStatementsUpdate(self, index, remove = True):
		if remove:
			self.__removeData('financial_statements/')

		stockSymbols = open_symbols_file(index)

		n = len(stockSymbols)
		A = np.arange(0, n, n / 5)
		A = np.array([int(i) for i in A])

		S = [stockSymbols[A[i]:A[i + 1]] for i in range(len(A) - 1)]
		S.append(stockSymbols[A[-1]:])

		T = [Thread(target = self.__getFinancialStatements, args = (s, )) for s in S]

		print("Downloading financial statements...")

		for t in T:
			t.start()

		'''for t in T:
			t.join()'''

		n *= 2

		while len(os.listdir('financial_statements')) != (n - 12):
			progress_bar(len(os.listdir('financial_statements')), n, prefix = 'Progress:', length = 50)
			time.sleep(0.5)

		progress_bar(n, n, prefix = 'Progress:', length = 50)
		print()
		print("Complete.")

def main():
	#os.system('cls')

	d1 = DataUpdater()

	'''indexQuote = sys.argv[1]
	DataUpdater().runStockDataUpdate(indexQuote)'''

	d1.runStockDataUpdate('GSPC')
	#d1.runFinancialStatementsUpdate('GSPC')

	for _ in range(2):
		print('\a', end = '\r')

if __name__ == '__main__':
	main()