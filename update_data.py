# -*- coding: utf-8 -*-

import os
import time
import datetime as dt
import threading as thrd

import numpy as np
import pandas as pd
import pandas_datareader.data as pdr

from utilities import openSymbolsFile, progressBar

def updateDataMenu():
	while True:
		os.system('cls')

		print("Update Data")
		print('=' * 75)

		choice = input("1. Update historical data\n2. Update financial statements\n3. Both\n4. Back\n\nChoice:")
		choice = choice.upper()

		os.system('cls')

		if int(choice) == 1:
			_removeData('hist_data/')
			start, end = _getDates()

			stockSymbols = openSymbolsFile('GSPC')
			indicesSymbols = openSymbolsFile('indices')

			l = len(stockSymbols)
			I = np.arange(0, l, l / 5)
			I = np.array([int(i) for i in I])

			S1 = stockSymbols[I[0]:I[1]]
			S2 = stockSymbols[I[1]:I[2]]
			S3 = stockSymbols[I[2]:I[3]]
			S4 = stockSymbols[I[3]:I[4]]
			S5 = stockSymbols[I[4]:]

			S = [S1, S2, S3, S4, S5]
			T = [thrd.Thread(target = _getHistoricalData, args = (s, start, end)) for s in S]
			T.append(thrd.Thread(target = _getHistoricalData, args = (indicesSymbols, start, end)))

			print("Downloading historical data...")

			l += len(indicesSymbols)
			progressBar(0, l, prefix = 'Progress:', length = 50)

			for t in T:
				t.start()

			'''for t in T:
				t.join()'''

			while len(os.listdir('hist_data')) != l:
				progressBar(len(os.listdir('hist_data')), l, prefix = 'Progress:', length = 50)
				time.sleep(0.5)

		elif int(choice) == 2:
			_removeData('financial_statements/')

			print("Downloading financial statements...")
			_getFinancialStatements()
			break

		elif int(choice) == 3:
			_removeData('hist_data/')
			start, end = _getDates()

			print("Downloading historical data...")

			print()

			_removeData('financial_statements/')

			print("Downloading financial statements...")
			_getFinancialStatements()
			break

		elif int(choice) == 4:
			break

def _getDates():
	end = str(dt.datetime.now().year) + '-' + str(dt.datetime.now().month) + '-' + str(dt.datetime.now().day)
	start = dt.datetime.now() - dt.timedelta(days = 5 * 365)
	start = str(start.year) + '-' + str(start.month) + '-' + str(start.day)

	return start, end

def _getHistoricalData(symList, start, end):
	for sym in symList:
		try:
			histDF = pdr.DataReader(sym, 'yahoo', start, end)
			histDF.to_csv('hist_data/' + sym + '.dat')

		except Exception:
			pass

def _getFinancialStatements():
	symbols = openSymbolsFile('GSPC')
	l = len(symbols)
	i = 0

	progressBar(0, l, prefix = 'Progress:', length = 50)

	for sym in symbols:
		income_statement = pd.read_html('https://finance.yahoo.com/quote/' + sym + '/financials?p=' + sym)[0][1]
		balance_sheet = pd.DataFrame(pd.read_html('https://finance.yahoo.com/quote/' + sym + '/balance-sheet?p=' + sym)[0][1])

		income_statement.to_csv('financial_statements/inc_' + sym + '.dat', index = False)
		balance_sheet.to_csv('financial_statements/bal_' + sym + '.dat', index = False)

		i += 1
		progressBar(i, l, prefix = 'Progress:', length = 50)

		#print(sym)

def _removeData(directory):
	for f in os.listdir(directory):
		path = os.path.join(directory, f)

		if os.path.isfile(path):
			os.remove(path)

def main():
	updateDataMenu()

if __name__ == '__main__':
	main()