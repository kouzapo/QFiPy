import os
import pandas as pd
import pandas_datareader.data as pdr
import datetime as dt

def menu():
	while True:
		choice = input("'H' to update historical data, 'F' to update financial statements, 'B' for both: ")
		choice = choice.upper()

		if (choice == 'H'):
			removeHistoricalData()
			start, end = getDates()
			getHistoricalData(start, end)
			break

		elif (choice == 'F'):
			removeFinancialData()
			getFinancialStatements()
			break

		elif (choice == 'B'):
			removeHistoricalData()
			start, end = getDates()
			getHistoricalData(start, end)

			print("")

			removeFinancialData()
			getFinancialStatements()
			break

def openSymbolsFile():
	f = open('djia_symbols.dat', 'r')
	symbols = []

	for symbol in f:
		symbol = symbol.strip()
		symbols.append(symbol)

	f.close()

	return symbols

def getDates():
	end = str(dt.datetime.now().year) + '-' + str(dt.datetime.now().month) + '-' + str(dt.datetime.now().day)
	start = dt.datetime.now() - dt.timedelta(days = 365)
	start = str(start.year) + '-' + str(start.month) + '-' + str(start.day)

	return start, end

def getHistoricalData(start, end):
	symbols = openSymbolsFile()

	for sym in symbols:
		while True:
			try:
				hist_dt = pdr.DataReader(sym, 'yahoo', start, end)
				hist_dt.to_csv('hist_data/' + sym + '.dat')
				print(sym)
				break

			except Exception:
				print("---ERROR---")

def getFinancialStatements():
	symbols = openSymbolsFile()

	for sym in symbols:
		income_statement = pd.read_html('https://finance.yahoo.com/quote/' + sym + '/financials?p=' + sym)[0][1]
		balance_sheet = pd.DataFrame(pd.read_html('https://finance.yahoo.com/quote/' + sym + '/balance-sheet?p=' + sym)[0][1])

		income_statement.to_csv('financial_statements/inc_' + sym + '.dat', index = False)
		balance_sheet.to_csv('financial_statements/bal_' + sym + '.dat', index = False)

		print(sym)

def removeHistoricalData():
	symbols = openSymbolsFile()

	for sym in symbols:
		if os.path.isfile('hist_data/' + sym + '.dat'):
			os.remove('hist_data/' + sym + '.dat')

def removeFinancialData():
	symbols = openSymbolsFile()

	for sym in symbols:
		if os.path.isfile('financial_statements/inc_' + sym + '.dat'):
			os.remove('financial_statements/inc_' + sym + '.dat')

		if os.path.isfile('financial_statements/bal_' + sym + '.dat'):
			os.remove('financial_statements/bal_' + sym + '.dat')

def main():
	menu()

if __name__ == '__main__':
	main()