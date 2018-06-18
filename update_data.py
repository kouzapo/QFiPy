import os
import pandas as pd
import pandas_datareader.data as pdr
import datetime as dt

def menu():
	while True:
		choice = input("'H' to update historical data, 'F' to update financial statements, 'B' for both: ")
		choice = choice.upper()

		if (choice == 'H'):
			removeData('hist_data/')
			start, end = getDates()
			err = getHistoricalData(start, end)

			f = open('hist_data/' + end, 'w')
			f.write(str(err))
			f.close
			break

		elif (choice == 'F'):
			removeData('financial_statements/')
			getFinancialStatements()
			break

		elif (choice == 'B'):
			removeData('hist_data/')
			start, end = getDates()
			getHistoricalData(start, end)

			print("")

			removeData('financial_statements/')
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
	err = 0

	for sym in symbols:
		while True:
			try:
				hist_dt = pdr.DataReader(sym, 'yahoo', start, end)
				hist_dt.to_csv('hist_data/' + sym + '.dat')
				print(sym)
				break

			except Exception:
				err += 1
				print("---ERROR---")

	return err

def getFinancialStatements():
	symbols = openSymbolsFile()

	for sym in symbols:
		income_statement = pd.read_html('https://finance.yahoo.com/quote/' + sym + '/financials?p=' + sym)[0][1]
		balance_sheet = pd.DataFrame(pd.read_html('https://finance.yahoo.com/quote/' + sym + '/balance-sheet?p=' + sym)[0][1])

		income_statement.to_csv('financial_statements/inc_' + sym + '.dat', index = False)
		balance_sheet.to_csv('financial_statements/bal_' + sym + '.dat', index = False)

		print(sym)

def removeData(dir):
	for f in os.listdir(dir):
		path = os.path.join(dir, f)

		if os.path.isfile(path):
			os.remove(path)

def main():
	menu()

if __name__ == '__main__':
	main()