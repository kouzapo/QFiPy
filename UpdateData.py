import os
import pandas as pd
import pandas_datareader.data as pdr
import datetime as dt

def updateDataMenu():
	while True:
		os.system('clear')

		print("Update Data")
		print('=' * 66)

		choice = input("1. Update historical data\n2. Update financial statements\n3. Both\n4. Back\n\nChoice: ")

		os.system('clear')

		if int(choice) == 1:
			removeData('hist_data/')
			start, end = getDates(1)

			print("Downloading historical data...")

			err = getHistoricalData(start, end)

			getIndexData('DJI', start, end)
			getIndexData('IXIC', start, end)
			getIndexData('GSPC', start, end)

			f = open('hist_data/' + end, 'w')
			f.write(str(err))
			f.close
			break

		elif int(choice) == 2:
			removeData('financial_statements/')
			
			print("Downloading financial statements...")

			getFinancialStatements()
			break

		elif int(choice) == 3:
			removeData('hist_data/')
			start, end = getDates(1)
			getHistoricalData(start, end)

			print('')

			removeData('financial_statements/')
			getFinancialStatements()
			break

		elif int(choice) == 4:
			system.os('exit')

def openSymbolsFile(index):
	f = open(index + '_symbols.dat', 'r')
	symbols = []

	for symbol in f:
		symbol = symbol.strip()
		symbols.append(symbol)

	f.close()

	return symbols

def getDates(n):
	end = str(dt.datetime.now().year) + '-' + str(dt.datetime.now().month) + '-' + str(dt.datetime.now().day)
	start = dt.datetime.now() - dt.timedelta(days = n * 365)
	start = str(start.year) + '-' + str(start.month) + '-' + str(start.day)

	return start, end

def getHistoricalData(start, end):
	symbols = openSymbolsFile('GSPC')
	l = len(symbols)
	err = 0
	i = 0

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

def getIndexData(index, start, end):
	while True:
		try:
			hist_dt = pdr.DataReader('^' + index, 'yahoo', start, end)
			hist_dt.to_csv('hist_data/' + index + '.dat')
			print(index)
			break

		except Exception:
			pass
			print("---ERROR---")

def getFinancialStatements():
	symbols = openSymbolsFile('DJI')
	l = len(symbols)

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
	updateDataMenu()

if __name__ == '__main__':
	main()