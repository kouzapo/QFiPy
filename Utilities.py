import pickle
import psutil
import datetime as dt

import numpy as np
import pandas as pd

def getRiskFreeRate():
	RF = {}
	D = pd.read_html('https://www.treasury.gov/resource-center/data-chart-center/interest-rates/Pages/TextView.aspx?data=yield')[1]

	RF['1M'] = round(float(np.array(D[1])[-1]) / 100, 4)
	RF['3M'] = round(float(np.array(D[2])[-1]) / 100, 4)
	RF['6M'] = round(float(np.array(D[3])[-1]) / 100, 4)
	RF['1Y'] = round(float(np.array(D[4])[-1]) / 100, 4)
	RF['2Y'] = round(float(np.array(D[5])[-1]) / 100, 4)
	RF['3Y'] = round(float(np.array(D[6])[-1]) / 100, 4)
	RF['5Y'] = round(float(np.array(D[7])[-1]) / 100, 4)
	RF['7Y'] = round(float(np.array(D[8])[-1]) / 100, 4)
	RF['10Y'] = round(float(np.array(D[9])[-1]) / 100, 4)
	RF['20Y'] = round(float(np.array(D[10])[-1]) / 100, 4)
	RF['30Y'] = round(float(np.array(D[11])[-1]) / 100, 4)

	return RF

def readFromSer(fileName):
	inFile = open(fileName, 'rb')

	return pickle.load(inFile)

def writeToSer(obj, fileName):
	outFile = open(fileName, 'wb')
	pickle.dump(obj, outFile)

	outFile.close()

def getDJISymbols():
	f = open('DJI_symbols.dat', 'w')
	DJIA_list = pd.read_html('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average')

	for symbol in DJIA_list[1][2][1:]:
		f.write(symbol + '\n')

	f.close()

def getGSPCSymbols():
	f = open('GSPC_symbols.dat', 'w')
	SPX_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')

	for symbol in SPX_list[0][0][1:]:
		f.write(symbol + '\n')

	f.close()

def getGDAXISymbols():
	f = open('GDAXI_symbols.dat', 'w')
	GDAXI_list = pd.read_html('https://en.wikipedia.org/wiki/DAX')

	for symbol in GDAXI_list[2][3][1:]:
		f.write(symbol + '.DE\n')

	f.close()

def openSymbolsFile(index):
	f = open(index + '_symbols.dat', 'r')
	symbols = []

	for symbol in f:
		symbol = symbol.strip()
		symbols.append(symbol)

	f.close()

	return symbols

def getTime():
	return dt.datetime.now().strftime("%H:%M:%S")

def progressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
	percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
	filledLength = int(length * iteration // total)
	bar = fill * filledLength + '-' * (length - filledLength)

	print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')

	if iteration == total:
		print()

def main():
	pass

if __name__ == '__main__':
	main()