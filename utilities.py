# -*- coding: utf-8 -*-

"""
Utilities module.

This module provides some utility functions that can be used by other modules.
"""

import datetime as dt

import dill
import psutil

import numpy as np
import pandas as pd

def getRiskFreeRate():
	"""
	Returns the yields of US treasury securities.

	Returns
	-------
	RF: dict
		A dict object of risk free rates.
	"""

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

def writeToSer(obj, fileName):
	"""
	Makes an object serializable using the 'dill' module. Saves the serializable file in the current working directory with .ser extension.

	Parameters
	----------
	obj: Object
		The object to be serialized.

	fileName: str
		The name of the serialized object.
	"""

	outFile = open(fileName, 'wb')
	dill.dump(obj, outFile)

	outFile.close()

def readFromSer(fileName):
	"""
	Reads a serialized file and returns the object contained.

	Parameters
	----------
	fileName: str
		The name of the serialized object.

	Returns
	-------
	obj: Object
		The object contained in the serialized file.
	"""

	inFile = open(fileName, 'rb')
	obj = dill.load(inFile)

	return obj

def getDJISymbols():
	"""
	Downloads the symbols of the Dow Jones stocks and saves them in a .dat file.
	"""

	f = open('DJI_symbols.dat', 'w')
	DJIA_list = pd.read_html('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average')

	for symbol in DJIA_list[1][2][1:]:
		f.write(symbol + '\n')

	f.close()

def getGSPCSymbols():
	"""
	Downloads the symbols of the S&P500 stocks and saves them in a .dat file.
	"""

	f = open('GSPC_symbols.dat', 'w')
	GSPC_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')

	for symbol in GSPC_list[0][0][1:]:
		f.write(symbol + '\n')

	f.close()

def getGDAXISymbols():
	"""
	Downloads the symbols of the DAX stocks and saves them in a .dat file.
	"""

	f = open('GDAXI_symbols.dat', 'w')
	GDAXI_list = pd.read_html('https://en.wikipedia.org/wiki/DAX')

	for symbol in GDAXI_list[2][3][1:]:
		f.write(symbol + '.DE\n')

	f.close()

def openSymbolsFile(index):
	"""
	Opens a text file which contains the stock symbols of a given index and returns a list of them.

	Parameters
	----------
	index: str
		The symbol of the index.

	Returns
	-------
	symbols: list
		A list of the symbols of the given index.
	"""

	f = open(index + '_symbols.dat', 'r')
	symbols = [symbol.strip() for symbol in f]

	f.close()

	return symbols

def openSectorFile(sector):
	"""
	Opens a text file which contains the symbols of stocks belonging in a given sector and returns a list of them.

	Parameters
	----------
	index: str
		The sector file name.

	Returns
	-------
	symbols: list
		A list of the symbols of the given sector.
	"""

	f = open('GSPC_sectors\\' + sector + '.dat', 'r')
	symbols = [symbol.strip() for symbol in f]

	f.close()

	return symbols

def getTime():
	"""
	Returns the current time in a HH:MM:SS format.

	Returns
	-------
	currTime: str
		The current time.
	"""

	currTime = dt.datetime.now().strftime("%H:%M:%S")

	return currTime

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