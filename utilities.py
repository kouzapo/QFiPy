# -*- coding: utf-8 -*-

import datetime as dt

import dill
import psutil
import pandas as pd

def writeToSer(obj, fileName):
	outFile = open(fileName, 'wb')
	dill.dump(obj, outFile)

	outFile.close()

def readFromSer(fileName):
	inFile = open(fileName, 'rb')
	obj = dill.load(inFile)

	return obj

def getDJISymbols():
	f = open('DJI_symbols.dat', 'w')
	DJIA_list = pd.read_html('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average')

	for symbol in DJIA_list[1][2][1:]:
		f.write(symbol + '\n')

	f.close()

def getGSPCSymbols():
	f = open('GSPC_symbols.dat', 'w')
	GSPC_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')

	for symbol in GSPC_list[0][0][1:]:
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
	symbols = [symbol.strip() for symbol in f]

	f.close()

	return symbols

def openSectorFile(sector):
	f = open('GSPC_sectors\\' + sector + '.dat', 'r')
	symbols = [symbol.strip() for symbol in f]

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