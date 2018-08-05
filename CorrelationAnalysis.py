import numpy as np
import pandas as pd

def openSymbolsFile(index):
	f = open(index + '_symbols.dat', 'r')
	symbols = []

	for symbol in f:
		symbol = symbol.strip()
		symbols.append(symbol)

	f.close()

	return symbols

def calcCorrMatrix():
	symbols_indexes = []
	CloseData = []

	for symbol in openSymbolsFile('DJI'):
		closeDF = np.array(pd.read_csv('hist_data/' + symbol + '.dat')['Adj Close'])
		length = len(closeDF)

		if(length >= 200):
			CloseData.append(closeDF)
			symbols_indexes.append(symbol)

	D = pd.DataFrame(np.corrcoef(CloseData), columns = symbols_indexes, index = symbols_indexes)
	D.to_html('correlation_matrix.html')

	return D, symbols_indexes

def calcBeta(bnmrk):
	symbols_indexes = []
	beta = []

	benchmark = pd.read_csv('hist_data/' + bnmrk + '.dat')['Adj Close']
	benchmark = np.log(benchmark / benchmark.shift(1))

	for symbol in openSymbolsFile('DJI'):
		closeDF = pd.read_csv('hist_data/' + symbol + '.dat')['Adj Close']
		closeDF = np.log(closeDF / closeDF.shift(1))

		beta.append(closeDF.cov(benchmark) / benchmark.var())

	return np.array(beta)

def getUniqueSymbols(perc):
	corr_matrix, sym_indexes = calcCorrMatrix()
	uncorr_pairs = []

	for i, row in corr_matrix.iterrows():
		for sym in sym_indexes:
			if -perc <= row[sym] <= perc:
				uncorr_pairs.append(sym)

	return list(set(uncorr_pairs))

def main():
	pass

if __name__ == '__main__':
	main()