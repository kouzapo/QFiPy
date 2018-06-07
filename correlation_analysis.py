import numpy as np
import pandas as pd

def openSymbolsFile():
	f = open('djia_symbols.dat', 'r')
	symbols = []

	for symbol in f:
		symbol = symbol.strip()
		symbols.append(symbol)

	f.close()

	return symbols

def calcCorrMatrix():
	symbols_indexes = []
	CloseData = []

	for symbol in openSymbolsFile():
		closeDF = np.array(pd.read_csv('hist_data/' + symbol + '.dat')['Adj Close'])
		length = len(closeDF)

		if(length >= 200):
			CloseData.append(closeDF)
			symbols_indexes.append(symbol)

	D = pd.DataFrame(np.corrcoef(CloseData), columns = symbols_indexes, index = symbols_indexes)
	D.to_html('correlation_matrix.html')

	return D, symbols_indexes

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