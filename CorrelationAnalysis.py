import numpy as np
import pandas as pd

from Stock import *

def calcCorrMatrix():
	symbolsIndexes = []
	ReturnsData = []

	for symbol in openSymbolsFile('DJI'):
		closeDF = pd.read_csv('hist_data/' + symbol + '.dat')['Adj Close']
		logReturns = np.log(closeDF / closeDF.shift(1)).dropna()
		length = len(logReturns)

		if(length >= 200):
			ReturnsData.append(logReturns)
			symbolsIndexes.append(symbol)

	D = pd.DataFrame(np.corrcoef(ReturnsData), columns = symbolsIndexes, index = symbolsIndexes)
	D.to_html('correlation_matrix.html')

	return D, symbolsIndexes

def getUniqueStocks(perc):
	corrMatrix, symIndexes = calcCorrMatrix()
	uncorrPairs = []

	for i, row in corrMatrix.iterrows():
		for sym in symIndexes:
			if -perc <= row[sym] <= perc:
				uncorrPairs.append(sym)

	uncorrPairs = list(set(uncorrPairs))

	return [Stock(s) for s in uncorrPairs]

def main():
	pass

if __name__ == '__main__':
	main()