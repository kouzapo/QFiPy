import time

from CorrelationAnalysis import *
from Serialization import *
from MonteCarlo import *
from UpdateData import *
from Graphs import *
from Stock import *
from Portfolio import *

st = time.time()

symbols = getUniqueSymbols(0.015)
print(symbols)

stocks = [Stock(s) for s in symbols]

p1 = Portfolio(stocks)
p1.calcMinVarLine(0.1)
print(p1.calcPortfolioPerformance())


print(p1.getStocksWeights())

'''DJI_close = np.array(pd.read_csv('hist_data/DJI.dat')['Adj Close'])
symbols = openSymbolsFile('DJI')

D = {}

for s in symbols:
	D[s] = np.corrcoef(DJI_close, np.array(pd.read_csv('hist_data/' + s + '.dat')['Adj Close']))[0][1]

for s, v in D.items():
	print(s, v)'''









print("Execution time: " + str(time.time() - st))