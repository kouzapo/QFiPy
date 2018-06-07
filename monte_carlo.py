import numpy as np
import pandas as pd
from scipy import stats

def genRandomPortfolios(n, symbols):
	results = {}

	weights_record = []
	return_record = []
	std_record = []

	for i in range(n):
		print(i)
		weights = np.random.random(len(symbols))
		weights /= np.sum(weights)
		weights_record.append(weights)

		prtf_return, prtf_std = calcPortfolioPerf(weights, symbols)

		return_record.append(prtf_return)
		std_record.append(prtf_std)

	results['Weights'] = weights_record
	results['Return'] = return_record
	results['Std'] = std_record

	return pd.DataFrame(results)

def monteCarloCalculations(symbols, days):
	price_dict = {}

	for symbol in symbols:
		close_df = pd.read_csv('hist_data/' + symbol + '.dat')['Adj Close']
		last = close_df[len(close_df) - 1]
		log_returns = np.log(close_df / close_df.shift(1))

		price_series = []

		price = last * (1 + np.random.normal(0, log_returns.std()))
		price_series.append(price)

		for i in range(days - 1):
			price = price * (1 + np.random.normal(0, log_returns.std()))
			price_series.append(price)

		price_dict[symbol] = price_series

	return price_dict

def runSimulation(n, symbols):
	futurePrices = []

	for i in range(n):
		futurePrices.append(pd.DataFrame(monteCarloCalculations(symbols, 5)))
		print(i)

	return futurePrices

def main():
	pass

if __name__ == '__main__':
	main()