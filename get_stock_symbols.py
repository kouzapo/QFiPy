import pandas as pd

def getDJIASymbols():
	f = open('djia_symbols.dat', 'w')
	DJIA_list = pd.read_html('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average')

	for symbol in DJIA_list[1][2][1:]:
		f.write(symbol + '\n')

	f.close()

def getSPXSymbols():
	f = open('spx_symbols.dat', 'w')
	SPX_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')

	for symbol in SPX_list[0][0][1:]:
		f.write(symbol + '\n')

	f.close()

def main():
	getDJIASymbols()
	getSPXSymbols()

if __name__ == '__main__':
	main()