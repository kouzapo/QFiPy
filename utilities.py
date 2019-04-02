# -*- coding: utf-8 -*-

import os
import datetime as dt

import dill
import pandas as pd

def write_to_ser(obj, fileName):
	outFile = open(fileName, 'wb')
	dill.dump(obj, outFile)

	outFile.close()

def read_from_ser(fileName):
	inFile = open(fileName, 'rb')
	obj = dill.load(inFile)

	return obj

def get_DJI_symbols():
	f = open('data/symbols_files/DJI_symbols.dat', 'w')
	DJI_list = pd.read_html('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average')

	for symbol in DJI_list[1]['Symbol']:
		f.write(symbol + '\n')

	f.close()

def get_GSPC_symbols():
	f = open('data/symbols_files/GSPC_symbols.dat', 'w')
	GSPC_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')

	for symbol in GSPC_list[0]['Symbol']:
		f.write(symbol + '\n')

	f.close()

def get_GDAXI_symbols():
	f = open('data/symbols_files/GDAXI_symbols.dat', 'w')
	GDAXI_list = pd.read_html('https://en.wikipedia.org/wiki/DAX')

	for symbol in GDAXI_list[2]['Ticker symbol']:
		f.write(symbol + '.DE\n')

	f.close()

def open_symbols_file(index):
	f = open('data/symbols_files/' + index + '_symbols.dat', 'r')
	symbols = [symbol.strip() for symbol in f]

	f.close()

	return symbols

def open_sectors_file(sector):
	f = open('GSPC_sectors\\' + sector + '.dat', 'r')
	symbols = [symbol.strip() for symbol in f]

	f.close()

	return symbols

def get_directory_size(directory, MB = True):
	total = 0

	for dirpath, dirnames, filenames in os.walk(directory):
		for f in filenames:
			fp = os.path.join(dirpath, f)

			total += os.path.getsize(fp)

	if MB:
		return total / (1024 ** 2)
	else:
		return total

def get_current_time():
	return dt.datetime.now().strftime("%H:%M:%S")

def progress_bar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
	percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
	filled_length = int(length * iteration // total)
	bar = fill * filled_length + '-' * (length - filled_length)

	print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')

	if iteration == total:
		print()

def main():
	pass

if __name__ == '__main__':
	main()