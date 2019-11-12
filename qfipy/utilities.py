#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains some utility functions used by other modules in the package.
"""

import os
import datetime as dt

import dill
import pandas as pd

__author__ = "Apostolos Kouzoukos"
__license__ = "MIT"
__email__ = "kouzoukos97@gmail.com"
__status__ = "Development"

def save(obj, filename):
	"""
	This function serializes an object and saves it to a file.

	Parametes:
	---------
		obj: The object to be serialized.
		filename: str, the name of the serialized file to be saved.
	"""

	output_file = open(filename, 'wb')
	dill.dump(obj, output_file)

	output_file.close()

def load(filename):
	"""
	This function opens a serialized file and returns the object.

	Parameters:
	----------
		filename: str, the name of the serialized file to be opened.
	Returns:
	-------
		obj: The object contained in the serialized file.
	"""

	input_file = open(filename, 'rb')
	obj = dill.load(input_file)

	return obj

def get_DJI_symbols():
	"""
	This function downloads the symbols of the Dow Jones index and saves them in a .dat file.
	"""

	f = open('data/symbols_files/DJI_symbols.dat', 'w')
	DJI_list = pd.read_html('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average')

	for symbol in DJI_list[1]['Symbol']:
		f.write(symbol + '\n')

	f.close()

def get_GSPC_symbols():
	"""
	This function downloads the symbols of the S&P 500 index and saves them in a .dat file.
	"""

	f = open('data/symbols_files/GSPC_symbols.dat', 'w')
	GSPC_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')

	for symbol in GSPC_list[0]['Symbol']:
		f.write(symbol + '\n')

	f.close()

def get_GDAXI_symbols():
	"""
	This function downloads the symbols of the DAX index and saves them in a .dat file.
	"""

	f = open('data/symbols_files/GDAXI_symbols.dat', 'w')
	GDAXI_list = pd.read_html('https://en.wikipedia.org/wiki/DAX')

	for symbol in GDAXI_list[2]['Ticker symbol']:
		f.write(symbol + '.DE\n')

	f.close()

def open_symbols_file(index):
	"""
	This function accept an index symbol as parameter, opens the corresponding file
	and returns a list  of the symbols.

	Parameters:
	----------
		index: str, the index symbol.
	Returns:
	-------
		symbols: list of symbols.
	"""

	f = open('data/symbols_files/' + index + '_symbols.dat', 'r')
	symbols = [symbol.strip() for symbol in f]

	f.close()

	return symbols

def get_directory_size(directory, MB = True):
	"""
	This function calculates and returns the size of the files in a directory.

	Parameters:
	----------
		directory: str, the name of the directory.
		MB: boolean, if set ot True, returns the size in Megabytes, else in bytes.
		Defaults to True.
	Returns:
	-------
		total: float, the size of the directory.
	"""

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
	"""
	This function returns the current time in HH:MM:SS format.

	Returns:
	-------
		current_time: str, the current time.
	"""

	current_time = dt.datetime.now().strftime("%H:%M:%S")

	return current_time

def progress_bar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
	"""
	This function implements a progress bar for iterative tasks.
	
	Parameters:
	----------
		iteration: int, the current iteration.
		total: int, the total number of iterations.
		prefix: str, defaults to ''
		suffix: str, defaults to ''
		decimals: int, the number of decimals of the percent of completion. Defaults to 1.
		length: int, the length of the progress bar. Defaults to 100
		fill: str, defaults to █
	"""

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
