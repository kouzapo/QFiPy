import pickle

def writeToSer(obj, file_name):
	out_file = open(file_name, 'wb')
	pickle.dump(obj, out_file)
	out_file.close()

def readFromSer(file_name):
	in_file = open(file_name, 'rb')

	return pickle.load(in_file)

def main():
	pass

if __name__ == '__main__':
	main()