import pickle

def writeToSer(obj, fileName):
	outFile = open(fileName, 'wb')
	pickle.dump(obj, outFile)
	outFile.close()

def loadSer(fileName):
	inFile = open(fileName, 'rb')

	return pickle.load(inFile)

def main():
	pass

if __name__ == '__main__':
	main()