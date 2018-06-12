import matplotlib.pyplot as plt

def graphSimulatedEfficientFrontier(ex, std):
	plt.scatter(std, ex, s = 10, alpha = 0.4)
	plt.grid(True)
	plt.ylabel("Expected return")
	plt.xlabel("Standard deviation")
	plt.title("Simulated Efficient Frontier")
	plt.show()

def graphMinVarLine(ex, std):
	plt.plot(std, ex)
	plt.grid(True)
	plt.ylabel("Expected return")
	plt.xlabel("Standard deviation")
	plt.title("Minimum Variance Line")
	plt.show()