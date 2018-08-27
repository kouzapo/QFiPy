import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import style

from sklearn.linear_model import LinearRegression

#style.use('ggplot')

def graphSimulatedEfficientFrontier(ex, std):
	plt.scatter(std, ex, s = 12, alpha = 0.4, color = 'blue')
	plt.grid(True)
	plt.ylabel("Expected return")
	plt.xlabel("Standard deviation")
	plt.title("Simulated Efficient Frontier")
	plt.show()