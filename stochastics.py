# -*- coding: utf-8 -*-

import numpy as np

def simulateRandomWalk(N):
	R = np.random.binomial(1, 0.5, N - 1)
	R = np.array([-1 if i == 0 else i for i in R])

	return np.concatenate((np.array([0]), R))

def main():
	pass

if __name__ == '__main__':
	main()