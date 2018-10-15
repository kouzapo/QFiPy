import time

from CorrelationAnalysis import *
from Utilities import *
from MonteCarlo import *
from Stock import *
from Option import *
from Index import *
from Portfolio import *
from UpdateData import *

st = time.time()

stocks = [Stock(s) for s in openSymbolsFile('GSPC')]
i1 = Index('^GSPC')

for s in stocks:
	print(s.calcBeta(i1))


print("Execution time: " + str(time.time() - st))