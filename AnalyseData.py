import time

from CorrelationAnalysis import *
from Utilities import *
from Graphs import *
from MonteCarlo import *
from Stock import *
from Index import *
from Portfolio import *

st = time.time()

print(Stock('IBM').calcBeta(Index('GSPC'), True))


print("Execution time: " + str(time.time() - st))