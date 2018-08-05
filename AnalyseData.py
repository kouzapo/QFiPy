import time
from scipy import stats

from CorrelationAnalysis import *
from Serialization import *
from MonteCarlo import *
from UpdateData import *
from Stock import *
from Index import *
from Portfolio import *

st = time.time()

i1 = Index('GSPC')
s1 = Stock('FOX')

'''ret = s1.calcLogReturns()
res = stats.normaltest(ret)
print(res)

m, std = stats.norm.fit(ret)

plt.hist(s1.calcLogReturns(), bins = 75, color = 'blue', density = True)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, m, std)

plt.plot(x, p, color = 'red', linewidth = 1.4)
plt.show()'''

#print(plt.style.available)

s1.graphPrices()







print("Execution time: " + str(time.time() - st))