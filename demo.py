#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time

from utilities import *
from equities import *
from fixed_income import *
from portfolio import *

st = time.time()

rf = 0.0266

#-----Equities-----
s = Stock('HD')
I = Index('^GSPC')

s.descriptiveStats()
print()

print('Annualized Expected Return:', round(s.calcExpReturn(), 4))
print('Annualized Standard Deviation:', round(s.calcStd(), 4))
print()

print('Sharpe Ratio:', round(s.calcSharpeRatio(rf), 4))
print('Beta:', round(s.calcBetaAlpha(I)['beta'], 4))

s.graphPrice()
s.graphLogReturns()
s.graphQQPlot()
s.graphCorrelation(I)
s.graphACF(100)
s.graphPACF(100)

#-----Portfolio-----
stocks = [Stock('MCD'), Stock('KO'), Stock('HD'), Stock('WMT')]
portfolio = StockPortfolio(stocks)

portfolio.calcMinVarAlloc()
portfolio.printSummary(portfolio.calcPerformance(rf))

portfolio.calcMinVarLine(mv = 0.17)
portfolio.printSummary(portfolio.calcPerformance(rf))

portfolio.maximizeSharpeRatio(rf)
portfolio.printSummary(portfolio.calcPerformance(rf))

portfolio.graphEfficientFrontier()
portfolio.graphSimulatedRandomProtfolios(N = 1000)

#-----Fixed Income-----
b = CouponBond(par_value = 1000, c = 0.09, maturity = 10)
YTM = 0.06

print('Coupon:', b.coupon)
print()

print('Yield to Maturity:', b.calcYieldToMaturity(purchase_price = 840))
print('Price:', b.calcPrice(y = YTM))
print()

print('Macaulay Duration:', b.calcMacaulayDuration(y = YTM))
print('Modified Duration:', b.calcModifiedDuration(y = YTM))
print('Convexity:', b.calcConvexity(y = YTM))

b.graphPriceBehavior()

print("Execution time: " + str(time.time() - st))