import numpy as np
import pandas as pd
from scipy.optimize import broyden1

"""

Python module written for Columbia University's Financial Engineering and Risk Management Part I course offered on Coursera.

Inspired by https://github.com/govinda18/Financial-Engineering-and-Risk-Management-Part-I

Covers the setting up and calibration of binomial models for various asset classes, as well as pricing of mortgage-backed securities that are covered later in the course.

"""
class lattice:
    #creates a binomial lattice which is then passed on to the subclasses
    def printTree(self):
        for i in range(self.n+1):
            print(self.tree[i], "\n")

    def __init__(self, n):
        self.n = n

        self.tree = np.zeros([self.n+1, self.n+1])


class shortRate(lattice):
    #defines the short rate lattice with nodes at (i, j) where the values of u and d are assumed
    def constructTree(self):
        for i in range(self.n+1): #time i
            for j in range(i+1): #state j
                self.tree[i][j] = self.r0 * (self.u ** j) * (self.d ** (i-j))

    def __init__(self, n, r0, u, d):

        super().__init__(n)
        self.r0 = r0
        self.u = u
        self.d = d

        self.constructTree()

class shortRate2(lattice):
    #defines the short rate lattice with nodes at (i, j) for the BDT model where the short rate depends on drift and volatility parameters a and b
    def constructTree(self):
        for i in range(self.n+1):
            for j in range(i+1):
                self.tree[i][j] = self.a[i] * np.exp(self.b * j)

    def __init__(self, n, a, b):
        #n is the number of periods in the model
        #a is an array of drift parameter values
        #b is the value for the (assumed) fixed volatility parameter
        super().__init__(n)
        self.n = n
        self.a = a
        self.b = b

        self.constructTree()

class elementary(lattice):
    #defines the prices of elementary securities at time i, state j
    def constructTree(self):
        self.tree[0][0] = 1
        for i in range(1,self.n+1):
            for j in range(i+1):
                if j == 0:
                    self.tree[i][j] = self.tree[i-1][j] / (2 * (1 + self.r.tree[i-1][0]))
                elif j == i:
                    self.tree[i][j] = self.tree[i-1][i-1] / (2 * (1 + self.r.tree[i-1][i-1]))
                else:
                    self.tree[i][j] = self.tree[i-1][j] / (2 * (1 + self.r.tree[i-1][j])) + self.tree[i-1][j-1] / (2 * (1 + self.r.tree[i-1][j-1]))


    def __init__(self, n, r):
        super().__init__(n)

        self.n = n
        self.r = r #r is a short rate lattice

        self.constructTree()

class BDT():
    #sets up the Black-Derman-Toy model, including the calibraton function
    def getModelSpotRates(self):
        #returns a numpy array of the spot rates under current model parameters
        spotRates = []
        for i in range(1, self.e.n+1):
            total = np.sum(self.e.tree[i])
            spotRates.append((1 / total) ** (1 / i) - 1)

        return np.array(spotRates)

    def getCost(self):
        #returns sum of squared errors of the model
        costFunc = np.sum((self.getModelSpotRates() - self.spotRates) ** 2)

        return costFunc

    def calibrate(self, iterations):

        def getError(aArray):
            #updates short rate lattice and calculates the new error of the model
            self.e.r = shortRate2(self.e.r.n, aArray, self.e.r.b)
            self.e = elementary(self.e.n, self.e.r)
            return self.getModelSpotRates() - self.spotRates

        seed = self.e.r.a #initial values for drift parameter in the short rate model
        aArray = broyden1(getError, seed, iter=iterations) #calibration of drift parameters
        newError = (getError(aArray) ** 2).sum()

        return aArray, newError

    def __init__(self, e, spotRates):

        self.e = e #elementary security price lattice
        self.spotRates = spotRates #observed spot rates

class bond(lattice):
    #defines binomial model for a bond with no recovery and no default risk
    def constructTree(self):
        for i in range(self.n, -1, -1):
            for j in range(i+1):
                if i == self.n:
                    self.tree[i][j] = self.s0 + self.c * self.s0
                else:
                    self.tree[i][j] = (1 / (1+self.r.tree[i][j])) * ((self.qu * self.tree[i+1][j]) + (self.qd * self.tree[i+1][j+1])) + (self.c * self.s0)

    def __init__(self, n, r, qu, qd, s0, c):
        super().__init__(n)

        self.n = n
        self.qu = qu
        self.qd = qd
        self.r = r
        self.s0 = s0
        self.c = c

        self.constructTree()

class recoveryBond(lattice):
    #defines binomial model for a bond with recovery and default risk
    def constructTree(self):
        for i in range(self.n, -1, -1):
            for j in range(i+1):
                h = self.a * (self.b ** (j - i/2)) #calculate hazard rate at time i state j
                if i == self.n:
                    self.tree[i][j] = self.s0 + self.c * self.s0
                else:
                    #formula only applies for a ZCB
                    self.tree[i][j] = (1 / (1+self.r.tree[i][j])) * ((self.qu * (1 - h) * self.tree[i+1][j]) + (self.qd * (1 - h) * self.tree[i+1][j+1]) + (h * self.recovery * self.s0)) + (self.c * self.s0)

    def __init__(self, n, r, recovery, a, b, qu, qd, s0, c):
        super().__init__(n)

        self.n = n
        self.qu = qu
        self.qd = qd
        self.r = r
        self.recovery = recovery
        self.a = a
        self.b = b
        self.s0 = s0
        self.c = c

        self.constructTree()

class forward(lattice):
    #defines binomial model for a forward on a coupon-bearing bond
    def constructTree(self):
        for i in range(self.n, -1, -1):
            for j in range(i+1):
                if i == self.n:
                    self.tree[i][j] = self.bond.tree[i][j] - self.bond.s0 * self.bond.c
                else:
                    self.tree[i][j] = (1 / (1+self.bond.r.tree[i][j])) * ((self.bond.qu * self.tree[i+1][j]) + (self.bond.qd * self.tree[i+1][j+1]))

    def price(self):
        zcbModel = bond(self.n, self.bond.r, self.bond.qu, self.bond.qd, 1, 0) #price of a ZCB maturing at time n but with face value of $1
        print(self.tree[0][0] / zcbModel.tree[0][0])

    def __init__(self, bond, n):
        super().__init__(n)

        self.n = n
        self.bond = bond

        self.constructTree()

class futures(lattice):
    #defines binomial model for a futures contract on a coupon-bearing bond
    def constructTree(self):
        for i in range(self.n, -1, -1):
            for j in range(i+1):
                if i == self.n:
                    self.tree[i][j] = self.bond.tree[i][j] - self.bond.s0 * self.bond.c
                else:
                    self.tree[i][j] = ((self.bond.qu * self.tree[i+1][j]) + (self.bond.qd * self.tree[i+1][j+1]))

    def price(self):
        print(self.tree[0][0])

    def __init__(self, bond, n):

        super().__init__(n)

        self.n = n
        self.bond = bond

        self.constructTree()

class option(lattice):
    #defines binomial model for an option (can be call/put & american/european) on a bond
    def constructTree(self):
        for i in range(self.n, -1, -1):
            for j in range(i+1):
                if i == self.n:
                    self.tree[i][j] = max(0, self.bond.tree[i][j] - self.strike) if self.type else self.strike - max(0, self.bond.tree[i][j])
                else:
                    self.tree[i][j] = (1 / (1+self.bond.r.tree[i][j])) * ((self.bond.qu * self.tree[i+1][j]) + (self.bond.qd * self.tree[i+1][j+1]))
                    if self.ex: #can early exercise
                        earlyex = self.bond.tree[i][j] - self.strike if self.type else self.strike - self.bond.tree[i][j]
                        self.tree[i][j] = max(earlyex, self.tree[i][j])

    def price(self):
        print(self.tree[0][0])

    def __init__(self, bond, n, strike, american, call):

        super().__init__(n)

        self.n = n
        self.bond = bond
        self.strike = strike
        self.ex = american #true if american, false if european
        self.type = call #true if call, false if put

        self.constructTree()

class swap(lattice):
    #defines binomial model for an interest rate swap
    def constructTree(self):
        for i in range(self.end,-1,-1):
            for j in range(i+1):
                if i == self.end:
                    self.tree[i][j] = (1/(1+self.floating.tree[i][j])) * (self.floating.tree[i][j] - self.fixed)
                elif i >= self.start:
                    self.tree[i][j] = (1/(1+self.floating.tree[i][j])) * ((self.floating.tree[i][j] - self.fixed) + self.qu * self.tree[i+1][j+1] + self.qd * self.tree[i+1][j])
                else:
                    self.tree[i][j] = (1/(1+self.floating.tree[i][j])) * (self.qu * self.tree[i+1][j+1] + self.qd * self.tree[i+1][j])

    def price(self):
        print(self.tree[0][0] * self.notional)

    def __init__(self, start, end, qu, qd, fixed, floating, notional):

        super().__init__(end)

        self.start = start #first payment technically at t = start+1 but count it at t = start
        self.end = end #last payment at t = end+1 but count it at t = end
        self.qu = qu
        self.qd = qd
        self.fixed = fixed #float, fixed rate
        self.floating = floating #shortRate model
        self.notional = notional

        self.constructTree()


class swaption(lattice):
    #defines binomial model for a payer swaption
    def constructTree(self):
        for i in range(self.n,-1,-1):
            for j in range(i+1):
                if i == self.n:
                    self.tree[i][j] = max(self.swap.tree[i][j]-self.strike, 0)
                else:
                    self.tree[i][j] = (1/(1+self.swap.floating.tree[i][j])) * ((self.swap.qu * self.tree[i+1][j+1]) + (self.swap.qd * self.tree[i+1][j]))

    def price(self):
        print(self.tree[0][0] * self.swap.notional)

    def __init__(self, swap, strike, n):

        super().__init__(n)

        self.n = n
        self.swap = swap
        self.strike = strike

        self.constructTree()

class mortgagePassThrough():
    #constructs a pandas dataframe containing the various cashflows of a mortgage pass-through
    def constructDf(self):
        for i in range(1, self.n+1):
            cpr = self.prepay * (0.06 * (i / 30)) if i < 30 else self.prepay * 0.06
            smm = 1 - (1 - cpr) ** (1 / 12)
            monthly = self.p * self.c / (1 - (1 + self.c) ** - (self.n - i + 1)) #monthly payment received
            interestIn = self.p * self.c #interest received
            interestOut = self.p * self.ptrate #interest paid out
            princRepayment = monthly - interestIn #principal received (excludes prepayments)
            prepayment = (self.p - princRepayment) * smm #prepayments received
            totalPayment = princRepayment + prepayment + interestOut #monthly payment to investors
            self.p = self.p - princRepayment - prepayment #update remaining principal

            val = {
                "CPR": cpr,
                "SMM": smm,
                "Monthly Payment": monthly,
                "Interest In": interestIn,
                "Interest Out": interestOut,
                "Principal Repayment": princRepayment,
                "Prepayment": prepayment,
                "Total Payment": totalPayment,
                "Ending Balance": self.p
            }
            self.df = self.df.append(val, ignore_index=True)

    def __init__(self, n, c, ptrate, prepay, p):

        self.n = n #maturity of underlying in months
        self.c = c #coupon rate of underlying (monthly)
        self.ptrate = ptrate #pass-through rate
        self.prepay = prepay / 100 #prepayment multiplier in PSAs
        self.p = p #principal
        self.df = pd.DataFrame(columns = ["CPR", "SMM", "Monthly Payment", "Interest In", "Interest Out", "Principal Repayment", "Prepayment", "Total Payment", "Ending Balance"])

        self.constructDf()

def getPresentValue(series, r):
    #helper function: calculate present value of a series of cashflows (takes a pandas Series and risk-free rate r)
    pv = []
    for i in series.index:
        pv.append(series[i] * (1 / (1 + r)) ** (i+1))

    pvSeries = pd.Series(pv)
    return pvSeries

def getAverageLife(series):
    #helper function: calculate average life of a MBS
    acc = 0
    total = series.sum()
    for i in series.index:
        acc += (i+1) * series[i] / total

    return acc
