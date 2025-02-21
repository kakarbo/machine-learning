#!/usr/bin/env python3
'''
The probability module is a branch of mathematics that studies uncertainty and measures the likelihood of an event occurring. It is represented by a number between 0 and 1:

* 0 means the event will never occur.
* 1 means the event will always occur.
* An intermediate value indicates the probability of occurrence.
'''

class Poisson:
    """
    represent a poisson distribution
    """
    pi = 3.1415926536
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """contructor
        """
        self.data = data
        if data is None:
            self.lambtha = lambtha
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
        else:
            if isinstance(data, list):
                if len(data) < 2:
                    raise ValueError("data must contain multiple values")
                self.lambtha = sum(data) / len(data)
            else:
                raise TypeError("data must be a list")

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of 'successes'
        """

        if isinstance(k, int):
            if k < 0:
                return 0
        else:
            k = int(k)
        fact = 1
        for i in range(1, k+1):
            fact = fact * i
        print(self.lambtha)
        k = ((self.lambtha**k)*(self.e**self.lambtha))/fact
        return k



    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of 'successes'
        """
        count = 0.0
        if isinstance(k, int):
            if k < 0:
                return 0
        else:
            k = int(k)
        
        for value in self.data:
            if value <= k:
                count += 1
        k = count / len(self.data)
        return k

