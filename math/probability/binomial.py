#!/usr/bin/env python3
"""
Binomial Distribution
"""

class Binomial:
    """
    class binomial distribution
    """
    def __init__(self, data=None, n=1, p=0.5):
        """
        class contructor
        """
        if data is None:
            if n < 0:
                raise ValueError("n must be a positive value")
            if p < 0 and p > 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = n
            self.p = p
        else:
            if isinstance(data, list):
                if len(data) <=2:
                    raise ValueError("data must contain multiple values")
                mean = sum(data) / len(data)
                summation = 0
                for x in data:
                    summation += ((x - mean) ** 2)
                variance = (summation / len(data))
                q = variance / mean
                p = (1 - q)
                n = round(mean / p)
                p = float(mean / n)
                self.n = n
                self.p = p
            else:
                raise TypeError("data must be a list")

    def pmf(self, k):
        """
        calculates the value of the PMF for given number of "successes"
        """
        if isinstance(k, int):
            if k < 0:
                return 0
        else:
            k = int(k)
        q = (1 - self.p)
        n_factorial = 1
        for i in range(self.n):
            n_factorial *= (i +1)
        k_factorial = 1
        for i in range(k):
            k_factorial *= (i + 1)
        nk_factorial = 1
        for i in range(self.n - k):
            nk_factorial *= (i+ 1)
        binomial_c = n_factorial / (k_factorial * nk_factorial)
        pmf = binomial_c * (self.p ** k) * (q ** (self.n - k))
        return pmf

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given of successes
        """
        if isinstance(k, int):
            if k < 0:
                return 0
        else:
            k = int(k)
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf

