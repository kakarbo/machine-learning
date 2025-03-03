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
