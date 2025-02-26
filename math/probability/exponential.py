#!/usr/bin/env python3
"""
exponential distribution
"""

class Exponential:
    """
    class exponential distribution
    """
    pi = 3.1415926536
    e = 2.7182818285
    
    def __init__(self, data=None, lambtha=1.):
        """
        class contructor
        """
        if data is None:
            self.lambtha = float(lambtha)
            if lambtha < 0:
                raise ValueError("lambtha must be a positive value")
        else:
            if isinstance(data, list):
                if len(data) < 2:
                    raise ValueError("data must contain multiple values")
                self.lambtha = 1 / (sum(data) / len(data))
            else:
                raise TypeError("data must be a list")

    def pdf(self, x):
        """
        calculate the value of the PDF for a given time period
        """
        if x < 0:
            return 0
        x = (self.lambtha * (self.e**(-self.lambtha * x)))
        return x

