#!/usr/bin/env python3

class Poisson:
    """
    represent a poisson distribution
    """
    pi = 3.1415926536
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """contructor
        """
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

