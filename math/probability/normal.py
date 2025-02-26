#!/usr/bin/env python3
"""
normal distribution
"""

class Normal:
    """
    class normal distribution
    """
    def __init__(self, data=None, mean=0., stddev=1.):
        """
        class contructor
        """
        if data is None:
            self.mean = float(mean)
            self.stddev = float(stddev)
            if self.stddev <= 0:
                raise ValueError("stddev must be a positive value")
        else:
            if isinstance(data, list):
                if len(data) <= 2:
                    raise ValueError("data must contain multiple values")
                self.mean = sum(data) / len(data)
                variance = 0
                for x in data:
                    variance += (x - self.mean)**2
                variance = variance / len(data)
                self.stddev = variance**0.5
            else:
                raise TypeError("data must be a list")

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value
        """
        sub_mean = x - self.mean
        return sub_mean / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score
        """
        pass
