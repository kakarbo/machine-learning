#!/usr/bin/env python3
"""
normal distribution
"""

class Normal:
    """
    class normal distribution
    """
    pi = 3.1415926536
    e = 2.7182818285
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
        sub_mean = z * self.stddev
        return self.mean + sub_mean

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value
        """
        result_1 = -(x - self.mean) ** 2
        result_2 = 2*self.stddev ** 2
        result_3 = result_1 / result_2
        result_4 = self.e ** result_3
        result_5 = (2 * self.pi) ** 0.5
        result_6 = self.stddev * result_5
        result_pdf = (1 / result_6) * result_4
        return result_pdf

