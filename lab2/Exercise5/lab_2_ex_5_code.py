import numpy as np
import matplotlib.pyplot as plt
import math

# import our utility method library from statslab.py
import statslab as utils

def count_uncertainty(rate):
    return np.sqrt(rate)/20

def count_uncertainty_logarithmic(rate):
    err = count_uncertainty(rate)
    return err/rate

    
#  model function
def linear_model_function(x, a, b):
    return a*x + b 

def non_linear_model_function(x, a, b):
    return b * np.power(math.e, x * a)

filename = "Fiesta_30092021.txt"
_, measured_count = utils.read_data(filename,
                                                usecols=(0,1),
                                                skiprows=2,
                                                delimiter=None)

filename = "Fiesta_Background_30092021.txt"
_, bg_measured_count = utils.read_data(filename,
                                                usecols=(0,1),
                                                skiprows=2,
                                                delimiter=None)

mean_background_count = np.mean(bg_measured_count)

measured_count_corrected =  (measured_count - mean_background_count)

# corrected rate
measured_count_corrected_rate = measured_count_corrected / 20

plt.hist(measured_count, density=True)