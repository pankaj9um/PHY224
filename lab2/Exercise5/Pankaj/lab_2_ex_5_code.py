import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as stats

# import our utility method library from statslab.py
import statslab as utils

def count_uncertainty(count):
    return np.sqrt(count)

filename = "Fiesta_30092021.txt"
sample_number, measured_count = utils.read_data(filename,
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

mu = np.mean(measured_count_corrected)

plt.hist(measured_count, density=True)

x = np.arange(stats.poisson.ppf(0.01, mu), 
              stats.poisson.ppf(0.99, mu))
              
plt.plot(x, stats.poisson.pmf(x, mu), 'bo', ms=8, label='poisson pmf')

