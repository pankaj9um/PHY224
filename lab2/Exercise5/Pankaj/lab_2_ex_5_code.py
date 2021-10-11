#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Pankaj Patil
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# import our utility method library from statslab.py
import statslab as utils

# load the radioactive count from a Fiesta plate.
measured_count = utils.read_data("Fiesta_30092021.txt",
                                                usecols=(1),
                                                skiprows=2,
                                                delimiter=None)

# load the background count data
bg_measured_count = utils.read_data("Fiesta_Background_30092021.txt",
                                       usecols=(1),
                                       skiprows=2,
                                       delimiter=None)
                
# compute the mean background count                                
mean_background_count = np.mean(bg_measured_count)

# correct the measured count by reducing from it mean background count
measured_count_corrected =  (measured_count - mean_background_count)

# function to analyze given counts data
def analyze_data(data_type, counts):
    """ 
    Function to analyze given counts data, and plot histogram, along
    with Poisson Mass Function and Gaussian Distribution
    """
    
    # create new figure for this analysis
    plt.figure(figsize=(16, 10))
    plt.style.use("seaborn-whitegrid")

    # plot the histogram for the measured (corrected) counts
    utils.plot_histogram(counts)
    
    # the most appropriate value for mu = average value of measured (corrected) 
    # count data
    mu = np.average(counts)
    print("%s mu ="%data_type, mu)
    
    # standard daviation for Gaussian Distribution
    std = np.sqrt(mu)
    print("%s std = %.2f"% (data_type, std))

    # get the x range for our  data as input to distrribution functions
    x = np.arange(np.min(counts), np.max(counts))
             
    # plot the Possion Probability Mass Function
    pmf_data = stats.poisson.pmf(x.astype(int), mu)
    plt.plot(x, pmf_data, ms=1, label="Possion Probability Mass \nFunction")
             
    # plot the Gaussian Distrribution
    plt.plot(x, 
             stats.norm.pdf(x, loc=mu, scale=std), 
             label="Gaussian Distribution Function")
    
    # add legend
    plt.xlabel("Radioactive Count for %s" % data_type)
    plt.ylabel("Probability Density")
    plt.title("%s Random Data Analysis" % data_type)
    plt.legend()
    
    # save the plot
    plt.savefig("%s.png" % data_type)
    
# analyze the Data from Fiesta Plate
analyze_data("Fiesta Plate", measured_count_corrected)

# analyze the Data from Background
analyze_data("Background", bg_measured_count)
