#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Pankaj Patil
"""

import statslab as utils
import matplotlib.pyplot as plt
import numpy as np


frequency_uncertainity = 0.1

def model_function(currrent, slope, y_intercept):
    return currrent * slope + y_intercept

# import the first setup for Voltage data
# files = ["../../data/Highest_N_Data.csv", "../../data/Medium_N_Data.csv"]
files = ["../../data/Highest_N_Data.csv"]
measured_currents = []
measured_frequency = []

for f in files:
    c, f = utils.read_data(f,
                           usecols=(0, 1),
                           skiprows=1)

    measured_currents.extend(c)
    measured_frequency.extend(f)
    
frequency_errors = np.ones_like(measured_frequency) * frequency_uncertainity

popt, pstd = utils.fit_data(model_function, 
                          measured_currents, 
                          measured_frequency, 
                          frequency_errors,
                          guess=(0.030, 0.0))

print("slope = %.8f" %  popt[0])
print("y_intercept = %.2f" %  popt[1])

c_min = np.min(measured_currents)
c_max = np.max(measured_currents)

currrent_data_for_prediction = np.linspace(c_min, c_max, 1000)

predicted_fequency = model_function(currrent_data_for_prediction, 
                                            popt[0],
                                            popt[1])

# chi2r_curve_fit = utils.chi2reduced(measured_frequency, 
#                              model_function(measured_currents, 
#                              popt[0],
#                              popt[1]), #
#                              frequency_errors, 
#                              2)

plot_data = utils.plot_details("Battery 6V (Voltage Vs Current))")
plot_data.errorbar_legend("Measured Frequency")
plot_data.fitted_curve_legend("Curve Fit")
plot_data.x_axis_label("Current mA")
plot_data.y_axis_label("Frequency (MHz)")
plot_data.xdata(measured_currents)
plot_data.ydata(measured_frequency)
plot_data.yerrors(frequency_errors)
plot_data.xdata_for_prediction(currrent_data_for_prediction)
plot_data.ydata_predicted(predicted_fequency)
plot_data.legend_position("upper right")
# plot_data.chi2_reduced(chi2r_curve_fit)

# plot the data
utils.plot(plot_data)

plt.savefig("lab5_voltage_vs_current.png")
# print("chi2reduced = %.8f" % chi2r_curve_fit)
