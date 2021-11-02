#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Pankaj Patil
"""

import math
import statslab as utils
import matplotlib.pyplot as plt
import numpy as np


frequency_uncertainity = 1#10**6 # 1MHz
e = 1.602 * 10 ** (-19)
m = 9.109 * 10 ** (-31)

frequency_multiplier = 1#10**6 # MHz

def model_function(currrent, slope):
    return currrent * slope

# import the first setup for Voltage data
#files = ["../../data/Highest_N_Data.csv", "../../data/Medium_N_Data.csv"]
files = ["../../data/Highest_N_Data.csv"]
measured_currents = np.array([])
measured_frequency = np.array([])

for f in files:
    c, f = utils.read_data(f,
                           usecols=(2, 1),
                           skiprows=1)

    measured_currents = np.append(measured_currents, c)
    measured_frequency = np.append(measured_frequency, f)
    
measured_currents = measured_currents/2
measured_frequency = measured_frequency * frequency_multiplier # conver to SI units
frequency_errors = np.ones_like(measured_frequency) * frequency_uncertainity

mu = 4*math.pi* (10**(-7)) 
n = 320
R = 70
measured_magnetic_field = (4/5)**(3/2) * mu * n * measured_currents / R

popt, pstd = utils.fit_data(model_function, 
                          measured_magnetic_field, 
                          measured_frequency, 
                          frequency_errors)

slope = popt[0]
print("slope = %.4f" % slope)
# print("y_intercept = %.2f" %  popt[1])

c_min = np.min(measured_magnetic_field)
c_max = np.max(measured_magnetic_field)

gamma_measured = 2*math.pi*measured_frequency/measured_magnetic_field
g_measured = gamma_measured / (e / (2 * m)) * 10 ** 6
std_gamma = np.std(g_measured)
print("std of gamma = %.2f", std_gamma)

magnetic_field_for_prediction = np.linspace(c_min, c_max, 1000)

predicted_fequency = model_function(magnetic_field_for_prediction, 
                                            popt[0])

chi2r_curve_fit = utils.chi2reduced(measured_frequency, 
                              model_function(measured_magnetic_field, 
                              popt[0]), #
                              frequency_errors, 
                              1)

plot_data = utils.plot_details("Electron Spin Resonance \
                               (Resonance Frequency vs Magnetic Field)")
plot_data.errorbar_legend("Measured Frequency")
plot_data.fitted_curve_legend("Curve Fit")
plot_data.x_axis_label("Magnetic Field (Tesla)")
plot_data.y_axis_label("Frequency (MHz)")
plot_data.xdata(measured_magnetic_field)
plot_data.ydata(measured_frequency)
plot_data.yerrors(frequency_errors)
plot_data.xdata_for_prediction(magnetic_field_for_prediction)
plot_data.ydata_predicted(predicted_fequency)
plot_data.legend_position("upper left")
plot_data.chi2_reduced(chi2r_curve_fit)

# plot the data
utils.plot(plot_data)

plt.savefig("lab6_freq_vs_magnetic_field.png")
print("chi2reduced = %.4f" % chi2r_curve_fit)

fig  = plt.figure(figsize=(16, 10))
fig.tight_layout()
plt.style.use("seaborn-whitegrid")
#xdata = np.arange(0, len(measured_frequency))
plt.errorbar(measured_frequency, g_measured, marker="o", 
             yerr=std_gamma*np.ones_like(g_measured), fmt=" ")
plt.xlabel("Frequency(MHz)")
plt.ylabel("g Factor")
plt.title("g Factor vs.  Resonance Frequency")
plt.legend(loc="upper left")

gamma = 2*math.pi*slope
print("gamma = %.2f" % gamma)

g = gamma / (e / (2 * m)) * 10**6
print("g = %.4f" % g)
