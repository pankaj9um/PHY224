#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Pankaj Patil
"""

import math
import statslab as utils
import matplotlib.pyplot as plt
import numpy as np


voltage_uncertainity = .1

def model_function(currrent, v_infty, resistance):
    return  v_infty - resistance * currrent

# import the first setup for Voltage data
setup_1_file = "../../data/setup1-5V.csv"
measured_voltage = utils.read_data(setup_1_file,
                                                usecols=(2),
                                                skiprows=1)

# import the first setup for Current data
setup_2_file = "../../data/setup2-5V.csv"
measured_current = utils.read_data(setup_2_file,
                                                usecols=(1),
                                                skiprows=1)

title = "Battery 6V"

plt.figure(figsize=(16, 16))
plt.style.use("seaborn-whitegrid")

plt.plot(measured_current, measured_voltage, label="Curve Fit")

voltage_errors = np.ones_like(measured_voltage) * voltage_uncertainity


popt, pstd = utils.fit_data(model_function, 
                      measured_current, 
                      measured_voltage, 
                      voltage_errors)

print("V_infty = %.2f" % popt[0])
print("R_internal = %.8f ohm" % (popt[1] * np.power(10, 6)))

currrent_data = np.linspace(0, measured_current[-1], 100)
predicted_voltage = model_function(currrent_data, 
                                            popt[0],
                                            popt[1])

# plot the fitted curve
plt.plot(currrent_data, predicted_voltage, label="Curve Fit")
    
plt.errorbar(measured_current,
                    measured_voltage, 
                    yerr=voltage_errors, 
                    marker="o",
                    label="Measured Voltage",
                    capsize=2,
                    ls="")

plt.xlabel("Current ($mA$)")
plt.ylabel("Voltage ($V$)")
plt.legend(loc="upper right")

# calculate the chi2reduced for curve fitted model
chi2r_curve_fit = utils.chi2reduced(measured_voltage, 
                              model_function(measured_current, 
                                                         popt[0],
                                                         popt[1]), 
                              voltage_errors, 
                              2)

print("chi2reduced = %.8f" % chi2r_curve_fit)
