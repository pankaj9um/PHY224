#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Pankaj Patil and Fredrik Dahl Braten
"""

import statslab as utils
import matplotlib.pyplot as plt
import numpy as np


voltage_uncertainity = .1

def model_function(currrent, v_infty, resistance):
    return  v_infty - resistance * currrent

#Looping through the voltages for which the power supply outputted
#We would like to remember the max current for each of these voltages:
max_currents = []
for i in [6,10,15,20]:
    # import the first setup for Voltage data
    setup_1_file = "setup1-dc-"+str(i)+".csv"
    measured_voltages = utils.read_data(setup_1_file,
                                                    usecols=(2),
                                                    skiprows=1)
    
    # import the first setup for Current data
    setup_2_file = "setup2-dc-"+str(i)+".csv"
    measured_currents = utils.read_data(setup_2_file,
                                                    usecols=(1),
                                                    skiprows=1)
    
    voltage_errors = np.ones_like(measured_voltages) * voltage_uncertainity
    
    popt, pstd = utils.fit_data(model_function, 
                              measured_currents, 
                              measured_voltages, 
                              voltage_errors)
    
    print("for voltage of the power supply equal to:", i,"Volts, we have")
    print("V_infty = %.2f V" %  popt[0])
    print("R_internal = %.2f ohm" %  (popt[1] * np.power(10, 6)))
    
    currrent_data_for_prediction = np.linspace(measured_currents[0], 
                                                measured_currents[-1], 1000)
    
    predicted_voltages = model_function(currrent_data_for_prediction, 
                                                popt[0],
                                                popt[1])
    
    chi2r_curve_fit = utils.chi2reduced(measured_voltages, 
                                  model_function(measured_currents, 
                                                             popt[0],
                                                             popt[1]), 
                                  voltage_errors, 
                                  2)
    # finding the maximum current:
    max_currents.append(i/(popt[1] * np.power(10, 6)))
    
    
    plot_data = utils.plot_details("Power Supply 20V (Voltage Vs Current))")
    plot_data.errorbar_legend("Measured Voltage")
    plot_data.fitted_curve_legend("Curve Fit $V = V_{\infty} - RI$")
    plot_data.x_axis_label("Current mA")
    plot_data.y_axis_label("Voltage (V)")
    plot_data.xdata(measured_currents)
    plot_data.ydata(measured_voltages)
    plot_data.yerrors(voltage_errors)
    plot_data.xdata_for_prediction(currrent_data_for_prediction)
    plot_data.ydata_predicted(predicted_voltages)
    plot_data.legend_position("upper right")
    plot_data.chi2_reduced(chi2r_curve_fit)
    
    # plot the data
    utils.plot(plot_data)
    
    plt.savefig("Power supply 20V lab5_voltage_vs_current.png")
    print("chi2reduced = %.8f" % chi2r_curve_fit)
    
    
#Now we can plot the max current relationship as a function of V_inf:
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(1,1,1)
ax.set_title("Estimated Max Current vs. V_inf based on our linear models")
ax.plot([6,10,15,20], max_currents,lw=1,c='r')
ax.set_xlabel("V_inf (Volts)")
ax.set_ylabel("Max Current (Ampere)")
ax.figure.savefig("Estimated Max Current vs. V_inf based on our linear models"+".png")  
plt.show()
