#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Pankaj Patil
"""

import numpy as np
import matplotlib.pyplot as plt

# import our utility method library from statslab.py
import statslab as utils

# units for the measured quantities
units = {
    "voltage": "V",
    "current": "mA",
    "resistance": "kiloohm"
}

# assume 2% uncertinity due to connection errors, human factors etc.
uncertainty_correction = 0.02

# compute the uncertainty in current values
def current_uncertainty(current):
    """return the uncertainty in current for given values of current"""
    error_accuracy = 0.0075 * current
    error_precision = 0.01
    
    # multimeter unceertainty is maxmimum of the above two
    multimeter_uncertainty = max(error_accuracy, error_precision)
        
    # we additionaly compute uncertainty using 2% uncertainly due to 
    # connections and human factors
    return max(multimeter_uncertainty, uncertainty_correction*current)
    
#  linear model function
def linear_model_function(x, a, b):
    return a*x + b 

# analyse the data file
def analyse_file(filename, title): 
    print(filename)
    
    # read the data from the data file, and collect measured  voltages and
    # currents
    measured_voltages, measured_currents = utils.read_data(filename,
                                                           usecols=(0,1))

    # create error array for the current. use np.vectorize to make the 
    # uncertainty function operate on arrays.
    current_errors = np.vectorize(current_uncertainty)(measured_currents)
    
    # fit the measured data using curve_fit function
    popt, pstd = utils.fit_data(linear_model_function, 
                          measured_voltages, 
                          measured_currents, 
                          current_errors)
    
    # generate data for predicted values using estimated resistance
    # obtained using curve fit model
    voltage_data = np.linspace(0, measured_voltages[-1], 100)
    predicted_currents = linear_model_function(voltage_data, 
                                                popt[0],
                                                popt[1])
    
    # calculate the chi2reduced for curve fitted model
    chi2r_curve_fit = utils.chi2reduced(measured_currents, 
                                  linear_model_function(measured_voltages, 
                                                             popt[0],
                                                             popt[1]), 
                                  current_errors, 
                                  3)
    
    # fill in the plot details for curve fitted 
    # model in our plot_details object
    plot_data = utils.plot_details("Current Vs Voltage (%s)" % title)
    plot_data.errorbar_legend("Measured Current with Uncertainty (%s)" % units["current"])
    plot_data.fitted_curve_legend("Linear Fit for Ohm's Law")
    plot_data.x_axis_label("Voltage (%s)" % units["voltage"])
    plot_data.y_axis_label("Current (%s)" % units["current"])
    plot_data.xdata(measured_voltages)
    plot_data.ydata(measured_currents)
    plot_data.yerrors(current_errors)
    plot_data.xdata_for_prediction(voltage_data)
    plot_data.ydata_predicted(predicted_currents)
    plot_data.legend_position("upper left")
    plot_data.chi2_reduced(chi2r_curve_fit)
    
    # plot the data
    utils.plot(plot_data)
            
    print("\tLinear model slope (a) = %.2f" % popt[0])
    print("\tLinear model y-intercept (b) := %.4f" % popt[1])
    print("\tLinear model slope (a) uncertainty := %.4f" % pstd[0])
    print("\tLinear model y-intercept (b) uncertainty := %.4f" % pstd[1])
    print("\tEstimated Resistance (1/a) = %.4f %s" % (1/popt[0], 
                                                      units["resistance"]))

    print("\tchi2reduced (Curve Fit) := %.3f" % chi2r_curve_fit)
    print("\tUncertainty in resistance  := %.4f %s" % (pstd[0], 
                                                       units["resistance"]))
    
    plt.savefig("lab_1_ex_1_plot_%s.png" % filename[:-4].lower())

# files to analyse
file_titles ={
    "100k.csv": {
        "title": "Resistor: 100 $k\Omega$"
    },
    "Potentiometer.csv": {
        "title": "Potentiometer"
    }
}

for filename, data in file_titles.items():
    analyse_file(filename, data["title"])
    
