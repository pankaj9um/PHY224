#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Pankaj Patil
"""

import numpy as np
import scipy.optimize as optim
import matplotlib.pyplot as plt


###############################################################################
# Utility Methods Library
#
# This file contains some utility method which are common to our data analysis.
# This library also contains customized plotting methods.
###############################################################################

# use bigger font size for plots
plt.rcParams.update({'font.size': 20})

def chi2(y_measure,y_predict,errors):
    """Calculate the chi squared value given a measurement with errors and 
    prediction"""
    return np.sum( np.power(y_measure - y_predict, 2) / np.power(errors, 2) )

def chi2reduced(y_measure, y_predict, errors, number_of_parameters):
    """Calculate the reduced chi squared value given a measurement with errors
    and prediction, and knowing the number of parameters in the model."""
    return chi2(y_measure, y_predict, errors)/ \
            (y_measure.size - number_of_parameters)

def read_data(filename, skiprows=1, usecols=(0,1), delimiter=","):
    """Load give\n file as csv with given parameters, 
    returns the unpacked values"""
    return np.loadtxt(filename,
                      skiprows=skiprows,
                      usecols=usecols, 
                      delimiter=delimiter,
                      unpack=True)

def fit_data(model_func, xdata, ydata, yerrors, guess=None):
    """Utility function to call curve_fit given x and y data with errors"""
    popt, pcov = optim.curve_fit(model_func, 
                                xdata, 
                                ydata, 
                                absolute_sigma=True, 
                                sigma=yerrors,
                                p0=guess)

    pstd = np.sqrt(np.diag(pcov))
    return popt, pstd

# y = ax+b
def linear_regression(xdata, ydata):
    """Simple linear regression model"""
    x_bar = np.average(xdata)
    y_bar = np.average(ydata)
    a_hat = np.sum( (xdata - x_bar) * (ydata - y_bar) ) / \
            np.sum( np.power((xdata - x_bar), 2) )
    b_hat = y_bar - a_hat  * x_bar
    return a_hat, b_hat

class plot_details:
    """Utility class to store information about plots"""
    def __init__(self, title):
        self.title = title
        self.x_log_scale = False
        self.y_log_scale = False

    def errorbar_legend(self, v):
        self.errorbar_legend = v
    def fitted_curve_legend(self, v):
        self.fitted_curve_legend = v
    def x_axis_label(self, v):
        self.x_axis_label = v
    def y_axis_label(self, v):
        self.y_axis_label = v
    def xdata(self, x):
        self.xdata = x
    def ydata(self, y):
        self.ydata = y
    def yerrors(self, y):
        self.yerrors = y
    def xdata_for_prediction(self, x):
        self.xdata_for_prediction = x
    def ydata_predicted(self, y):
        self.ydata_predicted = y
    def legend_position(self, p):
        self.legend_loc = p
    def chi2_reduced(self, c):
        self.chi2_reduced = c
    def set_x_log_scale(self, c):
        self.x_log_scale = c
    def set_y_log_scale(self, c):
        self.y_log_scale = c
                
def plot(plot_details, new_figure=True, error_plot=True):
    """Utility method to plot errorbar and line chart together, 
    with given arguments"""
    if new_figure:
        plt.figure(figsize=(16, 10))
        plt.style.use("seaborn-whitegrid")
            
    # plot the error bar chart
    if error_plot:
        plt.errorbar(plot_details.xdata,
                    plot_details.ydata, 
                    yerr=plot_details.yerrors, 
                    marker="o",
                    label=plot_details.errorbar_legend,
                    capsize=2,
                    ls="")

    # plot the fitted curve
    plt.plot(plot_details.xdata_for_prediction, 
             plot_details.ydata_predicted,
             label=plot_details.fitted_curve_legend)

    # legend and title
    plt.title(plot_details.title)
    plt.xlabel(plot_details.x_axis_label)
    plt.ylabel(plot_details.y_axis_label)
    
    if plot_details.x_log_scale:
        plt.xscale("log")
        
    if plot_details.y_log_scale:
        plt.yscale("log")
    
    legend_pos = "upper left"
    if hasattr(plot_details, "legend_loc"):
        legend_pos = plot_details.legend_loc
        
    plt.legend(loc=legend_pos)

def plot_histogram(count_data, new_figure=True, 
                   title="", xlabel="", ylabel="", legend_pos='upper left'):
    """Utility method to plot histogram, with  densiity=True"""
    if new_figure:
        plt.figure(figsize=(16, 15))
        plt.style.use("seaborn-whitegrid")
            
    # plot the data
    count, bins, _ = plt.hist(count_data, bins=10, rwidth=0.95, density=True)

    # legend and title
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
        
    # set legend position
    plt.legend(loc=legend_pos)
    return count, bins
