#!/usr/bin/env python3
# @author: Pankaj
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

# constants
mu_0 = 4*np.pi*10**(-7)
n = 130 # number of turn of the coil.
R = 0.16 # 16 cm radius of the coil.
uncertainty_current = 0.1 # 0.1 Ampere
uncertainty_radius = 0.005 # 0.5 cm
V_constant = 125.0 # Volts
I_constant = 0.964 # Ampere

# characteristic of coil dimension
k = (1/np.sqrt(2))*np.power(4/5,3/2)*mu_0*n/R

# reading csv data file
def read_data(filename, skiprows=1, usecols=(0,1), delimiter=","):
    """Load give\n file as csv with given parameters, 
    returns the unpacked values"""
    return np.loadtxt(filename,
                      skiprows=skiprows,
                      usecols=usecols, 
                      delimiter=delimiter,
                      unpack=True)

# computation of B_e
def compute_B_e():
    # work with constant voltage data
    r, measured_currents = read_data("../../data/Changing_current.csv",
                    usecols=(0, 1),
                    skiprows=2)
    
    r = r / 100 # to meters
    r_1 = 1/r # reciprocal of radius
    B_c = np.power(4/5,3/2)*mu_0*n*measured_currents/R # coil magnetic  field
        
    B_c_errors = np.ones_like(measured_currents) \
        * (4/5)**(3/2)*mu_0*n/R * uncertainty_current
    
    slope, intercept, r_value, p_value, std_err = linregress(
        r_1, B_c)
    
    B_e = -intercept
    B_e_error = std_err
    print("External Magnetic Field B_e = %.6f \u00B1 %.6f Tesla" 
          % (B_e, B_e_error))
    
    # plot the predicted and measured data
    fig = plt.figure(figsize=(16,10))
    fig.tight_layout()
    
    xdata = np.linspace(np.min(r_1), 
                        np.max(r_1), 1000)
    ydata = xdata * slope + intercept # y = ax +b
    plt.plot(xdata, ydata, label="Linear Fit $B_c = a 1/\sqrt{r} + b$")
    plt.xlabel("$1/r$")
    plt.ylabel("$B_c$")
    plt.title("Plot of $B_c$ vs $1/r$")
    
    # plot the measured data error bars
    plt.errorbar(r_1,
                 B_c, 
                 yerr=B_c_errors, 
                 marker="o",
                 label="Measured $B_c$",
                 capsize=2,
                 ls="")
    
    plt.legend()
    plt.savefig("Coil B vs r_1.png", bbox_inches='tight')
    
    return B_e

def compute_em(B_e):
    # computation of e/m
    # work with constant current data
    r, measured_voltages = read_data("../../data/Changing_voltage.csv",
                                           usecols=(0, 1),
                                           skiprows=2)
    
    r = r / 100 # to meters
    r_1 = 1/r # reciprocal of radius
    r_1_errors = np.ones_like(r_1) * uncertainty_radius / (r ** 2)
    
    I_0 = B_e / k
    
    sqrt_measured_voltages = np.sqrt(measured_voltages)
    sqrt_measured_voltages_1 = 1 / sqrt_measured_voltages
    
    slope, intercept, r_value, p_value, std_err = linregress(
        sqrt_measured_voltages_1, r_1)
    print("Slope of the line = %.2f" % slope)
    
    # plot the predicted and measured data
    fig = plt.figure(figsize=(16,10))
    fig.tight_layout()
        
    xdata = np.linspace(np.min(sqrt_measured_voltages_1), 
                        np.max(sqrt_measured_voltages_1), 1000)
    ydata = xdata * slope + intercept # y = ax + b
    plt.plot(xdata, ydata, label="Linear Fit $1/r = a 1/\sqrt{V} + b$")
    plt.xlabel("$1/\sqrt{V}$")
    plt.ylabel("$1/r$")
    plt.title("Plot of $1/r$ vs $1/\sqrt{V}$")
    
    plt.errorbar(sqrt_measured_voltages_1,
                 r_1, 
                 yerr=r_1_errors, 
                 marker="o",
                 label="Measured $1/r$",
                 capsize=2,
                 ls="")
    
    plt.legend()
    plt.savefig("Charge To Mass Ratio.png", bbox_inches='tight')
    
    charge_to_mass_ratio = (slope / (k * (I_constant - I_0) ) ) ** 2
    charge_to_mass_ratio_error = 2 * slope * std_err/ \
        (k * (I_constant - I_0) ) ** 2
    print("Charge to Mass Ratio for Electron = -%.2e \u00B1 %.2e C/kg"\
          % (charge_to_mass_ratio, charge_to_mass_ratio_error))
    

B_e = compute_B_e()
compute_em(B_e)
