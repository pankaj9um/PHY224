#!/usr/bin/env python3
# @author: Pankaj
# -*- coding: utf-8 -*-

import statslab as utils
import matplotlib.pyplot as plt
import numpy as np

# constants
mu_0 = 4*np.pi*10**(-7)
n = 75 # number of turn of the coil.
R = 0.15 # 15 cm radius of the coil.

# characteristic of coil dimension
k = 1/np.sqrt(2)*(4/5)**(3/2)*mu_0*n/R

# computation of B_e
# work with constant voltage data
r, measured_currents = utils.read_data("../../data/Changing_current.csv",
                usecols=(0, 1),
                skiprows=2)

r = r * 10 ** (-2) # to meters
r_1 = 1/r
B_c = (4/5)**(3/2)*mu_0*n/R*measured_currents

V_constant = 125.0 # Volts

# linear fitting equation
def model_function_Be(x, a, b):
    return  a*x + b

B_c_errors = np.ones_like(measured_currents) * (4/5)**(3/2)*mu_0*n/R * 0.1

popt, pstd = utils.fit_data(model_function_Be, 
                            r_1, 
                            B_c, 
                            B_c_errors)

# get the y intercept
B_e = -popt[1]
print("External Magnetic Field B_e = %.5f +/- %.5f Tesla" % (B_e, pstd[1]))


# plot the predicted and measured data
fig = plt.figure(figsize=(16,10))
fig.tight_layout()

xdata = np.linspace(np.min(r_1), 
                    np.max(r_1), 1000)
ydata = model_function_Be(xdata, popt[0], popt[1])
plt.plot(xdata, ydata, label="Theoretical $B_c$")
plt.xlabel("$1/r$")
plt.ylabel("$B_c$")
plt.title("Plot of $B_c$ vs $1/r$")

plt.errorbar(r_1,
             B_c, 
             yerr=B_c_errors, 
             marker="o",
             label="Measured $B_c$",
             capsize=2,
             ls="")

plt.legend()
plt.savefig("Coil B vs r_1.png", bbox_inches='tight')


# work with constant current data
r, measured_voltages = utils.read_data("../../data/Changing_voltage.csv",
                                       usecols=(0, 1),
                                       skiprows=2)

r = r * 10 ** (-2) # to meters
r_1 = 1/r
r_1_errors = np.ones_like(r_1) * 0.005 / (r ** 2)

I_constant = 0.964 # Ampere
I_0 = B_e / k

# linear fitting equation
def model_function(x, a):
    return  a*x

popt, pstd = utils.fit_data(model_function, 
                            1 / np.sqrt(measured_voltages),
                            r_1, 
                            r_1_errors)

slope = popt[0]

print("Slope of the line = %.2f" % slope)

# plot the predicted and measured data
fig = plt.figure(figsize=(16,10))
fig.tight_layout()

xdata = np.linspace(np.min(1 / np.sqrt(measured_voltages)), 
                    np.max(1 / np.sqrt(measured_voltages)), 1000)
ydata = model_function(xdata, slope)
plt.plot(xdata, ydata, label="Theoretical $1/r$")
plt.xlabel("$1/\sqrt{V}$")
plt.ylabel("$1/r$")
plt.title("Plot of $1/r$ vs $1/\sqrt{V}$")

plt.errorbar(1 / np.sqrt(measured_voltages),
             r_1, 
             yerr=r_1_errors, 
             marker="o",
             label="Measured $1/r$",
             capsize=2,
             ls="")

plt.legend()
plt.savefig("Charge To Mass Ratio.png", bbox_inches='tight')


charge_to_mass_ratio = (slope / (k * (I_constant - I_0) ) ) ** 2
charge_to_mass_ratio_error = 2 * slope * pstd[0]/ \
    (k * (I_constant - I_0) ) ** 2
print("Charge to Mass Ratio for Electron = -%.2e +/- %.2e C/kg"\
      % (charge_to_mass_ratio, charge_to_mass_ratio_error))

