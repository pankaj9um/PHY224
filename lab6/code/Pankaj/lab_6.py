#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import statslab as utils
import matplotlib.pyplot as plt
import numpy as np


# constants for this exercise

# constant uncertainity in frequency measurement in MHz
frequency_uncertainity = 2

# electron charge in Coulomb
e = 1.602 * 10 ** (-19)

# electron mass ini kg
m = 9.109 * 10 ** (-31)

# permeability
mu = 4*math.pi* (10**(-7)) 

# number of turns of the external coils
n = 320

# radius of the coils in mm
R = 70

# linear model function without y-intercept
def model_function(currrent, slope):
    return currrent * slope

# import the measured data from data files
files = ["../../data/Highest_N_Data.csv", "../../data/Medium_N_Data.csv"]

measured_currents = np.array([])
measured_frequency = np.array([])

# iterate over all the files to collect measured currrents and frequency
for f in files:
    c, f = utils.read_data(f,
                           usecols=(2, 1),
                           skiprows=1)

    measured_currents = np.append(measured_currents, c)
    measured_frequency = np.append(measured_frequency, f)
    
# measure current correction as the current is split between two coils.
measured_currents = measured_currents/2

# frequency errors
frequency_errors = np.ones_like(measured_frequency) * frequency_uncertainity

# compute the measured magnetic field in Tesla
measured_magnetic_field = (4/5)**(3/2) * mu * n * measured_currents / R

# compute the gyromagnetic ratio for each measurement
gamma_measured = 2*math.pi*measured_frequency/measured_magnetic_field

# compute the Lande g factor
g_measured = gamma_measured / (e / (2 * m)) * 10 ** 6

# standard devation in g factor
std_g = np.std(g_measured)
print("Standard deviation of measured Lande g Factor = %.2f" % std_g)


# fit the data to our model function
popt, pstd = utils.fit_data(model_function, 
                          measured_magnetic_field, 
                          measured_frequency, 
                          frequency_errors)

slope = popt[0]

print("Slope of the line (frequency vs magnetic field) = %.4f" % slope)

# compute gamma (gyromagnetic  ratio) using fit data
gamma = 2*math.pi*slope
gamma_var = 2*math.pi* pstd[0]
print("gamma (gyromagnetic  ratio) = %.2f +/- %.2f rad/s/T" % (gamma, gamma_var))

# compute Lange g Factor
g = gamma / (e / (2 * m)) * 10**6
gvar = gamma_var / (e / (2 * m)) * 10**6
print("Lande g Factor= %.2f +/- %.2f" % (g, gvar))

# prepare data for prediction using the curve fit slope value
m_min = np.min(measured_magnetic_field)
m_max = np.max(measured_magnetic_field)
magnetic_field_for_prediction = np.linspace(m_min, m_max, 1000)

# compute the predicted frequencies
predicted_fequency = model_function(magnetic_field_for_prediction, 
                                    slope)

# compute chi2r
chi2r_curve_fit = utils.chi2reduced(measured_frequency, 
                                    model_function(measured_magnetic_field, 
                                                   slope),
                                    frequency_errors, 
                                    1)

print("chi2reduced = %.4f" % chi2r_curve_fit)

# plot measured frequency vs magnetic field, with the curve fit
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

# save the plot
plt.savefig("lab6_freq_vs_magnetic_field.png")

fig  = plt.figure(figsize=(16, 10))
fig.tight_layout()
plt.style.use("seaborn-whitegrid")
plt.errorbar(measured_frequency, g_measured, marker="o",
             yerr=std_g*np.ones_like(g_measured), fmt=" ", 
             label="Measured Lande g Factor")
plt.xlabel("Frequency(MHz)")
plt.ylabel("Lande g Factor")
plt.title("Lande g Factor vs. Resonance Frequency (MHz)")
plt.legend(loc="upper left")

# save the plot
plt.savefig("lab6_g_factor_vs_freq.png")