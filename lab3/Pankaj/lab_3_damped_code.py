#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Pankaj Patil
"""

import math
import statslab as utils
import matplotlib.pyplot as plt
import numpy as np

damped_mass = 215.1 # in grams
damped_mass_uncertainty = 0.1 # in grams

def displacement_damped(time, gamma):
    return y_0 + amplitude * np.power(math.e, -gamma*time) * np.sin(omega*time)

def displacement_decay(time, gamma):
    return y_0 + amplitude * np.power(math.e, -gamma*time)

damped_filename = "data/damped_point_data.txt"
measured_time, measured_distance = utils.read_data(damped_filename,
                                                usecols=(0,1),
                                                skiprows=2,
                                                delimiter=None)

measured_time = measured_time - measured_time[0]

y_0 = np.mean(measured_distance)
amplitude = 1.7
gamma = 0.009
omega = 2*math.pi / 0.723

fig  = plt.figure(figsize=(16, 10))
plt.style.use("seaborn-whitegrid")

# plot the actual data
plt.subplot(2, 1, 1)
plt.errorbar(measured_time, measured_distance, fmt=" ", 
             marker="o", markersize=1, label="measured data points")
plt.plot(measured_time, displacement_decay(measured_time, gamma), 
         label="curve fit data points", color="r")

plt.xlabel("time (s)")
plt.ylabel("distance (cm)")
plt.legend(loc="upper right")
plt.title("Distance Vs Time")

axes = plt.gca()
# axes.set_ylim(19.0, 21)

predicted_displacement = displacement_damped(measured_time, gamma)

plt.subplot(2, 1, 2)

plt.errorbar(measured_time, measured_distance, fmt=" ", 
             marker="o", markersize=1, label="measured data points")
plt.plot(measured_time, predicted_displacement, 
         label="curve fit data points", color="r")

plt.xlabel("time (s)")
plt.ylabel("distance (cm)")
plt.legend(loc="upper right")   

axes = plt.gca()
axes.set_xlim(0, 10)

spring_constant = damped_mass * omega ** 2

print("spring constant %.4f g/s^2" % spring_constant)