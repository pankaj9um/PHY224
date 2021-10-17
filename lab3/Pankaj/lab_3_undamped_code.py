#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Pankaj Patil
"""

import math
import statslab as utils
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


undamped_mass = 200 # in grams
undamped_mass_uncertainty = 0.1 # in grams

damped_mass = 215.1 # in grams
damped_mass_uncertainty = 0.1 # in grams

def displacement_undamped(time):
    return y_0 + amplitude*np.sin(omega*time)


undampled_filename = "../data/undamped_point_data_set2.txt"
measured_time, measured_distance = utils.read_data(undampled_filename,
                                                usecols=(0,1),
                                                skiprows=2,
                                                delimiter=None)

measured_time = measured_time - measured_time[0]

fig  = plt.figure(figsize=(16, 10))
plt.style.use("seaborn-whitegrid")

# plot the actual data
plt.errorbar(measured_time, measured_distance, fmt=" ", 
             marker="o", markersize=1, label="measured data points")

y_0 = np.mean(measured_distance)
amplitude = 0.7
omega = 2*math.pi / 0.70

predicted_displacement = displacement_undamped(measured_time)

plt.plot(measured_time, predicted_displacement, label="curve fit data points")

plt.xlabel("time (s)")
plt.ylabel("distance (cm)")
plt.legend(loc="upper left")
plt.title("Distance Vs Time")

axes = plt.gca()
axes.set_ylim(19.75, 21.75)

spring_constant = undamped_mass * omega ** 2

print("spring constant %.4f g/s^2" % spring_constant)

# animation
y_i = amplitude
v_i = 0

# Create new Figure and an Axes which fills it.
fig = plt.figure(figsize=(7, 7))
plot, = plt.plot([1], [y_i], marker="o", markersize=20)

print(plot)

axes = plt.gca()
axes.set_ylim(19.5, 22)
axes.set_xticks([])

def update(time):
    global y_i, v_i
    interval = 0.01 # seconds
    y_i = y_i + interval * v_i
    v_i = v_i - interval * (omega ** 2) * y_i

    plot.set_ydata([y_0 + y_i])
    

x = np.linspace(0, 10, 1000)
plt.plot(x, np.ones_like(x) * np.max(measured_distance))
plt.plot(x, np.ones_like(x) * y_0)
plt.plot(x, np.ones_like(x) * np.min(measured_distance))

# Construct the animation, using the update function as the animation director.
animation = FuncAnimation(fig, update, interval=0.01)
plt.show()