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

# model function for damped displacement
def displacement_damped(time, gamma):
    return y_0 + amplitude * np.power(math.e, -gamma*time) * np.sin(omega*time)

# model function for decay in dispacement
def displacement_decay(time, gamma):
    return y_0 + amplitude * np.power(math.e, -gamma*time)

# load the data file
damped_filename = "../data/damped_point_data.txt"
measured_time, measured_distance = utils.read_data(damped_filename,
                                                usecols=(0,1),
                                                skiprows=2,
                                                delimiter=None)

# offset the  time values to start at t=0
measured_time = measured_time - measured_time[0]

# the initial position is mean  of  the measured  distances
y_0 = np.mean(measured_distance)

# model parameter values 
amplitude = 1.64
gamma = 0.0085
omega = 2*math.pi / 0.723

# compute the spring constant
spring_constant = damped_mass * omega ** 2

print("Amplitude of Oscillations = %.4f cm" 
      % amplitude)
print("Initial Position = %.2f cm"  % y_0)
print("Damping Coefficient = %.4f"  % gamma)
print("Frequency of Oscillations = %.4f rad/s"  % omega)
print("Estimated Spring Constant (Damped Oscillations) = %.4f g/s^2" 
      % spring_constant)

def plot_measured_data():
    # compute the predicted displacement  values using our model function
    predicted_displacement = displacement_damped(measured_time, gamma)
    
    # create  a figure  for  out subplots
    plt.figure(figsize=(16, 16))
    plt.style.use("seaborn-whitegrid")
    
    # plot the measured data for Distance vs Time
    plt.subplot(2, 1, 1)
    plt.scatter(measured_time, measured_distance, 
                 marker='.',lw=0.5,
                 label="Measured Distance Values (cm)")
    
    # plot the envelop using decay curve with gamma as decay coefficient
    plt.plot(measured_time, displacement_decay(measured_time, gamma), 
             label="Decay Curve", color="r")
    
    plt.title("Damped Oscillations: Distance Vs Time")
    axes = plt.gca()
    axes.set_ylim(18, 23)
    
    plt.ylabel("Distance (cm)")
    plt.legend(loc="upper right")
    
    # plot the measured data for Distance vs Time
    plt.subplot(2, 1, 2)
    plt.scatter(measured_time, measured_distance, 
                 marker='.',lw=0.5,
                 label="Measured Distance Values (cm)")
    
    plt.plot(measured_time, predicted_displacement, 
             label="Curve Fit on Measured Values", color="r")
    
    plt.xlabel("Time (s)")
    plt.ylabel("Distance (cm)")
    plt.legend(loc="upper right")   
    
    axes = plt.gca()
    axes.set_xlim(0, 10)
    axes.set_ylim(18, 23)
    
    # save the plot
    plt.savefig("lab3_damped_distance_vs_time.png")

    
def plot_simulated_data():
    # time spte of 0.001 seconds
    dt = 0.001 
    time = np.linspace(0, 10, int(1/dt*10))
    y = np.zeros_like(time)
    v = np.zeros_like(time)
    energy = np.zeros_like(time)
    
    # initialize the first elements of our velocity and distance arrays
    v[0] = omega * amplitude
    y[0] = y_0 
    
    # use Euler-Cromer method for simulation 
    for i in range(len(time)-1):
        y[i+1] = y[i] + dt * v[i]
        v[i+1] = v[i] - dt * (omega ** 2) *  (y[i+1]-y_0) - gamma * v[i] * dt
        energy[i] = 0.5 * damped_mass * (v[i] ** 2) + \
            0.5 * spring_constant * (y[i]-y_0) ** 2
    
    # create a figure for our plots
    plt.figure(figsize=(16, 16))
    plt.style.use("seaborn-whitegrid")
    
    # plot the simulated data for Distance vs Time
    plt.scatter(measured_time, measured_distance, 
                 marker='.',lw=0.5,
                 label="Measured Distance Values (cm)")
    
    # plot distance vs time
    plt.plot(time, y, 
             label="Simulated Distance Values (cm)", color="r")
    
    plt.xlabel("Time (s)")
    plt.ylabel("Distance (cm)")
    plt.legend(loc="upper right")   
    plt.title("Damped Oscillations: Distance Vs Time (Simulated)")

    axes = plt.gca()
    axes.set_xlim(0, 10)
    axes.set_ylim(18, 22.5)
    
    # save the plot
    plt.savefig("lab3_damped_sim_distance_vs_time.png")

    # create  a figure for our plots
    plt.figure(figsize=(16, 16))
    plt.style.use("seaborn-whitegrid")
        
    # plot velocity vs time
    plt.plot(time, v, 
             label="Simulated Velocity Values (cm/s)")
    
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (cm/s)")
    plt.legend(loc="upper right")   
    plt.title("Damped Oscillations: Velocity Vs Time (Simulated)")
    
    # save the plot
    plt.savefig("lab3_damped_sim_velocity_vs_time.png")

    # create a figure for our plots
    plt.figure(figsize=(16, 16))
    plt.style.use("seaborn-whitegrid")
        
    # plot energy vs time
    plt.plot(time[:-1], energy[:-1] / np.power(10, 7), 
             label="Simulated Evergy Values (Joules)")
    
    plt.xlabel("Time (s)")
    plt.ylabel("Energy (Joules)")
    plt.legend(loc="upper right")   
    plt.title("Damped Oscillations: Energy Vs Time (Simulated)")
    
    # save the plot
    plt.savefig("lab3_damped_sim_energy_vs_time.png")


# plot the measured data
plot_measured_data()

# plot simulated  data
plot_simulated_data()