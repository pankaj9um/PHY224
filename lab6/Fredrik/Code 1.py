# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 11:37:08 2021

@author: Fredrik
"""
#Importing modules
import numpy as np
import matplotlib.pyplot as plt
import Functions as F

#Importing data
Current_H, Frequency_H, Voltage_H = F.read_data("Highest_N_Data.csv", ", ", 1)
Current_M, Frequency_M, Voltage_M = F.read_data("Medium_N_Data.csv", ", ", 1)

#Now we combine this data into single arrays.
Current,Frequency, Voltage = [],[],[]
"""
Current.extend(Current_M)
Frequency.extend(Frequency_M)
Voltage.extend(Voltage_M)
"""
Current.extend(Current_H)
Frequency.extend(Frequency_H)
Voltage.extend(Voltage_H)

popt, pstd = fit_data(model_func, xdata, ydata, yerrors, guess)

sigmadata = np.zeros(len(Current))+1
#we then plot our simulated curves along with our data.
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(1,1,1)
ax.legend(loc=1)
ax.set_title("Current vs. Frequency")
ax.errorbar(Current,Frequency,sigmadata,c='r', ls='', marker='o',lw=1,capsize=2,
             label = 'Points of measurement with frequency uncertainty')
ax.legend(loc=1)
ax.set_xlabel("Time in seconds (s)")
ax.set_ylabel("Distance (cm)")
ax.grid()
ax.figure.savefig("Simulated Undamped Oscillation. Distance vs. Time"+".png")  
plt.show()