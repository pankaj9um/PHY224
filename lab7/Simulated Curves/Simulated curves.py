# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 16:47:07 2021

@author: Fredrik
"""

#Importing modules
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")

#Defining constants
R_RC_B = 470*10**3 #Ohm
C_RC_B = 1*10**-6 #Ferad
V0_RC_B = 1

R_RC_W = 500
C_RC_W = 0.022*10**-6 
V0_RC_W = 15

R_LR_W = 500
L_LR_W = 46*10**-3 #Henry
V0_LR_W = 20

#Tau:
Tau_RC_B = R_RC_B*C_RC_B
Tau_RC_W = R_RC_W*C_RC_W
Tau_LR_W = L_LR_W/R_LR_W

#Making array of points for plotting graphs:
t_RC_B = np.linspace(0,0.2*9,1000)
V_RC_B = V0_RC_B*np.exp(-t_RC_B/Tau_RC_B)
t_RC_W = np.linspace(0,10*10**-6*5,1000)
V_RC_W = V0_RC_W*np.exp(-t_RC_W/Tau_RC_W)
t_LR_W = np.linspace(0,200*10**-6*3,1000)
V_LR_W = V0_LR_W*np.exp(-t_LR_W/Tau_LR_W)


#we then plot the corresponding curves of exponential decay or increase
#For the RC circuit with battery:
fig = plt.figure(figsize=(16,10))
ax = fig.add_subplot(1,2,1)
ax.axis('off')
ax.set_title("Observed Exponetial Decay \nFor RC Curcuit with Battery")
ax.grid()

ax = fig.add_subplot(1,2,2)
ax.set_title("Exponential Decay of the Potensial Over the Resistor"+
             "\n For an RC Curcuit with a Battery")
ax.plot(t_RC_B,V_RC_B,c='r', ls='', marker='.',lw=1)
ax.set_xlabel("Time in seconds (s)")
ax.set_ylabel("Voltage (V)")
ax.figure.savefig("Exponential Decay of the Potential Over the Resistor"+
                  "For an RC Curcuit with a Battery"+".png")  

#For the RC cirvuit with wave generator
fig = plt.figure(figsize=(16,10))
ax = fig.add_subplot(1,2,1)
ax.axis('off')
ax.set_title("Observed Exponetial Decay \nFor RC Curcuit with Wave Generator")
ax.grid()

ax = fig.add_subplot(1,2,2)
ax.set_title("Exponential Decay of the Potensial Over the Resistor"+
             "\n For an RC Curcuit with the Wave Generator")
ax.plot(t_RC_W,V_RC_W,c='r', ls='', marker='.',lw=1)
ax.set_xlabel("Time in seconds (s)")
ax.set_ylabel("Voltage (V)")
ax.figure.savefig("Exponential Decay of the Potential Over the Resistor"+
                  "For an RC Curcuit with the Wave Generator"+".png")  


#For the LR circuit with wave generator
fig = plt.figure(figsize=(16,10))
ax = fig.add_subplot(1,2,1)
ax.axis('off')
ax.set_title("Observed Exponetial Increase \nFor LR Curcuit with Wave Generator")
ax.grid()

ax = fig.add_subplot(1,2,2)
ax.set_title("Exponential Increase of the Potential Over the Resistor"+
             "\n For an LR Curcuit with the Wave Generator")
ax.plot(t_LR_W,10-V_LR_W,c='r', ls='', marker='.',lw=1)
ax.set_xlabel("Time in seconds (s)")
ax.set_ylabel("Voltage (V)")
ax.figure.savefig("Exponential Increase of the Potential Over the Resistor"+
                  "For an LR Curcuit with the Wave Generator"+".png") 


plt.show()