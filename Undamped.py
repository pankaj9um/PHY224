#Importing modules
import numpy as np
import matplotlib.pyplot as plt
import Functions as F


##Undamped Oscillation:
    
#Specifying modelling function
def model(t,a,b,c):
    return a+b*np.sin(c*t)

#Importing data
Time, Distance = F.read_data('undamped_point_data_set2.txt', None,2)

#Defining Constants
Sample_time = 0.01 #seconds
m = 200.0/1000 #Kilograms
m_uncertainty = 0.1/1000 #kilograms

#Specifying parameters for the model function
a = np.mean(Distance)
b = 0.7
c = (2*np.pi)/0.693

#Offsetting time array
Time = np.array([i-0.49 for i in Time])

#Plotting data points and model curve
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(1,1,1)
ax.plot(Time, Distance, marker='.',lw=0.5,label='measured data')
ax.plot(Time, model(Time,a,b,c),lw=1,label='fitted curve')
ax.set_ylim((19.9, 21.75))
ax.legend(loc=1)
ax.set_xlabel("Time in seconds (s)")
ax.set_ylabel("Distance from sensor in centimeters (cm)")
ax.set_title("Undamped oscillation. Distance vs. Time")
ax.grid()
plt.show()

##Damped Oscillation:
    
    
#Specifying modelling function
def model(t,a,b,c,d):
    return a+b*np.exp(-c*t)*np.sin(d*t)

#Importing data
Time, Distance = F.read_data('damped_point_data.txt', None,2)

#Defining Constants
Sample_time = 0.01 #seconds
m = 215.1/1000 #Kilograms
m_uncertainty = 0.1/1000 #kilograms

#Specifying parameters for the model function
a = np.mean(Distance)
b = 0.7
c = 0.2
d = (2*np.pi)/0.693

#Offsetting time array
Time = np.array([i-0.49 for i in Time])[0:100]
Distance = Distance[0:100]

#Plotting data points and model curve
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(1,1,1)
ax.plot(Time, Distance, marker='.',lw=0.5,label='measured data')
ax.plot(Time, model(Time,a,b,c),lw=1,label='fitted curve')
ax.set_ylim((19.9, 21.75))
ax.legend(loc=1)
ax.set_xlabel("Time in seconds (s)")
ax.set_ylabel("Distance from sensor in centimeters (cm)")
ax.set_title("Undamped oscillation. Distance vs. Time")
ax.grid()
plt.show()