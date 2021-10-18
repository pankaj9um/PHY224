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

#Calculating the spring constant based on these parameters
spring_constant = m * c**2
print("The spring constant of the string estimated in the undamped system",
      "exercise, is:", spring_constant, "kg/s^2")

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
ax.figure.savefig("Undamped oscillation. Distance vs. Time"+".png")
plt.show()

print("Our specified parameters produce a curve which fit our data vert well.")

## With the parameters we found, we do a simulation and compare it to our data:
#We use a Euler Forward timestep approach
dt = 1/1000 #our timestep will be dt.
t_sim = np.linspace(0,10,int(1/dt*10)) #array of time points
y_sim = np.zeros(len(t_sim)) #so far empty array of relative distances
v_sim = np.zeros(len(t_sim)) #The same for velocities
v_sim[0] = b*c#cm/s. I adjusted this parameter such that 
                #our simulation fit our data
y_sim[0] = np.mean(Distance)
#Now we perform the simulation, looping forward in time:
for i in range(len(t_sim)-1):
    v_sim[i+1] = v_sim[i]-dt*spring_constant/m*(y_sim[i]-np.mean(Distance)) 
    #*100 for m->cm in sping constant
    y_sim[i+1] = y_sim[i]+dt*v_sim[i]

#Euler-Cromer method:
y_sim_2 = np.zeros(len(t_sim)) #so far empty array of relative distances
v_sim_2 = np.zeros(len(t_sim)) #The same for velocities
v_sim_2[0] = b*c #cm/s. I adjusted this parameter such that 
               #our simulation fit our data
y_sim_2[0] = np.mean(Distance)
#Now we perform the simulation, looping forward in time:
for i in range(len(t_sim)-1):
    y_sim_2[i+1] = y_sim_2[i]+dt*v_sim_2[i]
    v_sim_2[i+1] = v_sim_2[i]-dt*spring_constant/m*(y_sim_2[i+1]-np.mean(
        Distance))
    #*100 for m/s->cm/s
    

#we then plot our simulated curves along with our data.
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(1,1,1)
ax.scatter(Time, Distance, marker='.',lw=0.5,label='measured data')
ax.legend(loc=1)
ax.set_title("Simulated Undamped Oscillation. Distance vs. Time")
ax.plot(t_sim, y_sim,lw=1,label='Simulated curve, forward Euler',c='r')
ax.plot(t_sim, y_sim_2,lw=1,label='Simulated curve, Euler-Cromer',c='g')
ax.set_ylim((19.8, 21.8))
ax.legend(loc=1)
ax.set_xlabel("Time in seconds (s)")
ax.set_ylabel("Distance (cm)")
ax.set_xlim(0,10)
ax.grid()
ax.figure.savefig("Simulated Undamped Oscillation. Distance vs. Time"+".png")  
plt.show()

print("We see that the Euler-Cromer simulation fit our data over time much",
      " better than the Forward Euler simulation, which amplitude grows",
      " significantly in time. (Depending on our time step size ofc.).",
      " The amplitude should however be constant, if not be weakly decreasing,",
      " due to small unavoidable damping in our system of experiment.")



#Phase and Velocity plot
Velocity = b*c*np.cos(c*Time) #Using the time derivative of our curve fit model

#Velocity plot
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(1,1,1)
ax.scatter(Time, Velocity, marker='.',lw=0.5,label='measured data')
ax.legend(loc=1)
ax.set_title("Simulated Undamped Oscillation. Velocity vs. Time")
ax.plot(t_sim, v_sim,lw=1,label='Simulated curve, forward Euler',c='r')
ax.plot(t_sim, v_sim_2,lw=1,label='Simulated curve, Euler-Cromer',c='g')
ax.legend(loc=1)
ax.set_xlabel("Time in seconds (s)")
ax.set_ylabel("Velocity (cm/s)")
ax.set_xlim(0,10)
ax.grid()
ax.figure.savefig("Simulated Undamped Oscillation. Velocity vs. Time"+".png")  
plt.show()

#Phase plot
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(1,1,1)
ax.scatter(Distance,Velocity,marker='.',lw=0.5,
           label='Phase plot, measured velocity vs. distance')
ax.plot(y_sim,v_sim,
        label='Phase plot, Forward Euler simulated velocity vs. distance')
ax.plot(y_sim_2,v_sim_2,
        label='Phase plot, Euler-Cromer simulated velocity vs. distance')
ax.legend(loc=1)
ax.set_xlabel("Distance (cm)")
ax.set_ylabel("Velocity (cm/s)")
ax.set_title("Undamped oscillation. Distance vs. Velocity")
ax.grid()
ax.figure.savefig("Undamped oscillation. Distance vs. Velocity"+".png")
plt.show()

print("As expected, we get elliptical phase plots for our measured and",
      " simulated Distance and Velocity, however, as we will see in the",
      " energy plot, the Forward Euler simulation grows in energy, which",
      "causes the phase space plot of the Forward Euler simulation ellipse to",
      " increase in radius over time. This also corresponds to the increase",
      " in amplitude in the position and velocity plot over time.")



#Energy vs time plot
#Dividing by 100 to get from cm and cm/s to m and m/s. Etot=Ekin+Epot
Etot = m*(Velocity/100)**2/2 + spring_constant*(
    (Distance-np.mean(Distance))/100)**2/2 
Etot_sim = m*(v_sim/100)**2/2 + spring_constant*(
    (y_sim-np.mean(Distance))/100)**2/2 
Etot_sim_2 = m*(v_sim_2/100)**2/2 + spring_constant*(
    (y_sim_2-np.mean(Distance))/100)**2/2 

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(1,1,1)
ax.plot(Time,Etot,lw=0.5,marker='.', label='Measured Data')
ax.plot(t_sim,Etot_sim,lw=0.5,label='Forward Euler')
ax.plot(t_sim,Etot_sim_2,lw=0.5,label='Euler-Cromer')
ax.legend(loc=1)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Energy (J)")
ax.set_title("Undamped oscillation. Total energy vs. time")
ax.grid()
ax.figure.savefig("Undamped oscillation. Total energy vs. time"+".png")
plt.show()

#Zoomed in
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(1,1,1)
ax.plot(Time[0:500],Etot[0:500],lw=0.5,marker='.', label='Measured Data')
ax.plot(t_sim[0:5000],Etot_sim[0:5000],lw=0.5,label='Forward Euler')
ax.plot(t_sim[0:5000],Etot_sim_2[0:5000],lw=0.5,label='Euler-Cromer')
ax.legend(loc=1)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Energy (J)")
ax.set_title("Undamped oscillation. Total energy vs. time, zoomed in")
ax.grid()
ax.figure.savefig(
    "Undamped oscillation. Total energy vs. time, zoomed in"+".png")
plt.show()

print("As we can see from the total Energy plots, the total energy of the",
      "system, is approximately conserved for the Euler-Cromer simulation,",
      " grows in time (unphysical) and is not conserved in the Forward Euler",
      " simulation, and is approximately conserved for our measured data,",
      " when accounting for uncertainties in our measurements of the distance",
      " over time. In theory, total energy should be conserved, but the",
      " energy will oscillate to be in the form of potential and kinetic",
      " energy.")

print("The radius of the elliptical phase plots, correspond to the total",
      " energy of the system. When considering our total energy plots over",
      " time, it makes sense that the Euler Cromer and measured data plots",
      " give approximately stable phase ellipses, though the Forward Euler",
      " phase plot has increasing radius with time, as its energy is",
      " increasing with time. The reason the radius should be constant, is",
      " because the total energy should be constant. The x and y component of",
      " the radius oscillate in their contribution to the radius length,",
      " corresponding to how the total energy is conserved, but oscillates",
      " in being in the form of kinetic and potential energy.")  
    