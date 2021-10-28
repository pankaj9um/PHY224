"""
@author: Fredrik Dahl Braten
"""

#Importing modules
import numpy as np
import matplotlib.pyplot as plt
import Functions as F


#we find our estimates of the ammeters resistance:
R_A_estimates_PS = []

#Looping through the voltages, V_inf, that we used.
for i in [6,10,15,20]:
    #Importing data
    Resistance1, Voltage1, Current1 = F.read_data('setup1-dc-'+str(i)+'.csv',
                                                  ',',1)
    Resistance2, Voltage2, Current2 = F.read_data('setup2-dc-'+str(i)+'.csv',
                                                  ',',1)
    #Now we change units to SI units:
    Resistance1, Resistance2, Current2, Current1 = (Resistance1*1000, 
                            Resistance2*1000, Current2/1000, Current1/1000)
    a,b = 0,5
    Resistance1, Voltage1, Current1, Resistance2, Voltage2, Current2 = (
        Resistance1[a:b], Voltage1[a:b], Current1[a:b], Resistance2[a:b], 
        Voltage2[a:b], Current2[a:b])
    #Now we calculate the resistance corresponding to each resistor measurement
    for j in range(len(Resistance1)):
        R_A_estimates_PS.append((Voltage1[j]-Voltage2[j])/Current1[j])
        
        print("The current through the voltmeter is:", Current2[j]-Current1[j], 
              "Ampere when the total current throught the circuit is",
          Current2[j], "Ampere, the terminal voltage is:",i,"volts, and the",
          "resistor in the circuit is", Resistance2[j], "Ohm.\n")
#Our estimate of the resistance is the average of all these        
R_A_Estimate_PS = np.mean(R_A_estimates_PS)

#Now we calculate the standard deviation of the resistances. 
#This will be the uncertainties of our estimate
R_A_sd_PS = np.std(R_A_estimates_PS)/np.sqrt(len(R_A_estimates_PS))
   
print("Our estimate of the resistance of the ammeter is:",R_A_Estimate_PS,"+-",
      R_A_sd_PS)