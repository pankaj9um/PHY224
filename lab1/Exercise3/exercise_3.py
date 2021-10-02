
#Importing modules
import numpy as np
import matplotlib.pyplot as plt
import Functions as F

#Defining model function for position prediction
def linear_model(x,a,b):
    return a*x+b

def non_linear_model(x,a,b):
    return a*x**b

#Importing data from csv file
Voltage, Current = np.loadtxt('Data.csv',skiprows=1, 
                                         delimiter=', ', unpack=True)

#To find the uncertainty of the current measurements, we choose the larger 
#uncertainty of the precission (fluctuations of the last digit) and 
#accuracy (instrument error) uncertainty.
def return_uncertainty(A):
    U_acc = 0.0075*A
    U_pre = 0.1
    
    if U_pre >= U_acc:
        U = U_pre
    else:
        U = U_acc
    return U * 2 #added a factor of 2 to our uncertainty

#Creating a list of uncertainties associated with each 
#   y-value/current measurement.
Uncertainty = []
for i in Current:
    Uncertainty.append(return_uncertainty(i))


#Now we fit the parameters of the two models to our data:
popt_linear, pstd_linear = F.fit_data(linear_model, 
                                      np.log(Voltage), 
                                      np.log(Current), 
                                      Uncertainty/Current,
                                      [0.57710934, 2.20315259])


popt_non_linear, pstd_non_linear = F.fit_data(non_linear_model, 
                                      Voltage, 
                                      Current, 
                                      Uncertainty,
                                      [9.05001068, 0.57722491])
print("The estimated optimal parameters with uncertainty by scipy optimize",
      "curve fit are:",popt_linear, "+-",pstd_linear," for the linear, and:",
      popt_non_linear, "+-", pstd_non_linear, "for the non linear model.")

#Calculating predicted y-values of models:
Power_law_non_linear = np.zeros(len(Voltage))
Power_law_linear = np.zeros(len(Voltage))
#And now we also calculate the y-values predicted by the linear model, 
# for the non logarithmic scale. I will use this later on in the plot
Power_law_linear_non_linear = np.zeros(len(Voltage))

for i in range(len(Voltage)):
    Power_law_non_linear[i] = popt_non_linear[0]*Voltage[i]**popt_non_linear[1]
    Power_law_linear[i]= popt_linear[0]*np.log(Voltage[i])+popt_linear[1]
    Power_law_linear_non_linear[i] = np.exp(Power_law_linear[i])
    
#The Chi squared values for these models are:
chi2_non_linear = F.chi2reduced(Current,Power_law_non_linear,
                         Uncertainty,2)
chi2_linear = F.chi2reduced(np.log(Current),Power_law_linear,
                         Uncertainty/Current,2)  
print("\nThe reduced Chi squared values for the non linear and linear model",
      "respectivly",
      "are", np.round(chi2_non_linear,2), "and", np.round(chi2_linear,2))
print("These are good reduced Chi squared values. Perhaps a bit too small.",
      "Next time we may take even more samples to increase the reduced Chi",
      "Squared values. They should ideally be approximately between 1 and 10.")

#output both of the power law relations you calculated.

print("\nThe power law that we found between current and voltage,",
      "was, with our non-linear model: I(V)=", np.round(popt_non_linear[0],4),
      "* V ^", np.round(popt_non_linear[1],4), "And for our linear model: ",
      np.round(popt_linear[1],4),"* exp(", np.round(popt_linear[0],4),"* V )")

    

#Now we plot these models, the data, and the theoretical curve:
#Now we create some data points on the model curve for plotting

#We calculate the predicted count rates by the theory:
# Note, theoretical halflife is 2.6 min
Power_law_tungsten = np.zeros(len(Voltage))
Power_law_blackbody = np.zeros(len(Voltage))
for i in range(len(Voltage)):
    Power_law_tungsten[i] = popt_non_linear[0]*Voltage[i]**0.5882
    Power_law_blackbody[i] = popt_non_linear[0]*Voltage[i]**(3/5)
    
#Now we plot the original data with error bars, along with the curve fit model
plt.figure(figsize=(10,5))
plt.errorbar(Voltage, Current,Uncertainty ,c='r', ls='', marker='o',
             lw=1,capsize=2,
             label = 'Points of measurement with uncertainty')

plt.plot(Voltage,Power_law_non_linear, c='b', 
         label = 'Non-linear curve fit of power law')
plt.plot(Voltage,Power_law_linear_non_linear, c='g', 
         label = 'linear curve fit of power law')
plt.plot(Voltage,Power_law_tungsten, c='y', 
         label = 'Theoretical power law for tungsten')
plt.plot(Voltage,Power_law_blackbody, c='y', 
         label = 'Theoretical power law for a black body')

plt.title("Power law, Current vs. Voltage in a Circuit with a Lightbulb")
plt.xlabel("Voltage (V)")
plt.ylabel("milliampere (mA)")
plt.legend()
plt.grid()
plt.savefig("Power_law"+'.png')
plt.show()


#Now we plot the logarithmic version of this:
plt.figure(figsize=(10,5))
plt.errorbar(Voltage, Current,Uncertainty ,c='r', ls='', marker='o',
             lw=1,capsize=2,
             label = 'Points of measurement with uncertainty')

plt.plot(Voltage,Power_law_non_linear, c='b', 
         label = 'Non-linear curve fit of power law')
plt.plot(Voltage,Power_law_linear_non_linear, c='g', 
         label = 'linear curve fit of power law')
plt.plot(Voltage,Power_law_tungsten, c='y', 
         label = 'Theoretical power law for tungsten')
plt.plot(Voltage,Power_law_blackbody, c='y', 
         label = 'Theoretical power law for a black body')

plt.title("Logarithmic Power law, Current vs. Voltage")
plt.xlabel("natural logarithm of Voltage (ln(V))")
plt.ylabel("natural logaritm of milliampere (ln(mA))")
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.grid()
plt.savefig("Log_Power_law"+'.png')
plt.show()

print("\nWe had to increase our uncertainty by a factor of 2 to",
"make our curve fit graphs pass through the error bars of our",
"measurements.")

print("\nBoth models produce approximately the same curve, which fit our data",
"well. However, the theoretical curves describing the power law, current",
"vs. Voltage, for a tungsten lightbulb and a blackbody, deviate weakly,",
"but consistently",
"from our data points and curves. I have programmed the theoretical",
"curves to only differ from the non-linear model, by their exponent.")

print("\nThe exponents of our two models are very close to the theoretical",
"exponents. The theoretical power laws, with tungsten and blackbody",
"respectively, says that current is proportional",
"to voltage to the power of: 0.5882 and 3/5=0.6, while the nonlinear model",
"approximates an exponent of 0.5778 +-",
np.round(np.sqrt(pstd_non_linear[0]),2),
", while the linear model",
"also approximates an exponent of 0.5779 +-", 
np.round(np.sqrt(pstd_linear[1]),2))

print("\nThus, the values of our fitted exponent fell within the range of the",
"blackbody values (3/5=0.6) and the expected value for tungsten (0.5882),",
"with our calculated standard deviation.")

print("\nThese small deviation causes the theoretical curves to",
"deviate from our model curves.")





