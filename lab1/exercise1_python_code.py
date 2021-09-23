#Importing modules
import numpy as np
import scipy.optimize as sc
from matplotlib import pyplot as plt

#Defining model function for curve fit
def model_function (x , a , b ):
    return a * x + b

#Importing data from csv file
Resistor = 100000
data = np.loadtxt(str(Resistor)+'.csv',skiprows=1, delimiter=',')

xdata = data[:,0] #the voltage, in our first column, will be our x values
ydata = data[:,1] #The current will be our y data.

#A function which inputs the measured current and returns the 
#   measurements associated uncertenty.
def return_uncertenty(A):
    if A>100: #above 100mA
        return 1 #Uncertenty of current
    elif  A<=100 and A>10:
        return 0.1
    else:
        return 0.01

#Defining a list of an initial guess for a and b
initial_guess = [1/(Resistor/1000),0]

#Creating a list of uncertainties associated with each 
#   y-value/current measurement.
sigmadata = []
for i in ydata:
    sigmadata.append(return_uncertenty(i))

#Now we use the scipy curve_fit function to estimate the values of a and b 
#   which whill minimize the euclidian distance between our data points, and 
#   the model curve.
p_opt , p_cov = sc.curve_fit (model_function , xdata , ydata, 
                              p0 = initial_guess, sigma = sigmadata , 
                              absolute_sigma = True )

#The uncertainties of a and b, are the square root of the diagonals of the 
#   p_cov matrix.
p_std = np.sqrt( np.diag ( p_cov ))

print("The optimal values for our curve fit, for a and b, are:",np.round(p_opt,3))
print("Their associated uncertainties  are:", np.round(p_std,4))
print("This curve fit implies an estimated resistance of the resistor of:", 1/np.round(p_opt[0],3), "kilo Ohm.")

#Creating a function to calculate the Chi square value of this model and data
Chi_squared = 0
for i in range(len(ydata)):
    Chi_squared += (ydata[i]-model_function(xdata[i],p_opt[0],p_opt[1]))**2/sigmadata[i]**2

Chi_squared = Chi_squared/(len(ydata)-2)
print("The value of Chi squared for our model and data is:", 
      np.round(Chi_squared,5), "Which is very low. This is not supprising as",
      "our model seem to fit the data points really well, see plot below.")

#Now we create some data points on the model curve for plotting
xvalues_for_plot = np.linspace(xdata[0],xdata[-1],1000)
yvalues_for_plot = []
for i in xvalues_for_plot:
    yvalues_for_plot.append(model_function(i,p_opt[0],p_opt[1]))

#Now we plot the original data with error bars, along with the curve fit model
plt.figure(figsize=(10,5))
plt.errorbar(xdata,ydata,sigmadata,c='r', ls='', marker='o',lw=1,capsize=2,
             label = 'Points of measurement with current uncertainty')
plt.plot(xvalues_for_plot,yvalues_for_plot, c='b', label = 'Linear curve fit')
plt.title('Current versus voltage, plotted for the '+str(Resistor)+' Ohm resistor')
plt.xlabel('Voltage (V)')
plt.ylabel('Milliampere (mA)')
plt.legend()
plt.grid()
plt.show()