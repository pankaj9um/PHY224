# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 09:47:45 2021

@author: Fredrik
"""
#Importing modules
import numpy as np
import matplotlib.pyplot as plt
import Functions as F


#Defining Constants
Sample_time = 20 #seconds
Experiment_length = 20*60 #Seconds. = 20 min

#Specifying modelling function
def non_linear_model(x,a,b):
    return a*np.exp(b*x)

def linear_model(x,a,b):
    return a*x+b


#Importing data
Sample_number_barium, Number_of_counts_barium = F.read_data('Barium.txt', 
                                                            None,2)
Sample_number_bar_background, Number_of_counts_bar_background = F.read_data(
    'Barium_Background.txt', None,2)

#However, background radiation is not measured at the same time as the 
# radiation from the Barium. Thus I subtract the mean number of counts from
# the background radiation, from the number of counts for each Barium sample.
Number_of_counts_bar_background = np.mean(Number_of_counts_bar_background)

Number_of_counts = Number_of_counts_barium - Number_of_counts_bar_background

#Now we remove the latter part of the data, where the radiation from the
# barium is at the same level as the background radiation. When this happens
# the Number_of_counts value above may be negative.
list_of_negative_values = []
for i in range(len(Number_of_counts)):
    if Number_of_counts[i]<0:
        list_of_negative_values.append(i)
if len(list_of_negative_values)==0:
    list_of_negative_values.append(-1)

#Now that we know at what index this happen, we shorten all the arrays we will 
# use later on
Number_of_counts = Number_of_counts[0:list_of_negative_values[0]]
Number_of_counts_barium = Number_of_counts_barium[0:list_of_negative_values[0]]
Sample_number_barium = Sample_number_barium[0:list_of_negative_values[0]]

#Standard deviation for each point, for derivation, see exercise 2 document.
Count_rate_uncertainty = np.sqrt(
    Number_of_counts_barium + Number_of_counts_bar_background)

Count_rate = Number_of_counts/Sample_time

#Now we fit the parameters of the two models to our data:
popt_linear, pstd_linear = F.fit_data(linear_model, 
                                      Sample_number_barium*Sample_time, 
                                      np.log(Count_rate), 
                                      Count_rate_uncertainty/Count_rate,
                                      [-0.00393908,  3.75408883])


popt_non_linear, pstd_non_linear = F.fit_data(non_linear_model, 
                                      Sample_number_barium*Sample_time, 
                                      Count_rate, 
                                      Count_rate_uncertainty,
                                      [4.36112953e+01, -4.08767785e-03])
print("The estimated optimal parameters with uncertainty by scipy optimize",
      "curve fit are:",popt_linear, "+-",pstd_linear," for the linear, and:",
      popt_non_linear, "+-", pstd_non_linear, "for the non linear model.")

#Calculating predicted y-values of models:
Count_rate_predicted_non_linear = np.zeros(len(Sample_number_barium))
Count_rate_predicted_linear = np.zeros(len(Sample_number_barium))
#And now we also calculate the y-values predicted by the linear model, 
# for the non logarithmic scale. I will use this later on in the plot
Count_rate_predicted_linear_non_linear = np.zeros(len(Sample_number_barium))

for i in range(len(Sample_number_barium)):
    Count_rate_predicted_non_linear[i] = popt_non_linear[0]*np.exp(
        popt_non_linear[1]*i*Sample_time)
    Count_rate_predicted_linear[i]= popt_linear[0]*i*Sample_time+popt_linear[1]
    Count_rate_predicted_linear_non_linear[i] = np.exp(popt_linear[1])*np.exp(
        popt_linear[0]*i*Sample_time)
   
#The Chi squared values for these models are:
chi2_non_linear = F.chi2reduced(Count_rate,Count_rate_predicted_non_linear,
                         Count_rate_uncertainty,2)
chi2_linear = F.chi2reduced(np.log(Count_rate),Count_rate_predicted_linear,
                         Count_rate_uncertainty/Count_rate,2)  
print("\nThe reduced Chi squared values for the non linear and linear",
      "model respectivly",
      "are", np.round(chi2_non_linear,4), "and", np.round(chi2_linear,4))

print("These values are very low. This means that our models fit our data",
      "very well. Thus, the euclidian distance between the data points and our",
      "curves are in general low.")
print("However, this is not necesserely a good sign. Our reduced Chi squared",
      "values should ideally both be equal to one. That we have extreamly low",
      "reduced Chi squared values implies that we have to little data.",
      "That we are in risk of overfitting our models to our data.")

#Now we calculate the estimate of the halflife, for our two models:
# We know half life = (mean lifetime)*ln(2) 
# Furthermore, we know that the parameter b in the non-linear model, is equal
# to -1/mean_lifetime. Thus, for the non-linear model, we have:
b_non_linear = popt_non_linear[1]
mean_lifetime_non_linear = -1/b_non_linear
Half_life_non_linear = mean_lifetime_non_linear*np.log(2)
#Now we calculate the uncertainty of the halflife. multiplying the quantity
# by scalars, means that we must also multiply the uncertainties by these
# scalars. However, when we take the quantity 1/a, then the uncertainty of
# 1/a is equal to the uncertainty of a divided by a^2:
Half_life_non_linear_uncertainty = -np.log(2) * ( 
    pstd_non_linear[1]/popt_non_linear[1]**2)

print("\nThe Halflife of the Barium, predicted by the non-linear model, is:",
      np.round(Half_life_non_linear,0), "+-", np.round(
          Half_life_non_linear_uncertainty,0))
#For the linear model, we have:
a_linear = popt_linear[0]
mean_lifetime_linear = -1/a_linear
Half_life_linear = mean_lifetime_linear*np.log(2)
Half_life_linear_uncertainty = -np.log(2) * (pstd_linear[0]/popt_linear[0]**2)


print("The Halflife of the Barium, predicted by the linear model, is:",
      np.round(Half_life_linear,0), "+-", np.round(
          Half_life_linear_uncertainty,0))
print("The non-linear model gave a half-life closer to the expected",
      "half-life of 2.6 minutes/156 seconds.")


print("\nThe non-linear regression method gave a half-life closer to the",
      "expected half-life of 2.6 minutes. 170 seconds is closer to 156",
      "seconds, than 176 seconds. However, note that the theoretical half",
      "life falls within both of our estimates of the half lifes with",
      "assosiated uncertaintees.")

#Now we plot these models, the data, and the theoretical curve:
#Now we create some data points on the model curve for plotting

#We calculate the predicted count rates by the theory:
# Note, theoretical halflife is 2.6 min
Count_rate_theory_predicted = np.zeros(len(Sample_number_barium))
for i in range(len(Sample_number_barium)):
    Count_rate_theory_predicted[i] = popt_non_linear[0]*np.exp(
        -i*Sample_time/(2.6*60)*np.log(2))
    
#Now we plot the original data with error bars, along with the curve fit model
plt.figure(figsize=(10,5))
plt.errorbar(Sample_number_barium*Sample_time, 
             Count_rate,Count_rate_uncertainty ,c='r', ls='', 
             marker='o',lw=1,capsize=2,
             label = 'Points of measurement with uncertainty')

plt.plot(Sample_number_barium*Sample_time,
         Count_rate_predicted_non_linear, c='b', 
         label = 'Non-linear curve fit')
plt.plot(Sample_number_barium*Sample_time,
         Count_rate_predicted_linear_non_linear, c='g', 
         label = 'linear curve fit')
plt.plot(Sample_number_barium*Sample_time,Count_rate_theory_predicted, c='y', 
         label = 'Theoretical curve')

plt.title("Count Rates of Barium (Ba-137m)")
plt.xlabel("Time in seconds (s)")
plt.ylabel("Count Rate")
plt.legend()
plt.grid()
plt.savefig("Count Rates of Barium (Ba-137m)"+'.png')
plt.show()


#Now we plot the logarithmic version of this:
plt.figure(figsize=(10,5))
plt.errorbar(Sample_number_barium*Sample_time, 
             Count_rate,Count_rate_uncertainty ,c='r', ls='', 
             marker='o',lw=1,capsize=2,
             label = 'Points of measurement with uncertainty')

plt.plot(Sample_number_barium*Sample_time,
         Count_rate_predicted_non_linear, c='b', 
         label = 'Non-linear curve fit')
plt.plot(Sample_number_barium*Sample_time,
         Count_rate_predicted_linear_non_linear, c='g', 
         label = 'linear curve fit')
plt.plot(Sample_number_barium*Sample_time,Count_rate_theory_predicted, c='y', 
         label = 'Theoretical curve')

plt.title("Logarithmic Count Rates of Barium (Ba-137m)")
plt.xlabel("Time in seconds (s)")
plt.ylabel("Count Rate")
plt.legend()
plt.grid()
plt.yscale('log')
plt.savefig("Logarithmic Count Rates of Barium (Ba-137m)"+'.png')
plt.show()

print("\nThough it is hard to distinguish the two models in the non-linear",
      "plot, in the logaritmic, linear plot, you can more easily see that",
      "the non-linear model returns a curve closer to the theoretical curve,",
      "than the curve returned by the linear model.")
print("Both models does however fit the theoretical curve quite good.",
      "Furthermore, both models are well within the uncertainty of our",
      "measurements, see plots above.")


