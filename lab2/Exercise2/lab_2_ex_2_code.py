import numpy as np
import matplotlib.pyplot as plt
import math

# import our utility method library from statslab.py
import statslab as utils

def count_uncertainty(rate):
    return np.sqrt(rate)/20

def count_uncertainty_logarithmic(rate):
    err = count_uncertainty(rate)
    return err/rate

    
#  model function
def linear_model_function(x, a, b):
    return a*x + b 

def non_linear_model_function(x, a, b):
    return b * np.power(math.e, x * a)

filename = "Experiment1_30092021.txt"
sample_number, measured_count = utils.read_data(filename,
                                                usecols=(0,1),
                                                skiprows=2,
                                                delimiter=None)

# convert the sample  number into seconds
sample_seconds = sample_number *  20

filename = "Background_30092021.txt"
_, bg_measured_count = utils.read_data(filename,
                                                usecols=(0,1),
                                                skiprows=2,
                                                delimiter=None)

mean_background_count = np.mean(bg_measured_count)

measured_count_corrected =  (measured_count - mean_background_count)

# corrected rate
measured_count_corrected_rate = measured_count_corrected / 20

# create error array for the current
measured_count_errors = count_uncertainty(measured_count + bg_measured_count)

# create logarithmic error array for the current
measured_count_errors_log = count_uncertainty_logarithmic(measured_count_corrected_rate)

def do_linear_analysis():
    measured_count_corrected_rate_logs = np.log(measured_count_corrected_rate)
        
    # fit the measured data using curve_fit function
    popt, pstd = utils.fit_data(linear_model_function, 
                          sample_seconds, 
                          measured_count_corrected_rate_logs, 
                          measured_count_errors_log)
    
    mean_lifetime = abs(1/popt[0]/60)
    half_life = mean_lifetime * np.log(2)
    
    # generate data for predicted values using estimated resistance
    # obtained using  curve fit model
    predicted_counts = linear_model_function(sample_seconds, 
                                                popt[0],
                                                popt[1])
    
    # calculate the chi2reduced for curve fitted model
    chi2r_curve_fit = utils.chi2reduced(measured_count_corrected_rate_logs, 
                                  linear_model_function(sample_seconds, 
                                                              popt[0],
                                                              popt[1]), 
                                  measured_count_errors_log, 
                                  3)
    
    # fill in the plot details for curve fitted 
    # model in our plot_details object
    plot_data = utils.plot_details("Count Vs Time (s)")
    plot_data.errorbar_legend("count")
    plot_data.fitted_curve_legend("Fitted  Data")
    plot_data.x_axis_label("Time (s)")
    plot_data.y_axis_label("count")
    plot_data.xdata(sample_seconds)
    plot_data.ydata(measured_count_corrected_rate_logs)
    plot_data.yerrors(measured_count_errors_log)
    plot_data.xdata_for_prediction(sample_seconds)
    plot_data.ydata_predicted(predicted_counts)
    plot_data.legend_position("upper right")
    plot_data.chi2_reduced(chi2r_curve_fit)
    
    # plot the data
    utils.plot(plot_data)
        
    print(pstd)
    print("Linear model slope (a) = %.5f" % popt[0])
    print("Linear model y-intercept (b) := %.5f" % popt[1])
    print("chi2reduced (Curve Fit) := %.3f" % chi2r_curve_fit)    
    print("Mean lifetime := %.3f +/- %.5f" % (mean_lifetime, pstd[0]))  
    print("Half lifetime := %.3f +/- %.5f" % (half_life, pstd[0]))

    plt.savefig("lab_2_ex_2_plot_barium_linear.png")
    return popt
    
def do_non_linear_analysis(guess):
    # fit the measured data using curve_fit function
    popt, pstd = utils.fit_data(non_linear_model_function, 
                          sample_seconds, 
                          measured_count_corrected_rate, 
                          measured_count_errors, guess=guess)
    
    # generate data for predicted values using estimated resistance
    # obtained using  curve fit model
    predicted_rate = non_linear_model_function(sample_seconds, 
                                                popt[0],
                                                popt[1])
    
    # calculate the chi2reduced for curve fitted model
    chi2r_curve_fit = utils.chi2reduced(measured_count_corrected_rate, 
                                  non_linear_model_function(sample_seconds, 
                                                              popt[0],
                                                              popt[1]), 
                                  measured_count_errors, 
                                  3)
    
    # fill in the plot details for curve fitted 
    # model in our plot_details object
    plot_data = utils.plot_details("Rate Vs Time (s)")
    plot_data.errorbar_legend("count")
    plot_data.fitted_curve_legend("Fitted  Data")
    plot_data.x_axis_label("Time (s)")
    plot_data.y_axis_label("Rate")
    plot_data.xdata(sample_seconds)
    plot_data.ydata(measured_count_corrected_rate)
    plot_data.yerrors(measured_count_errors)
    plot_data.xdata_for_prediction(sample_seconds)
    plot_data.ydata_predicted(predicted_rate)
    plot_data.legend_position("upper right")
    plot_data.chi2_reduced(chi2r_curve_fit)
    
    # plot the data
    utils.plot(plot_data)
    
    mean_lifetime = abs((1 / popt[0])/60)
    half_life = mean_lifetime * np.log(2)
        
    print("Non Linear (a) = %.4f +/- %.4f" % (popt[0], pstd[0]))
    print("Non Linear (b) := %.2f +/- %.4f" % (popt[1],  pstd[1]))
    print("chi2reduced (Curve Fit) := %.3f" % chi2r_curve_fit)    
    print("Mean lifetime := %.3f +/- %.4f" % (mean_lifetime, pstd[0]))  
    print("Half lifetime := %.3f +/- %.4f" % (half_life, pstd[0]))

    plt.savefig("lab_2_ex_2_plot_barium_non_linear.png")    
    return popt
    
# do linear analysis
guess = do_linear_analysis()

do_non_linear_analysis((guess[0], np.power(math.e, guess[1])))