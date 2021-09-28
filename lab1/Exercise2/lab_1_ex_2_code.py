import numpy as np
import matplotlib.pyplot as plt

# import our utility method library from statslab.py
import statslab as utils

# precision iundertainity for the measurements
precision_uncertainity = 0.5

def current_uncertainty(current):
    """return the uncertainty in current for given values of current"""
    multimeter_uncertainty = 0.0
    if current > 100:
        multimeter_uncertainty = 1
    elif current > 10:
        multimeter_uncertainty = 0.1
    else:
        multimeter_uncertainty = 0.01
        
    return max(multimeter_uncertainty, precision_uncertainity)

def current_uncertainty_logarithmic(current):
    """return the uncertainty in current for given values of current"""
    multimeter_uncertainty = 0.0
    if current > 100:
        multimeter_uncertainty = 1
    elif current > 10:
        multimeter_uncertainty = 0.1
    else:
        multimeter_uncertainty = 0.01
        
    return max(precision_uncertainity, multimeter_uncertainty) / current

    
#  model function
def linear_model_function(x, a, b):
    return a*x + b 

def non_linear_model_function(x, a, b):
    return a*np.power(x, b)


filename = "Data.csv"
units = {
    "voltage": "V",
    "current": "mA",
    "resistance": "kiloohm"
}

title = "exercise 2"

measured_voltages, measured_currents = utils.read_data(filename,
                                                       usecols=(0,1))

# create error array for the current
current_errors = np.vectorize(current_uncertainty)(measured_currents)

# create logarithmic error array for the current
current_errors_logarithmic = np.vectorize(current_uncertainty_logarithmic)(measured_currents)

def do_linear_analysis():
    # fit the measured data using curve_fit function
    popt, pstd = utils.fit_data(linear_model_function, 
                          measured_voltages, 
                          measured_currents, 
                          current_errors, 
                          guess=(1/100, 0))
    
    # generate data for predicted values using estimated resistance
    # obtained using  curve fit model
    voltage_data = np.linspace(0, measured_voltages[-1], 100)
    predicted_currents = linear_model_function(voltage_data, 
                                                popt[0],
                                                popt[1])
    
    # calculate the chi2reduced for curve fitted model
    chi2r_curve_fit = utils.chi2reduced(measured_currents, 
                                  linear_model_function(measured_voltages, 
                                                             popt[0],
                                                             popt[1]), 
                                  current_errors, 
                                  3)
    
    # fill in the plot details for curve fitted 
    # model in our plot_details object
    plot_data = utils.plot_details("Current Vs Voltage (%s)" % title)
    plot_data.errorbar_legend("measured current (%s)" % units["current"])
    plot_data.fitted_curve_legend("curve fit prediction")
    plot_data.x_axis_label("Voltage (%s)" % units["voltage"])
    plot_data.y_axis_label("Current (%s)" % units["current"])
    plot_data.xdata(measured_voltages)
    plot_data.ydata(measured_currents)
    plot_data.yerrors(current_errors)
    plot_data.xdata_for_prediction(voltage_data)
    plot_data.ydata_predicted(predicted_currents)
    plot_data.legend_position("upper left")
    plot_data.chi2_reduced(chi2r_curve_fit)
    
    # plot the data
    utils.plot(plot_data)
        
    print("\tLinear model slope (a) = %.2f" % popt[0])
    print("\tLinear model y-intercept (b) := %.2f" % popt[1])
    print("\tEstimated Resistance (1/a) = %.2f %s" % (1/popt[0], 
                                                      units["resistance"]))
    
    print("\tchi2reduced (Curve Fit) := %.3f" % chi2r_curve_fit)
    print("\tUncertainty in resistance  := %.3f %s" % (pstd[0], 
                                                       units["resistance"]))
    
    plt.savefig("lab_1_ex_2_plot_linear_%s.png" % filename[:-4].lower())
    
def do_non_linear_analysis():
    # fit the measured data using curve_fit function
    popt, pstd = utils.fit_data(non_linear_model_function, 
                          measured_voltages, 
                          measured_currents, 
                          current_errors, 
                          guess=(1/100, 0))
    
    # generate data for predicted values using estimated resistance
    # obtained using  curve fit model
    voltage_data = np.linspace(0, measured_voltages[-1], 100)
    predicted_currents = non_linear_model_function(voltage_data, 
                                                popt[0],
                                                popt[1])
    
    # calculate the chi2reduced for curve fitted model
    chi2r_curve_fit = utils.chi2reduced(measured_currents, 
                                  non_linear_model_function(measured_voltages, 
                                                             popt[0],
                                                             popt[1]), 
                                  current_errors_logarithmic, 
                                  3)
    
    # fill in the plot details for curve fitted 
    # model in our plot_details object
    plot_data = utils.plot_details("Current Vs Voltage (%s)" % title)
    plot_data.errorbar_legend("measured current (%s)" % units["current"])
    plot_data.fitted_curve_legend("curve fit prediction")
    plot_data.x_axis_label("Voltage (%s)" % units["voltage"])
    plot_data.y_axis_label("Current (%s)" % units["current"])
    plot_data.xdata(measured_voltages)
    plot_data.ydata(measured_currents)
    plot_data.yerrors(current_errors)
    plot_data.xdata_for_prediction(voltage_data)
    plot_data.ydata_predicted(predicted_currents)
    plot_data.legend_position("upper left")
    plot_data.chi2_reduced(chi2r_curve_fit)
    
    # plot the data
    utils.plot(plot_data)
    
    # plot_data.set_x_log_scale(True)
    # plot_data.set_y_log_scale(True)
    utils.plot(plot_data, new_figure=False)

        
    print("\tNon-Linear model slope (a) = %.2f" % popt[0])
    print("\tNon-Linear model y-intercept (b) := %.2f" % popt[1])
    print("\tEstimated Resistance (1/a) = %.2f %s" % (1/popt[0], 
                                                      units["resistance"]))
    
    print("\tchi2reduced (Curve Fit) := %.3f" % chi2r_curve_fit)
    print("\tUncertainty in resistance  := %.3f %s" % (pstd[0], 
                                                       units["resistance"]))
    
    plt.savefig("lab_1_ex_2_plot_non_linear_%s.png" % filename[:-4].lower())
    

# do linear analysis
do_linear_analysis()

# do non-linear analysis
do_non_linear_analysis()