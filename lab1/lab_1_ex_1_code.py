import numpy as np
import matplotlib.pyplot as plt

# import our utility method library from statslab.py
import statslab as utils

# assume 5% uncertinity due to connection errors, human factors etc.
setup_uncetainty = 0.05

def current_uncertainty(current):
    """return the uncertainty in current for given values of current"""
    multimeter_uncertainty = 0.0
    if current > 100:
        multimeter_uncertainty = 1
    elif current > 10:
        return max(0.1, multimeter_uncertainty*current)
    else:
        multimeter_uncertainty = 0.01
        
    return max(multimeter_uncertainty, setup_uncetainty*current)
    
#  model function
def linear_model_function(x, a, b):
    return a*x + b 

def analyse_file(filename, title):
    # read the data for feature experiment
    print(filename)
    
    measured_voltages, measured_currents = utils.read_data(filename,
                                                           usecols=(0,1))

    # create error array for the current
    current_errors = np.vectorize(current_uncertainty)(measured_currents)
    
    # fit the measured data using curve_fit function
    popt, pstd = utils.fit_data(linear_model_function, 
                          measured_voltages, 
                          measured_currents, 
                          current_errors, 
                          guess=(1/100, 0))
    
    # generate data for predicted values using estimated speed and initial position
    # obtained using  curve fit model
    voltage_data = np.linspace(0, measured_voltages[-1], 100)
    predicted_currents = linear_model_function(voltage_data, 
                                                popt[0],
                                                popt[1])
    
    # fill in the plot details for curve fitted model in our plot_details object
    plot_data = utils.plot_details("Current Vs Voltage)")
    plot_data.errorbar_legend("measured current (mA)")
    plot_data.fitted_curve_legend("curve fit prediction")
    plot_data.x_axis_label("Voltage (V)")
    plot_data.y_axis_label("Current (mA)")
    plot_data.xdata(measured_voltages)
    plot_data.ydata(measured_currents)
    plot_data.yerrors(current_errors)
    plot_data.xdata_for_prediction(voltage_data)
    plot_data.ydata_predicted(predicted_currents)
    plot_data.legend_position("upper left")
    
    # plot the data
    utils.plot(plot_data)
    
    # calculate the chi2reduced for curve fitted model
    chi2r_curve_fit = utils.chi2reduced(measured_currents, 
                                  linear_model_function(measured_voltages, 
                                                             popt[0],
                                                             popt[1]), 
                                  current_errors, 
                                  3)
        
    print("\tLinear model slope (a) = %.2f" % popt[0])
    print("\tLinear model y-intercept (b) := %.2f" % popt[1])
    print("\tEstimated Resistance (1/a) = %.2f kiloohm" % (1/popt[0]))

    print("\tchi2reduced (Curve Fit) := %.3f" % chi2r_curve_fit)
    print("\tUncertainty in resistance  := %.3f kiloohm" % pstd[0])

    plt.savefig("lab_1_ex_1_plot_%s.png" % filename[:-4].lower())

# files to analyse
file_titles ={
    "100k.csv": "100 kiloohm",
    "Potentiometer.csv": "potentiometer"
}

for filename, title in file_titles.items():
    analyse_file(filename, title)