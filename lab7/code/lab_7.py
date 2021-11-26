#!/usr/bin/env python3
# @author: Pankaj
# -*- coding: utf-8 -*-

import math
import statslab as utils
import matplotlib.pyplot as plt
import numpy as np

# import the measured data from data files
datasets = [
    {
     "name": "AC Sine Wave (RC)",
     "file": "../data/rc_sine.csv",
     "R": 100000,
     "C": 0.022 * 10.0 ** (-6),
     "L": 0,
     "impedance_func": lambda w, R, C, L: 
         np.sqrt(1.0 / (w * C) ** 2 + R ** 2)
    },
    {
     "name": "AC Sine Wave (LR)",
     "file": "../data/lr_sine.csv",
     "R": 100000,
     "C": 0,
     "L": 46*10**-3,
     "impedance_func": lambda w, R, C, L: 
         np.sqrt( (w * L) ** 2 + R ** 2)
    },
    {
     "name": "AC Sine Wave (LCR)",
     "file": "../data/lcr_sine.csv",
     "R": 100000,
     "C": 0.022 * 10.0 ** (-6),
     "L": 46*10** (-3),
     "impedance_func": lambda w, R, C, L: 
         np.sqrt( R ** 2 + np.power(w * L - 1.0 / (w * C), 2 ) )
    }
]


def analyze_data(data):
    print(data["name"])
    print("\tAnalyzing file %s" % data["file"])
    frequency, v_total, v_r, phase = utils.read_data(data["file"],
                           usecols=(0, 1, 2, 3),
                           skiprows=1)

    # frequency = omega / 2pi
    omega = 2*math.pi*frequency

    # z = v/v_r * R
    z_measured = v_total / v_r * data["R"]
    
    # z = sqrt((omega L - 1/omega C)^2 + R^2)
    z_theory = data["impedance_func"](omega, data["R"], data["C"], data["L"])
            
    fig = plt.figure(figsize=(16,10))
    fig.tight_layout()
    
    plt.subplot(2,2,1)
    plt.semilogx(frequency,z_measured)
    
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Z (Measured) $\Omega$")

    plt.subplot(2,2,2)
    plt.semilogx(frequency, z_theory)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Z (Theoretical)  $\Omega$")

    plt.subplot(2,1,2)
    plt.semilogx(frequency, phase)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase $^\circ$")
    plt.title(data["name"], y=-0.5)
    
    plt.savefig("%s.png" % data["name"], bbox_inches='tight')
    
    
for data in  datasets:
    analyze_data(data)