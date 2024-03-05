# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 15:03:57 2023

@author: okrarup
"""


import matplotlib.pyplot as plt
import numpy as np

from ssfm_functions import wavelengthToFreq, freqToWavelength, wavelengthBWtoFreqBW, freqBWtoWavelengthBW, getGammaFromFiberParams


def ramanGainFunction(freq_Hz):
    a=5.2e-14/(10e12) #m/W/Hz
    b=503e12
    c=16e12
    return a*freq_Hz*np.exp(-freq_Hz/b -(freq_Hz/c)**12 )
    


if __name__ == "__main__":
    
    
    centerWavelength = 426.9e-9 #laser wl in m  
    centerFreq = wavelengthToFreq(centerWavelength)
    freqList = np.arange(centerFreq,centerFreq+300e12,13e12)
    
    fplot=np.linspace(0,40e12,1000)

    ramanGainPlot = ramanGainFunction(fplot)
    gain_max = np.amax(ramanGainPlot)
    i_gain_max = np.where(ramanGainPlot == gain_max)
    fmax = fplot[i_gain_max]
    
    
    plt.figure()
    plt.plot(fplot/1e12,ramanGainPlot/1e-14)
    plt.plot(fmax/1e12,gain_max/1e-14,'r.',label=f"fmax={fmax/1e12},gmax={gain_max/1e-14}")
    plt.grid()
    plt.legend()
    plt.show()
    
    
    fiber_diameter = 9e-6 #m
    n2_silica=2.2e-20 #m**2/W
    gamma_test = getGammaFromFiberParams(centerWavelength,n2_silica,fiber_diameter)
    
    testAmplitude = np.sqrt(11e3)
    
    y0_list = np.ones(25)*1e-100
    
    y0_list[0] = testAmplitude
    
    
    
    
    
    