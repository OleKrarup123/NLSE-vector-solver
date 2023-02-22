# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 10:23:48 2022

@author: okrarup
"""

import numpy as np
from scipy.fftpack import fft, ifft, fftshift, ifftshift, fftfreq

from scipy.constants import pi, c


import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import cm

import os

from datetime import datetime








def getFreqRangeFromTime(time_s):
    """ 
    Calculate frequency range for spectrum based on time basis. 
    
    When plotting a discretized pulse signal as a function of time,
    a time range is needed. To plot the spectrum of the pulse, one
    can compute the FFT and plot it versus the frequency range 
    calculated by this function
    
    Parameters:
        time_s (nparray): Time range in seconds
        
    Returns:
        nparray: Frequency range in Hz 
    
    """
    
    return fftshift(fftfreq(len(time_s), d=time_s[1]-time_s[0]))

def getPhase(pulse):
    """ 
    Gets the phase of the pulse from its complex angle
    
    Calcualte phase by getting the complex angle of the pulse, 
    unwrapping it and centering on middle entry.

    Parameters:
        pulse (nparray): Pulse amplitude in time domain
        
    Returns:
        nparray: Phase of pulse at every instance in radians 

    """

    
    phi=np.unwrap(np.angle(pulse)) #Get phase starting from 1st entry
    phi=phi-phi[int(len(phi)/2)]   #Center phase on middle entry
    return phi    


def getChirp(time_s,pulse):
    """ 
    Get local chirp at every instance of pulse

    Calculate local chirp as the (negative) time derivative of the local phase

    Parameters:
        time_s (nparray): Time range in seconds
        pulse  (nparray): Pulse amplitude in time domain
        
    Returns:
        nparray: Chirp in Hz at every instance 

    """
    
    phi=getPhase(pulse)
    dphi=np.diff(phi ,prepend = phi[0] - (phi[1]  - phi[0]  ),axis=0) #Change in phase. Prepend to ensure consistent array size 
    dt  =np.diff(time_s,prepend = time_s[0]- (time_s[1] - time_s[0] ),axis=0) #Change in time.  Prepend to ensure consistent array size

    return -1.0/(2*pi)*dphi/dt #Chirp = - 1/(2pi) * d(phi)/dt
    


class timeFreq_class:
    """
    Class for storing info about the time axis and frequency axis. 
    
    Attributes:
        number_of_points (int): Number of time points
        time_step (float): Duration of each time step
        t (nparray): Array containing all the time points
        tmin (float): First entry in time array
        tmax (float): Last entry in time array
        
        centerFrequency (float): Central optical frequency
        f (nparray): Frequency range (relative to centerFrequency) corresponding to t when FFT is taken
        fmin (float): Lowest (most negative) frequency component
        fmax (float): Highest (most positive) frequency component
        freq_step (float): Frequency resolution
    """
    
    def __init__(self,N,dt,centerFrequency):        
        """
        Constructor for the timeFreq_class
        
        Parameters:
            self
            N (int): Number of time points
            dt (float): Duration of each time step
        """
        
        self.number_of_points=N
        self.time_step=dt
        t=np.linspace(0,N*dt,N)
        self.t=t-np.mean(t)
        self.tmin=self.t[0]
        self.tmax=self.t[-1]
        
        self.centerFrequency=centerFrequency
        self.f=getFreqRangeFromTime(self.t)
        self.fmin=self.f[0]
        self.fmax=self.f[-1]
        self.freq_step=self.f[1]-self.f[0]

        self.describe_config()
        
    def describe_config(self,destination = None):
        """
        Prints a description of the time and frequency info to destination
        
        Parameters:
            self
            destination (class '_io.TextIOWrapper') (optional): File to which destination should be printed. If None, print to console
        """
        
        print(" ### timeFreq Configuration Parameters ###" , file = destination)
        print(f"  Number of points \t\t= {self.number_of_points}", file = destination)
        print(f"  Start time, tmin \t\t= {self.tmin*1e12:.3f}ps", file = destination)
        print(f"  Stop time, tmax \t\t= {self.tmax*1e12:.3f}ps", file = destination)
        print(f"  Time resolution \t\t= {self.time_step*1e12:.3f}ps", file = destination)
        print("  ", file = destination)
        
        print(f"  Center frequency\t\t= {self.centerFrequency/1e12:.3f}THz", file = destination) 
        print(f"  Start frequency\t\t= {self.fmin/1e12:.3f}THz", file = destination)
        print(f"  Stop frequency \t\t= {self.fmax/1e12:.3f}THz", file = destination)
        print(f"  Frequency resolution \t\t= {self.freq_step/1e6:.3f}MHz", file = destination)
        print( "   ", file = destination)
        

    def saveTimeFreq(self):
        """
        Saves info needed to construct this timeFreq_class instance to .csv 
        file so they can be loaded later using the load_timeFreq function.
        
        Parameters:
            self
        """
        timeFreq_df = pd.DataFrame(columns=['number_of_points', 'dt_s','centerFreq_Hz'])

        timeFreq_df.loc[  len(timeFreq_df.index) ] = [self.number_of_points,
                                                  self.time_step,self.centerFrequency]
        
        timeFreq_df.to_csv("timeFreq.csv")  
        
    

def load_timeFreq(path:str): 
    """ 
    Loads timeFreq_class for previous run

    Takes a path to a previous run, opens the relevant .csv file and extracts
    stored info from which the timeFreq class for that run can be restored.

    Parameters:
        path (str): Path to previous run
        
    Returns:
        timeFreq_class: timeFreq_class used in previous run.

    """
    
    df = pd.read_csv(path+'\\timeFreq.csv')
    number_of_points = df['number_of_points']
    dt_s = df['dt_s']
    centerFreq_Hz = df['centerFreq_Hz']
    
    
    return timeFreq_class(int(number_of_points[0]), dt_s[0],centerFreq_Hz[0])
    


        

def getPower(amplitude_in_time_or_freq_domain):
    
    """ 
    Computes temporal power or PSD

    For a real electric field,    P = 1/T int_0^T( E_real**2 )dt.
    For a complex electric field, P = 0.5*|E|**2.
    Using the complex field makes calculations easier and the factor of
    0.5 is simply absorbed into the nonlinear parameter, gamma.
    Same thing works in the frequency domain.

    Parameters:
        amplitude_in_time_or_freq_domain (nparray): Temporal or spectral amplitude
        
    Returns:
        nparray: Temporal power or PSD at any instance or frequency 

    """
    
    return np.abs(amplitude_in_time_or_freq_domain)**2  



def getEnergy(time_or_freq,amplitude_in_time_or_freq_domain):
    """ 
    Computes energy of signal or spectrum

    Gets the power or PSD of the signal from 
    getPower(amplitude_in_time_or_freq_domain)
    and integrates it w.r.t. either time or 
    frequency to get the energy. 

    Parameters:
        time_or_freq (nparray): Time range in seconds or freq. range in Hz
        amplitude_in_time_or_freq_domain (nparray): Temporal or spectral amplitude
        
    Returns:
        float: Energy in J 

    """
    
    return np.trapz(getPower(amplitude_in_time_or_freq_domain),time_or_freq)





def GaussianPulse(time_s,peakAmplitude,duration_s,time_offset_s,freq_offset_Hz,chirp,order):
    """ 
    Creates Gaussian pulse

    Generates a Gaussian pulse with the specified properties. 

    Parameters:
        time_s          (nparray): Time range in seconds
        peakAmplitude   (float)  : Peak amplitude in units of sqrt(W)
        duration_s      (float)  : RMS width of Gaussian, i.e. time at which the amplitude is reduced by a factor exp(-0.5) = 0.6065 
        time_offset_s   (float)  : Time at which the Gaussian peaks
        freq_offset_Hz  (float)  : Center frequency relative to carrier frequency specified in timeFreq.
        chirp           (float)  : Dimensionless parameter controlling the chirp
        order           (int)    : Controls shape of pulse as exp(-x**(2*order)) will be approximately square for large values of 'order'
        
    Returns:
        nparray: Gaussian pulse in time domain in units of sqrt(W)

    """
    
    
    assert 1 <= order, f"Error: Order of gaussian pulse is {order}. Must be >=1"
    return peakAmplitude*np.exp(- (1+1j*chirp)/2*((time_s-time_offset_s)/(duration_s))**(2*np.floor(order)))*np.exp(-1j*2*pi*freq_offset_Hz*time_s)

def squarePulse(time_s,peakAmplitude,duration_s,time_offset_s,freq_offset_Hz,chirp): 
    """ 
    Creates square pulse
    
    Generates a square pulse using a high order super-Gaussian pulse. 
    
    Parameters:
        time_s          (nparray): Time range in seconds
        peakAmplitude   (float)  : Peak amplitude in units of sqrt(W)
        duration_s      (float)  : Width of square pulse
        time_offset_s   (float)  : Central time of square pulse
        freq_offset_Hz  (float)  : Center frequency relative to carrier frequency specified in timeFreq.
        chirp           (float)  : Dimensionless parameter controlling the chirp
        
    Returns:
        nparray: Square pulse in time domain in units of sqrt(W)
    
    """
    
    return GaussianPulse(time_s,peakAmplitude,duration_s,time_offset_s,chirp,100)*np.exp(-1j*2*pi*freq_offset_Hz*time_s)


def sechPulse(time_s,peakAmplitude,duration_s,time_offset_s,freq_offset_Hz,chirp):  
    """ 
    Creates hyperbolic secant pulse
    
    Generates a hyperbolic secant pulse (1/cosh(t)), which is the pulse shape that
    corresponds to a fundamental soliton; a solution to the NLSE for anormalous dispersion
    where the pulse remains unchanged as it propagates down the fiber.
    
    Parameters:
        time_s          (nparray): Time range in seconds
        peakAmplitude   (float)  : Peak amplitude in units of sqrt(W)
        duration_s      (float)  : Width of sech pulse, i.e. time at which the amplitude is reduced by a factor sech(1) = 0.64805 
        time_offset_s   (float)  : Time at which the Gaussian peaks
        freq_offset_Hz  (float)  : Center frequency relative to carrier frequency specified in timeFreq.
        chirp           (float)  : Dimensionless parameter controlling the chirp
        
        
        
    Returns:
        nparray: Sech pulse in time domain in units of sqrt(W)
    
    """
    return peakAmplitude/np.cosh((time_s-time_offset_s)/duration_s)*np.exp(- (1j*chirp)/2*((time_s-time_offset_s)/(duration_s))**2)*np.exp(-1j*2*pi*freq_offset_Hz*time_s)


def noise_ASE(time_s,noiseAmplitude):
    """ 
    Generates white noise in the time domain with the specified amplitude
    
    Generates an array of complex numbers with random phase from -pi to pi and 
    amplitudes distributed normally around 0 and a standard 
    deviation of noiseAmplitude in units of sqrt(W). 
    
    Parameters:
        time_s           (nparray): Time range in seconds
        noiseAmplitude   (float)  : StDev of amplitude noise
        
    Returns:
        nparray: Random amplitudes and phases to be added to signal 
    
    """
    randomAmplitudes=np.random.normal(loc=0.0, scale=noiseAmplitude, size=len(time_s))*(1+0j)
    randomPhases = np.random.uniform(-pi,pi, len(time_s))
    return randomAmplitudes*np.exp(1j*randomPhases)   


def getPulse(time_s,peakAmplitude,duration_s,time_offset_s,freq_offset_Hz,chirp,pulseType,order,noiseAmplitude):
    """ 
    Creates pulse with the specified properties

    Creates a Gaussian, sech or square pulse based on the 'pulseType' parameter.
    If pulseType == 'custom' it is assumed that the user wants to specify
    the pulse amplitude 'manually', in which case only noise is returned.

    Parameters:
        time_s          (nparray): Time range in seconds
        peakAmplitude   (float)  : Peak amplitude in units of sqrt(W)
        duration_s      (float)  : RMS width of Gaussian, i.e. time at which the amplitude is reduced by a factor exp(-0.5) = 0.6065 
        time_offset_s   (float)  : Time at which the Gaussian peaks
        freq_offset_Hz  (float)  : Center frequency relative to carrier frequency specified in timeFreq.
        chirp           (float)  : Dimensionless parameter controlling the chirp
        order           (int)    : Controls shape of pulse as exp(-x**(2*order)) will be approximately square for large values of 'order'
        
    Returns:
        nparray: Gaussian pulse in time domain in units of sqrt(W)
    
    """
    
    noise = noise_ASE(time_s,noiseAmplitude)
    
    if pulseType.lower()=="gaussian":
        return GaussianPulse(time_s,peakAmplitude,duration_s,time_offset_s,freq_offset_Hz,chirp,order)+noise
    
    if pulseType.lower()=="sech":
        return sechPulse(time_s,peakAmplitude,duration_s,time_offset_s,freq_offset_Hz,chirp)+noise
    
    if pulseType.lower()=="square":
        return squarePulse(time_s,peakAmplitude,duration_s,time_offset_s,freq_offset_Hz,chirp)+noise
    
    if pulseType.lower()=="custom":
        return noise


def getSpectrumFromPulse(time_s,pulse_amplitude):
    """ 
    Converts the amplitude of a signal in the time domain to spectral amplitude in freq. domain
    
    Uses the FFT to shift from time to freq. domain and ensures that energy is conserved
    
    Parameters:
        time_s          (nparray): Time range in seconds
        pulse_amplitude (nparray): Pulse amplitude in sqrt(W)
        
    Returns:
        nparray: spectrum amplitude in sqrt(W)/Hz.  
    
    """
    pulseEnergy=getEnergy(time_s,pulse_amplitude) #Get pulse energy
    f=getFreqRangeFromTime(time_s) 
    dt=time_s[1]-time_s[0]
    
    spectrum_amplitude=fftshift(fft(pulse_amplitude))*dt #Take FFT and do shift
    spectrumEnergy=getEnergy(f, spectrum_amplitude) #Get spectrum energy
    
    err=np.abs((pulseEnergy/spectrumEnergy-1))
    
    assert( err<1e-7 ), f'ERROR = {err}: Energy changed when going from Pulse to Spectrum!!!' 
    
    return spectrum_amplitude
    return spectrum_amplitude


def getTimeFromFrequency(frequency_Hz):  
    """ 
    Calculate time range for pulse based on frequency range. 
    
    Essentially the inverse of the getFreqRangeFromTime function. 
    If we have a frequency range and take the iFFT of a spectrum amplitude
    to get the pulse amplitude in the time domain, this function provides the
    appropriate time range.
    
    Parameters:
        frequency_Hz (nparray): Freq. range in Hz
        
    Returns:
        nparray: Time range in s
    
    """
    return fftshift(fftfreq(len(frequency_Hz), d=frequency_Hz[1]-frequency_Hz[0]))


def getPulseFromSpectrum(frequency_Hz,spectrum_amplitude):
    """ 
    Converts the spectral amplitude of a signal in the freq. domain temporal amplitude in time domain
    
    Uses the iFFT to shift from freq. to time domain and ensures that energy is conserved
    
    Parameters:
        frequency_Hz          (nparray): Frequency in Hz
        spectrum_amplitude    (nparray): Spectral amplitude in sqrt(W)/Hz
        
    Returns:
        nparray: Temporal amplitude in sqrt(W). 
    
    """   
    spectrumEnergy=getEnergy(frequency_Hz, spectrum_amplitude)
    
    time = getTimeFromFrequency(frequency_Hz)
    dt = time[1]-time[0]
     
    pulse = ifft(ifftshift(spectrum_amplitude))/dt
    pulseEnergy = getEnergy(time, pulse)
    
    err=np.abs((pulseEnergy/spectrumEnergy-1))

    assert( err<1e-7   ), f'ERROR = {err}: Energy changed when going from Spectrum to Pulse!!!' 
    
    return pulse


#Class for holding info about individual fibers
class fiber_class:
    """
    Class for storing info about a single fiber. 
    
    Attributes:
        Length (float): Length of fiber in [m]
        numberOfSteps (int): Number of identical steps the fiber is divided into
        gamma (float): Nonlinearity parameter in [1/W/m]
        beta_list (list): List of dispersion coefficients [beta2,beta3,...] [s^(entry+2)/m]
        alpha_dB_per_m (float): Attenuation coeff in [dB/m]
        alpha_Np_per_m (float): Attenuation coeff in [Np/m]
        total_loss_dB (float):  Length*alpha_dB_per_m
    """
    
    def __init__(self,L,numberOfSteps,gamma,beta_list,alpha_dB_per_m,ramanModel="None"):
        """
        Constructor for the fiber_class
        
        
        
        Parameters:
            self
            L (float): Length of fiber in [m]
            numberOfSteps (int): Number of identical steps the fiber is divided into
            gamma (float): Nonlinearity parameter in [1/W/m]
            beta_list (list): List of dispersion coefficients [beta2,beta3,...] [s^(entry+2)/m]
            alpha_dB_per_m (float): Attenuation coeff in [dB/m]
            ramanModel (str) (default="None"): String to select Raman model. Default, "None", indicates that Raman should be ignored for this fiber.
            
            
            
            
        """
        
        
        
        self.Length=L
        self.numberOfSteps = int(numberOfSteps)
        self.z_array=np.linspace(0,self.Length,self.numberOfSteps+1)
        self.dz=self.z_array[1]-self.z_array[0]
        
        self.gamma=gamma
        
        #Pad list of betas so we always have terms up to 8th order 
        while len(beta_list)<=6:
            beta_list.append(0.0)
            
        self.beta_list=beta_list
        self.alpha_dB_per_m=alpha_dB_per_m
        self.alpha_Np_per_m = self.alpha_dB_per_m*np.log(10)/10.0 #Loss coeff is usually specified in dB/km, but Nepers/km is more useful for calculations
        self.total_loss_dB =  alpha_dB_per_m*self.Length
        #TODO: Make alpha frequency dependent.  
        
        #Default: No Raman effect
        self.ramanModel=ramanModel
        self.fR=0.0
        self.tau1=0.0
        self.tau2=0.0
        
        self.RamanInFreqDomain_func = lambda freq: 0.0
        
        if ramanModel == "Agrawal":  #Raman parameters taken from Govind P. Agrawal's book, "Nonlinear Fiber Optics". 
            self.fR = 0.180  # Relative contribution of Raman effect to overall nonlinearity
            self.tau1 = 12.2*1e-15 # Average angular oscillation time of molecular bonds in silica lattice. Note: 1/(2*pi*12.2fs) = 13.05THz = Typical Raman frequency
            self.tau2 = 30.0*1e-15 # Average exponential decay time of molecular bond oscilaltions. Note: 2*1/(2*pi*30.0fs) = 10.61 THz = Typical Raman gain spectrum FWHM 
            
            #Frequency domain representation of Raman response taken from https://github.com/omelchert/GNLStools/blob/main/src/GNLStools.py
            self.RamanInFreqDomain_func = lambda freq: (self.tau1**2+self.tau2**2)/(self.tau1**2*(1-1j*freq*2*pi*self.tau2)**2+self.tau2**2) #Freq domain representation of Raman response
            
        
        
        
        self.describe_fiber()
    
    
    
    def describe_fiber(self,destination = None):
        """
        Prints a description of the fiber to destination
        
        Parameters:
            self
            destination (class '_io.TextIOWrapper') (optional): File to which destination should be printed. If None, print to console
        """
        print(' ### Characteristic parameters of fiber: ###', file = destination)
        print(f'Fiber Length [km] \t= {self.Length/1e3} ', file = destination)
        print(f'Number of Steps \t= {self.numberOfSteps} ', file = destination)
        print(f'dz [m] \t= {self.dz} ', file = destination)

        print(f'Fiber gamma [1/W/m] \t= {self.gamma} ', file = destination)
        
        for i, beta_n in enumerate(beta_list):
            print(f'Fiber beta{i+2} [s^{i+2}/m] \t= {self.beta_list[i]} ', file = destination)
        
        print(f'Fiber alpha_dB_per_m \t= {self.alpha_dB_per_m} ', file = destination)
        print(f'Fiber alpha_Np_per_m \t= {self.alpha_Np_per_m} ', file = destination)
        print(f'Fiber total loss [dB] \t= {self.total_loss_dB} ', file = destination)
        print(f'Raman Model \t= {self.ramanModel}. (fR,tau1,tau2)=({self.fR:.3},{self.tau1/1e-15:.3},{self.tau2/1e-15:.3}) ', file = destination)
        

        print(' ', file = destination)


#Class for holding info about span of concatenated fibers. 
class fiber_span_class:
    """
    Class for storing info about multiple concatenated fibers. 
    
    Attributes:
        fiber_list (list): List of fiber_class objects
        number_of_fibers_in_span (int): Number of fibers concatenated together
    """
    def __init__(self,fiber_list):
        """
        Constructor for the fiber_span_class
        
        Parameters:
            self
            fiber_list (list): List of fiber_class objects
        """
        
        self.fiber_list=fiber_list
        self.number_of_fibers_in_span=len(fiber_list)
        
    def saveFiberSpan(self):
        """
        Saves info about each fiber in span to .csv file so they can be loaded later by the load_fiber_span function 
        
        Parameters:
            self
        """
        fiber_df = pd.DataFrame(columns=['Length_m',
                                         'numberOfSteps',
                                         'gamma_per_W_per_m',
                                         'beta2_s2_per_m',
                                         'beta3_s3_per_m',
                                         'beta4_s4_per_m',
                                         'beta5_s5_per_m',
                                         'beta6_s6_per_m',
                                         'beta7_s7_per_m',
                                         'beta8_s8_per_m',
                                         'alpha_dB_per_m',
                                         'alpha_Np_per_m',
                                         'ramanModel'
                                         ])
                                         
        for fiber in self.fiber_list:
            fiber_df.loc[  len(fiber_df.index) ] = [fiber.Length,
                                                    fiber.numberOfSteps,
                                                    fiber.gamma,
                                                    fiber.beta_list[0],
                                                    fiber.beta_list[1],
                                                    fiber.beta_list[2],
                                                    fiber.beta_list[3],
                                                    fiber.beta_list[4],
                                                    fiber.beta_list[5],
                                                    fiber.beta_list[6],
                                                    fiber.alpha_dB_per_m,
                                                    fiber.alpha_Np_per_m,
                                                    fiber.ramanModel
                                                    ]
        
        fiber_df.to_csv("Fiber_span.csv")


def load_fiber_span(path:str):
    """ 
    Loads fiber_span_class for previous run
    
    Takes a path to a previous run, opens the relevant .csv file and extracts
    stored info from which the fiber_span_class for that run can be restored.
    
    Parameters:
        path (str): Path to previous run
        
    Returns:
        fiber_span_class: A class containing a list of fibers from a previous run.
    
    """    
    df = pd.read_csv(path+'\\Fiber_span.csv')
    Length_m = df['Length_m']
    numberOfSteps = df['numberOfSteps']
    gamma_per_W_per_m = df['gamma_per_W_per_m']
    beta2_s2_per_m = df['beta2_s2_per_m']
    beta3_s3_per_m = df['beta2_s3_per_m']
    beta4_s4_per_m = df['beta2_s4_per_m']
    beta5_s5_per_m = df['beta2_s5_per_m']
    beta6_s6_per_m = df['beta2_s6_per_m']
    beta7_s7_per_m = df['beta2_s7_per_m']
    beta8_s8_per_m = df['beta2_s8_per_m']
    
    alpha_dB_per_m = df['alpha_dB_per_m']
    ramanModel = df['ramanModel']
    
    fiber_list=[]
    
    for i in range(len(Length_m)):
        beta_list_i = [beta2_s2_per_m[i],
                       beta3_s3_per_m[i],
                       beta4_s4_per_m[i],
                       beta5_s5_per_m[i],
                       beta6_s6_per_m[i],
                       beta7_s7_per_m[i],
                       beta8_s8_per_m[i]]
        
        current_fiber = fiber_class(  Length_m[i],
                                    numberOfSteps[i],
                                    gamma_per_W_per_m[i], 
                                    beta_list_i,
                                    alpha_dB_per_m[i],
                                    ramanModel[i])
        fiber_list.append( current_fiber )    
    
    return fiber_span_class(fiber_list)


#Class for holding input signal sent into fiber. 
class input_signal_class:
    """
    Class for storing info about signal launched into a fiber span. 
    
    Attributes:
        Amax (float): Peak amplitude of signal in [sqrt(W)]
        Pmax (float): Peak power of signal in [W]
        duration (float): Temporal duration of signal [s]
        offset (float): Delay of signal relative to t=0 in [s]
        chirp (float): Chirping factor of sigal 
        pulseType (str): Selects pulse type from a set of pre-defined ones. Select "custom" to define the signal manually
        order (int): For n==1 a and pulseType = "Gaussian" a regular Gaussian pulse is returned. For n>=1 return a super-Gaussian  
        noiseAmplitude (float): Amplitude of added white noise in units of [sqrt(W)]. 
        
        timeFreq (timeFreq_class): Contains info about discretized time and freq axes
        
        amplitude (nparray): Numpy array containing the signal amplitude over time in [sqrt(W)]
        spectrum (nparray): Numpy array containing spectral amplitude obtained from FFT of self.amplitude in [sqrt(W)/Hz]
    """
    
    def __init__(self,timeFreq:timeFreq_class,peak_amplitude,duration,time_offset_s,freq_offset_Hz,chirp,pulseType,order,noiseAmplitude):
        """
        Constructor for input_signal_class
        
        Parameters:
            timeFreq (timeFreq_class): Contains info about discretized time and freq axes
            peak_amplitude (float): Peak amplitude of signal in [sqrt(W)]
            duration (float): Temporal duration of signal [s]
            time_offset_s (float): Delay of signal relative to t=0 in [s]
            chirp (float): Chirping factor of sigal 
            pulseType (str): Selects pulse type from a set of pre-defined ones. Select "custom" to define the signal manually
            order (int): For n==1 a and pulseType = "Gaussian" a regular Gaussian pulse is returned. For n>=1 return a super-Gaussian  
            noiseAmplitude (float): Amplitude of added white noise in units of [sqrt(W)]. 
        """

        self.Amax = peak_amplitude
        self.Pmax= self.Amax**2
        self.duration=duration
        self.time_offset_s=time_offset_s
        self.freq_offset_Hz=freq_offset_Hz
        self.chirp=chirp
        self.pulseType=pulseType
        self.order=order
        self.noiseAmplitude=noiseAmplitude        

        self.timeFreq=timeFreq
  
        
        self.amplitude = getPulse(self.timeFreq.t,
                                  peak_amplitude,
                                  duration,
                                  time_offset_s,
                                  freq_offset_Hz,
                                  chirp,
                                  pulseType,
                                  order,
                                  noiseAmplitude)
        

        if getEnergy(self.timeFreq.t, self.amplitude) == 0.0:
            self.spectrum = np.copy(self.amplitude)  
        else:
            self.spectrum = getSpectrumFromPulse(self.timeFreq.t,self.amplitude)   
        
        
        
        self.describe_input_signal()
        
    def describe_input_signal(self,destination = None):
        """
        Prints a description of the input signal to destination
        
        Parameters:
            self
            destination (class '_io.TextIOWrapper') (optional): File to which destination should be printed. If None, print to console
        """
        print(" ### Input Signal Parameters ###" , file = destination)
        print(f"  Pmax \t\t\t\t= {self.Pmax:.3f} W", file = destination)
        print(f"  Duration \t\t\t= {self.duration*1e12:.3f} ps", file = destination)
        print(f"  Time offset \t\t\t= {self.time_offset_s*1e12:.3f} ps", file = destination)
        print(f"  Freq offset \t\t\t= {self.freq_offset_Hz/1e9:.3f} GHz", file = destination)
        print(f"  Chirp \t\t\t= {self.chirp:.3f}", file = destination)
        print(f"  pulseType \t\t\t= {self.pulseType}", file = destination)
        print(f"  order \t\t\t= {self.order}", file = destination)
        print(f"  noiseAmplitude \t\t= {self.noiseAmplitude:.3f} sqrt(W)", file = destination)
        
        print( "   ", file = destination)

    def saveInputSignal(self):
        """
        Saves info needed to construct this input_signal_class instance to .csv 
        file so they can be loaded later using the load_InputSignal function.
        
        Parameters:
            self
        """
        #Initialize dataframe
        signal_df = pd.DataFrame(columns=['Amax_sqrt(W)',
                                          'Pmax_W',
                                          'duration_s',
                                          'time_offset_s',
                                          'freq_offset_Hz',             
                                          'chirp',
                                          'pulseType',
                                          'order',
                                          'noiseAmplitude_sqrt(W)'])
                                         
         
        
        #Fill it with values used for generating input signal
        signal_df.loc[  len(signal_df.index) ] = [self.Amax,
                                                  self.Pmax,
                                                  self.duration,
                                                  self.time_offset_s,
                                                  self.freq_offset_Hz,
                                                  self.chirp,
                                                  self.pulseType,
                                                  self.order,
                                                  self.noiseAmplitude]
        #Export dataframe to .csv file
        signal_df.to_csv("Input_signal.csv")
        
        #Also export timeFreq
        self.timeFreq.saveTimeFreq()
      
        if self.pulseType == "custom":
            custom_input_df =    pd.DataFrame(columns=[ "time_s", "amplitude_sqrt_W_real","amplitude_sqrt_W_imag" ] )
            
            custom_input_df["time_s"] = self.timeFreq.t
            custom_input_df["amplitude_sqrt_W_real"] = np.real(self.amplitude)
            custom_input_df["amplitude_sqrt_W_imag"] = np.imag(self.amplitude)
            
            custom_input_df.to_csv("Custom_input_signal.csv")
      

def load_InputSignal(path):    
    """ 
    Loads input_signal_class for previous run
    
    Takes a path to a previous run, opens the relevant .csv file and extracts
    stored info from which the input_signal_class for that run can be restored.
    
    Parameters:
        path (str): Path to previous run
        
    Returns:
        input_signal_class: A class containing the input signal and time base.
    
    """    
    #Open dataframe with pulse parameters
    df = pd.read_csv(path+'\\Input_signal.csv')
    
    Amax_sqrt_W             = df['Amax_sqrt(W)'][0]
    duration_s              = df['duration_s'][0]
    time_offset_s           = df['time_offset_s'][0]
    freq_offset_Hz           = df['freq_offset_Hz'][0]
    chirp                   = df['chirp'][0]
    pulseType               = df['pulseType'][0]
    order                   = int(df['order'][0])
    noiseAmplitude_sqrt_W   = df['noiseAmplitude_sqrt(W)'][0]
    
    #Load timeFreq 
    timeFreq = load_timeFreq( path )
    
    #Initialize class for loaded signal
    loaded_input_signal = input_signal_class(timeFreq,Amax_sqrt_W,duration_s,time_offset_s,freq_offset_Hz,chirp,pulseType,order, noiseAmplitude_sqrt_W)
    
    #If signal type is "custom", load the raw amplitude values
    if pulseType == "custom":
        df_custom = pd.read_csv( path + '\\Custom_input_signal.csv' )
        
        A_real = np.array(df_custom["amplitude_sqrt_W_real"])
        A_imag = np.array(df_custom["amplitude_sqrt_W_imag"])
        A = A_real+1j*A_imag
        
        loaded_input_signal.amplitude = A
        
    
    return loaded_input_signal  



def zstep_NL(z_m,fiber:fiber_class, input_signal:input_signal_class,stepmode,stepSafetyFactor):
    """ 
    Decide which approach to use for computing variable z-step size
    
    The magnitude of the z-steps can be chosen in different ways. A cautious
    approach computes the step based on the peak power divided by the 
    time resolution. The approx approach uses peak power divided by pulse
    duration.
    
    Parameters:
        z_m                 (float):              Current z-location in fiber
        fiber               (fiber_class):        Class containing fiber properties
        input_signal        (input_signal_class): Class containing signal properties
        stepmode            (str):                String describing step method
        stepSafetyFactor    (float):              Scales down calculated step by this factor
        
    Returns:
        float: calculated z-step in m 
    
    """

    
    
    
    if stepmode.lower()=="cautious":
        return np.abs(fiber.beta_list[0])*pi/(fiber.gamma*input_signal.Pmax*input_signal.duration)**2*np.exp(2*fiber.alpha_Np_per_m*z_m)/stepSafetyFactor
    
    if stepmode.lower()=="approx":
        return np.abs(fiber.beta_list[0])*pi/(fiber.gamma*input_signal.Pmax)**2/(input_signal.duration*input_signal.timeFreq.time_step)*np.exp(2*fiber.alpha_Np_per_m*z_m)/stepSafetyFactor    


    else:
        return 1.0



def getVariableZsteps( fiber:fiber_class, input_signal:input_signal_class,stepmode,stepSafetyFactor):    
    """ 
    Calculates z-steps and z-locations if a variable step size is desired.
    
    This function calls zstep_NL multiple times until the end of the array is reached.
    Then, it returns an array of the z-locations and z-steps.
    NOTE: The variable z-steps can take a long time to compute and is generally
    not much more efficient or accurate than simply selecting a fixed stepsize.      
    
    Parameters:
        fiber               (fiber_class):        Class containing fiber properties
        input_signal        (input_signal_class): Class containing signal properties
        stepmode            (str):                String describing step method
        stepSafetyFactor    (float):              Scales down calculated step by this factor
        
    Returns:
        list(nparray,nparray): List contains z_array, which are z-locations inside the fiber and dz_array, which contains step sizes 
    """    
    z_so_far=0.0
    z_array=np.array([z_so_far])
    dz_array=np.array([])
    
    
    
    dz_current_step_to_next_step = zstep_NL(0,fiber,input_signal,stepmode,stepSafetyFactor)
    
    
    while (z_so_far+ dz_current_step_to_next_step <= fiber.Length):
        z_so_far+=dz_current_step_to_next_step 
        
        z_array=np.append(z_array,z_so_far)
        dz_array= np.append(dz_array,dz_current_step_to_next_step)
        
        dz_current_step_to_next_step = zstep_NL(z_so_far,fiber,input_signal,stepmode,stepSafetyFactor)
        
    z_array=np.append(z_array,fiber.Length)
    dz_array= np.append(dz_array,fiber.Length-z_so_far)
    
    return (z_array, dz_array)

# def getZsteps(fiber:fiber_class,input_signal:input_signal_class,stepConfig_list,fiber_index=""):
#     """ 
#     Decides whether to use fixed or variable step size and returns steps.
    
#     This function either calls getVariableZsteps to get variable z-steps or simply generates
#     fixed z-steps based on the number of steps specified. Also plots and saves charts of 
#     z-locations and z-step sizes for future reference.     
    
#     Parameters:
#         fiber               (fiber_class):        Class containing fiber properties
#         input_signal        (input_signal_class): Class containing signal properties
#         stepConfig_list     (list):               Contains stepMode, stepApproach and stepSafetyFactor
#         fiber_index=""      (str):                Index of fiber in span as a string.
        
#     Returns:
#         list(nparray,nparray): List contains z_array, which are z-locations inside the fiber and dz_array, which contains step sizes 
#     """    
#     current_dir=os.getcwd()+'\\'
    
#     newFolderName = "Z-step-graphs\\"
#     zStepFolder = current_dir + newFolderName
    
#     os.makedirs(zStepFolder,exist_ok=True)
#     os.chdir(zStepFolder)
    
    
#     stepMode=stepConfig_list[0]
#     stepApproach=stepConfig_list[1]       
#     stepSafetyFactor=stepConfig_list[2]
    
#     #Initialize zinfo to default 
#     zinfo = (np.array([0,fiber.Length]),np.array([fiber.Length]))
    
#     number_of_beta_n_greater_than_zero = int(np.sum([beta_n!=0.0 for beta_n in fiber.beta_list]))
    

    
#     if (fiber.gamma == 0.0):
#         print("There is no nonlinearity, so do all dispersion and loss in one step")
#         zinfo = (np.array([0,fiber.Length]),np.array([fiber.Length]))
#     elif fiber.alpha_Np_per_m == 0.0 and number_of_beta_n_greater_than_zero == 0 and fiber.ramanModel == "None":
#         print("There is no loss or dispersion, do all NL in one step.")
#         zinfo = (np.array([0,fiber.Length]),np.array([fiber.Length]))

#     else:
        
#         if stepMode.lower() == "fixed":
            
#             if type(stepApproach) == str:
            
#                 dz=zstep_NL(0,fiber, input_signal,stepApproach,stepSafetyFactor)
#                 z_array=np.arange(0,fiber.Length,dz)
                
#                 if z_array[-1] != fiber.Length:
#                     z_array=np.append(z_array,fiber.Length)
                
#                 dz_array = np.diff( z_array)
                
                
#             else:
#                 stepApproach=int(stepApproach)
#                 z_array=np.linspace(0,fiber.Length,stepApproach+1)
#                 dz_array=np.ones( stepApproach)*(z_array[1]-z_array[0])            
    
                
#             zinfo   =(z_array,dz_array)
            
            
            
            
#         else:
#             zinfo = getVariableZsteps(fiber,input_signal,stepApproach,stepSafetyFactor)
        
#     fig,ax = plt.subplots(dpi=200)
#     ax.set_title(f"Fiber number = {fiber_index}, \n Stepmode = ({stepConfig_list[0]},{stepConfig_list[1]}), stepSafetyFactor = {stepConfig_list[2]}")
#     ax.plot(zinfo[0]/1e3,'b.',label = f"z-locs ({len(zinfo[0])})")

#     ax.set_xlabel('Entry')
#     ax.set_ylabel('z-location [km]')
#     ax.tick_params(axis='y',labelcolor='b')
    
#     ax2=ax.twinx()
#     ax2.plot(zinfo[1]/1e3,'r.',label = f"$\Delta$z-steps ({len(zinfo[1])})")
#     ax2.set_ylabel('$\Delta$z [km]')
#     ax2.tick_params(axis='y',labelcolor='r')
    
#     fig.legend(bbox_to_anchor=(1.3,0.8))
    
#     plt.savefig(f'Z-step_chart_{fiber_index}.png', 
#                 bbox_inches ="tight",
#                 pad_inches = 1,
#                 orientation ='landscape')
#     plt.show()
    
    
#     os.chdir(current_dir)
    
#     return zinfo
   


        
#Class for holding result of SSFM simulation
class ssfm_result_class:
    """
    Class for storing info about results computed by SSFM. 
    
    Attributes:
        input_signal ( input_signal_class ): Signal launched into fiber
        fiber ( fiber_class ): Fiber signal was sent through
        experimentName ( str ): Name of experiment
        dirs ( tuple ): Contains directory where current script is located and the directory where output is to be saved
        
        pulseMatrix ( nparray ): Amplitude of pulse at every z-location in fiber
        spectrumMatrix ( nparray ): Spectrum of pulse at every z-location in fiber       
    """
    def __init__(self, input_signal:input_signal_class, fiber:fiber_class,experimentName,directories):

        """
        Constructor for ssfm_result_class. 
        
       Parameters:
            input_signal ( input_signal_class ): Signal launched into fiber
            fiber ( fiber_class ): Fiber signal was sent through
            experimentName ( str ): Name of experiment
            directories ( tuple ): Contains directory where current script is located and the directory where output is to be saved 
        """ 
        self.input_signal = input_signal
        self.fiber = fiber
        self.experimentName=experimentName
        self.dirs = directories

        self.pulseMatrix = np.zeros((len(fiber.z_array),input_signal.timeFreq.number_of_points ) )*(1+0j)
        self.spectrumMatrix = np.copy(self.pulseMatrix)
        
        self.pulseMatrix[0,:]=np.copy(input_signal.amplitude)   
        self.spectrumMatrix[0,:] = np.copy(input_signal.spectrum)
        
        

def getUnitsFromValue(value):
    """ 
    Helper function for getting SI prefix (k, M, G, T, etc.) 
    
    SSFM simulations can be used for both fibers with lengths on the scale from m to km
    and for photonic integrated circuits, which may only be a few cm long. Similarly,
    power, frequencies and times of interrest may have a wide range of scales.
    This function automatically determines the SI prefix and a scaling factor to be
    used for plotting. 
    
    Parameters:
        value (float): Value whose order of magnitude is to be determined
        
    Returns:
        scalingFactor (float): If we want to plot a frequency of unknown magnitude, we would do plt.plot(f/scalingFactor) 
        prefix (str): In the label of the plot, we would write plt.plot(f/scalingFactor,label=f"Freq. [{prefix}Hz]") 
    """    
    logval=np.log10(value)
    
    scalingFactor = 1.0
    prefix = ""
    
    if logval < -12:
        scalingFactor=1e-15
        prefix="f"
        return  scalingFactor, prefix 
    
    if logval < -9:
        scalingFactor=1e-12
        prefix="p"
        return  scalingFactor, prefix 

    if logval < -6:
        scalingFactor=1e-9
        prefix="n"
        return  scalingFactor, prefix 
    
    if logval < -3:
        scalingFactor=1e-6
        prefix="u"
        return  scalingFactor, prefix 


    if logval < -2:
        scalingFactor=1e-3
        prefix="m"
        return  scalingFactor, prefix     

    if logval < 0:
        scalingFactor=1e-2
        prefix="c"
        return  scalingFactor, prefix 

    if logval < 3:
        scalingFactor=1e-0
        prefix=""
        return  scalingFactor, prefix 

    if logval < 6:
        scalingFactor=1e3
        prefix="k"
        return  scalingFactor, prefix 

    if logval < 9:
        scalingFactor=1e6
        prefix="M"
        return  scalingFactor, prefix 

    if logval < 12:
        scalingFactor=1e9
        prefix="G"
        return  scalingFactor, prefix 

    if logval < 15:
        scalingFactor=1e12
        prefix="T"
        return  scalingFactor, prefix 

    if logval > 15:
        scalingFactor=1e15
        prefix="P"
        return  scalingFactor, prefix 

    

def describe_sim_parameters(fiber:fiber_class,input_signal:input_signal_class,fiber_index,destination=None):    
    """ 
    Computes, prints and plots characteristic distances (L_eff, L_D, L_NL)
    
    When solving the NLSE, different effects such as attenuation, dispersion,
    SPM, soliton oscillations etc. take place on different length scales.
    This function computes these length scales, prints them and plots a 
    comparison, which is saved for reference.Note that the plot adaptively
    detects if beta2 is positive or negative. 
    
    Parameters:
        fiber (fiber_class): Class containing info about the current fiber
        input_signal (input_signal_class): Class containing info about input signal
        fiber_index (int): Index of fiber in the span
        destination (std) (optional): If None, print to console. Otherwise, print to file and make plot
        
    Returns:
         
    
    """    
    scalingfactor, prefix= getUnitsFromValue(fiber.Length)
    length_list=np.array([])
    #Ensure that we don't measure distances in Mm or Gm
    if scalingfactor > 1e3:
        scalingfactor = 1e3
        prefix = 'k'
    
    if destination != None:
        fig,ax=plt.subplots(dpi=200)
        ax.set_title(f" Fiber Index = {fiber_index} \nComparison of characteristic lengths") 
    
    
    
    
    print(' ### Characteristic parameters of simulation: ###', file = destination)
    print(f'  Length_fiber \t= {fiber.Length/scalingfactor:.2e} {prefix}m', file = destination)
    
    if fiber.alpha_Np_per_m>0:
    
        if fiber.alpha_Np_per_m == 0.0:
            L_eff = fiber.Length
        
        else:
            L_eff = (1-np.exp(-fiber.alpha_Np_per_m*fiber.Length))/fiber.alpha_Np_per_m
        print(f"  L_eff       \t= {L_eff/scalingfactor:.2e} {prefix}m", file = destination)
        
        length_list=np.append(length_list,L_eff)
        
    if destination != None:
        ax.barh("Fiber Length", fiber.Length/scalingfactor, color ='C0')
        
        if fiber.alpha_Np_per_m>0:
            ax.barh("Effective Length", L_eff/scalingfactor, color ='C1')


    Length_disp_array = np.ones_like(fiber.beta_list)*1.0e100    

    for i, beta_n in enumerate(fiber.beta_list):    

        if beta_n != 0.0:
            Length_disp = input_signal.duration**(2+i)/np.abs(beta_n)
            print(f"  Length_disp_{i+2} \t= {Length_disp/scalingfactor:.2e} {prefix}m", file = destination)  
            Length_disp_array[i]=Length_disp
            
            length_list=np.append(length_list,Length_disp)
        
            if destination != None:
                ax.barh(f"Dispersion Length (n = {i+2})",Length_disp/scalingfactor, color ='C2')
        
        
        else:
            Length_disp=1e100
        
        Length_disp_array[i] = Length_disp 
        
    
    if fiber.gamma !=0.0:
        Length_NL = 1/fiber.gamma/input_signal.Pmax   
        N_soliton=np.sqrt(Length_disp_array[0]/Length_NL)
    else:
        Length_NL=1e100
        N_soliton=np.NaN
    
    length_list=np.append(length_list,Length_NL)
    
    if destination != None:
        ax.barh("Nonlinear Length",Length_NL/scalingfactor, color ='C3')
    
    print(f"  Length_NL \t= {Length_NL/scalingfactor:.2e} {prefix}m", file = destination)
    print(f"  N_soliton \t= {N_soliton:.2e}", file = destination)
    print(f"  N_soliton^2 \t= {N_soliton**2:.2e}", file = destination)


    if fiber.beta_list[0]<0:
        
        z_soliton = pi/2*Length_disp
        length_list=np.append(length_list,z_soliton)
        if destination != None:
            ax.barh("Soliton Length",z_soliton/scalingfactor, color ='C4')
        
        print(' ', file = destination)
        print(f'  sign(beta2) \t= {np.sign(fiber.beta_list[0])}, so Solitons and Modulation Instability may occur ', file = destination)
        print(f"   z_soliton \t= {z_soliton/scalingfactor:.2e} {prefix}m", file = destination)
        print(f"   N_soliton \t= {N_soliton:.2e}", file = destination)
        print(f"   N_soliton^2 \t= {N_soliton**2:.2e}", file = destination)
        

        print(" ", file = destination)
        
        # https://prefetch.eu/know/concept/modulational-instability/
        f_MI=np.sqrt(2*fiber.gamma*input_signal.Pmax/np.abs(fiber.beta_list[0]))/2/pi    
        gain_MI=2*fiber.gamma*input_signal.Pmax
        print(f"   Freq. w. max MI gain = {f_MI/1e9:.2e}GHz", file = destination)
        print(f"   Max MI gain \t\t= {gain_MI*scalingfactor:.2e} /{prefix}m ", file = destination)
        print(f"   Min MI gain distance = {1/(gain_MI*scalingfactor):.2e} {prefix}m ", file = destination)
        print(' ', file = destination)
        length_list=np.append(length_list,1/gain_MI)
        if destination != None:
            ax.barh("MI gain Length",1/(gain_MI*scalingfactor), color ='C5')
        
    elif fiber.beta_list[0]>0:           
        #https://prefetch.eu/know/concept/optical-wave-breaking/
        Nmin_OWB = np.exp(3/4)/2 #Minimum N-value of Optical Wave breaking with Gaussian pulse
        
        N_ratio = N_soliton/Nmin_OWB
        if N_ratio<=1:
            Length_wave_break = 1e100
        else:
            Length_wave_break = Length_disp_array[0]/np.sqrt(N_ratio**2-1)  #Characteristic length for Optical Wave breaking with Gaussian pulse
        length_list=np.append(length_list,Length_wave_break)
        print(' ', file = destination)
        print(f'   sign(beta2) \t\t\t\t= {np.sign(fiber.beta_list[0])}, so Optical Wave Breaking may occur ', file = destination)
        print( "   Nmin_OWB (cst.) \t\t\t= 0.5*exp(3/4) (assuming Gaussian pulses)", file = destination)
        print(f"   N_ratio = N_soliton/Nmin_OWB \t= {N_ratio:.2e}", file = destination)
        print(f"   Length_wave_break \t\t\t= {Length_wave_break/scalingfactor:.2e} {prefix}m", file = destination)    
    
        if destination != None:
            ax.barh("OWB Length",Length_wave_break/scalingfactor, color ='C6')
    
    if destination != None:
        ax.barh("$\Delta$z",fiber.dz/scalingfactor, color ='C7')
        length_list=np.append(length_list,fiber.dz)
            
        ax.set_xscale('log')
        ax.set_xlabel(f'Length [{prefix}m]')
        
        Lmin = np.min(length_list)/scalingfactor*1e-1 
        Lmax = fiber.Length/scalingfactor*1e2
        ax.set_xlim(Lmin ,Lmax )

        plt.savefig(f'Length_chart_{fiber_index}.png', 
                    bbox_inches ="tight",
                    pad_inches = 1,
                    orientation ='landscape')
    
        plt.show()
    
    #End of describe_sim_parameters

   

def describe_run( current_time, current_fiber:fiber_class,  current_input_signal:input_signal_class,fiber_index=""  ,destination = None):
    """ 
    Prints info about fiber, characteristic lengths and stepMode
    
    Calls the self-describe function of fiber, the describe_sim_parameters function and prints stepMode info to specified destination 
    
    Parameters:
        current_time           (datetime): Current date and time at which run was initiated
        current_fiber          (fiber_class): Info about current fiber 
        current_input_signal   (input_signal_class): Info about input signal
        fiber_index=""         (str) (optional): String of integer indexing fiber in fiber span 
        destination = None     (class '_io.TextIOWrapper') (optional): If None, print to console. Else, print to specified file
        
    Returns:
    """  
    print("Info about fiber",file = destination )
    current_fiber.describe_fiber(destination = destination)
    print(' ', file = destination)
    
    
    describe_sim_parameters(current_fiber,current_input_signal,fiber_index,destination=destination)
    
    



def describeInputConfig(current_time, fiber:fiber_class,  input_signal:input_signal_class,fiber_index=""):
    """ 
    Prints info about fiber, characteristic lengths and stepMode
    
    Calls the self-describe function of fiber, the describe_sim_parameters function and prints stepMode info to console and file
    
    Parameters:
        current_time            (datetime): Current date and time at which run was initiated
        fiber                   (fiber_class): Info about current fiber 
        input_signal            (input_signal_class): Info about input signal
        fiber_index=""          (str) (optional): String of integer indexing fiber in fiber span 
        
    Returns:
        
    """  
    with open(f"input_config_description_{fiber_index}.txt","w") as output_file:
            #Print info to terminal

            describe_run( current_time, fiber,  input_signal,fiber_index=fiber_index)
            
            #Print info to file
            describe_run( current_time, fiber,  input_signal, fiber_index=fiber_index  ,destination = output_file)    


def createOutputDirectory(experimentName):
    os.chdir(os.path.realpath(os.path.dirname(__file__)))
    base_dir=os.getcwd()+'\\'
    os.chdir(base_dir)
    

    current_dir = ""
    current_time = datetime.now()
    
    if experimentName == "most_recent_run":
        current_dir = base_dir+"most_recent_run\\"
        overwrite_folder_flag = True  
    else: 
        
        current_dir =base_dir+ f"Simulation Results\\{experimentName}\\{current_time.year}_{current_time.month}_{current_time.day}_{current_time.hour}_{current_time.minute}_{current_time.second}\\"
        overwrite_folder_flag = False 
        
    os.makedirs(current_dir,exist_ok=overwrite_folder_flag)
    os.chdir(current_dir)
    
    print(f"Current time is {current_time}")
    print("Current dir is "+current_dir)
    
    return (base_dir,current_dir) , current_time



def saveStepConfig(stepConfig):
    """ 
    Saves stepConfig to .csv file
    
    Uses pandas to save stepConfig to .csv file so it can be loaded later
    
    Parameters:
        stepConfig (list): Contains stepmode ('fixed'' or 'variable'), stepNumber or stepApproach ('cautious' or 'adaptive') and stepSafetyFactor (float)
        
    Returns:
         
    
    """
    #Initialize dataframe
    stepConfig_df = pd.DataFrame(columns=['stepmode',
                                      'stepNumber_or_stepApproach',
                                      'SafetyFactor'])
                                     
    #Fill it with values
    stepConfig_df.loc[  len(stepConfig_df.index) ] = [stepConfig[0],
                                                      stepConfig[1],
                                                      stepConfig[2]]
    #Export dataframe to .csv file
    stepConfig_df.to_csv("stepConfig.csv")        


def load_StepConfig(path):
    """ 
    Loads stepConfig from previous run
    
    Loads stepConfig from previous run when path to run folder is specified. 
    
    Parameters:
        path (str): Path to folder
        
    Returns:
        list: Contains stepMode ('fixed'' or 'variable'), stepNumber or stepApproach ('cautious' or 'adaptive') and stepSafetyFactor (float)
    
    """    
    df = pd.read_csv(path+'\\stepConfig.csv')
    
    stepmode = df['stepmode'][0]
    stepNumber_or_stepApproach = df['stepNumber_or_stepApproach'][0]
    SafetyFactor = df['SafetyFactor'][0]
    
    return (stepmode,stepNumber_or_stepApproach,SafetyFactor)
    
    
def load_previous_run(basePath):
    """ 
    Loads all relevant info about previous run
    
    When path to previous run folder is specified, open .csv files describing fiber, signal and stepconfig.
    Use the stored values to reconstruct the parameters for the run.
    
    Parameters:
        basePath (str): Path to run folder
        
    Returns:
        fiber_span (fiber_span_class): Info about fiber span consisting of 1 or more fibers
        input_signal (input_signal_class): Info about input signal
        stepConfig (list):  Info about step configuration
    
    """    
    print(f"Loading run in {basePath}")
    
    fiber_span      = load_fiber_span(basePath+'\\input_info\\')
    input_signal    = load_InputSignal(basePath+'\\input_info\\')
    stepConfig      = load_StepConfig( basePath+'\\input_info\\')
    
    print(f"Successfully loaded run in {basePath}")
    
    return fiber_span, input_signal, stepConfig


def NL_simple(fiber:fiber_class, timeFreq:timeFreq_class, pulse,dz):
    return np.exp(1j*fiber.gamma*getPower(pulse)*dz)

def NL_full(fiber:fiber_class, timeFreq:timeFreq_class, pulse,dz):
    
    fR = fiber.fR
    freq = timeFreq.f
    t = timeFreq.t
    
    f0 = timeFreq.centerFrequency
    RamanInFreqDomain=fiber.RamanInFreqDomain_func(freq)

    
    NR_func = lambda current_pulse: (1-fR)*getPower(current_pulse)*current_pulse + fR*current_pulse*getPulseFromSpectrum(freq,getSpectrumFromPulse(t,getPower(current_pulse))*RamanInFreqDomain)
    
    NR_in_freq_domain = 1j*dz*fiber.gamma*(1.+freq/f0)*getPulseFromSpectrum(freq,NR_func(getSpectrumFromPulse(timeFreq.t, pulse)))
    
    
    return np.exp( getPulseFromSpectrum(freq, NR_in_freq_domain) ) 
    
    

    

def SSFM(fiber_span:fiber_span_class,
         input_signal:input_signal_class,
         experimentName ="most_recent_run",
         showProgressFlag = False):
    """ 
    Runs the Split-Step Fourier method and calculates field throughout fiber
    
    Runs the SSFM to solve the NLSE with the specified parameters. Goes through the following steps:
        1) Create folder for saving input config and results
        2) Loops over fibers in fiber_span. Gets zsteps for each and runs the SSFM
        3) Stores results of each fiber in a separate ssfm_result_class and uses pulse at the end as input to next one
        4) Returns list of ssfm_result_class objects
    
    Parameters:
        fiber_span (fiber_span_class): Class holding fibers through which the signal is propagated
        input_signal (input_signal_class): Class holding info about initial input signal
        numberOfSteps = 2**10 (optional): Number of z-steps taken during SSFM. 
        experimentName ="most_recent_run" (optional): Name of folder for present simulation.
        
    Returns:
        list: List of ssfm_result_class corresponding to each fiber segment.  
    
    """
    print("########### Initializing SSFM!!! ###########")
    
    t = input_signal.timeFreq.t
    f = input_signal.timeFreq.f
    
    
    
    #Create output directory, switch to it and return appropriate paths and current time
    dirs , current_time = createOutputDirectory(experimentName)
    
    
    #Make new folder to hold info about the input signal and fiber span
    base_dir    = dirs[0]
    current_dir = dirs[1]
    
    newFolderName = "input_info\\"
    newFolderPath = newFolderName
    os.makedirs(newFolderPath,exist_ok=True)
    os.chdir(newFolderPath)

    #Save parameters of fiber span to file in directory
    fiber_span.saveFiberSpan()
    
    #Save input signal parameters
    input_signal.saveInputSignal()
    
    #saveStepConfig(stepConfig)
    
    
    
    #Return to main output directory
    os.chdir(current_dir)
    
    #TODO: Make sure code handles current_input_signal correctly for concatenated fibers!!!
    current_input_signal = input_signal
    
    ssfm_result_list = []
    
    print(f"Starting SSFM loop over {len(fiber_span.fiber_list)} fibers")
    
    for fiber_index, fiber in enumerate(fiber_span.fiber_list):
    
        print(f"Propagating through fiber number {fiber_index+1} out of {fiber_span.number_of_fibers_in_span}")
 
        
        #Get z-steps and z-locations throughout fiber and save plots of these values to new folder
        #zinfo = getZsteps(fiber,current_input_signal,stepConfig,fiber_index=str(fiber_index))
        

        
        #Initialize arrays to store pulse and spectrum throughout fiber
        ssfm_result = ssfm_result_class(current_input_signal,fiber,experimentName,dirs)

    
        newFolderName = "Length_info\\"
        newFolderPath = newFolderName
        os.makedirs(newFolderPath,exist_ok=True)
        os.chdir(newFolderPath)

        #Print simulation info to both terminal and .txt file in output folder
        describeInputConfig(current_time, fiber,  current_input_signal,fiber_index=str(fiber_index))
        
        #Return to main output directory
        os.chdir(current_dir)
        

        
    
        
        
        
        #Pre-calculate dispersion term
        dispterm=np.zeros_like(input_signal.timeFreq.f)*1.0
        for n, beta_n in enumerate(fiber.beta_list):
            dispterm+=beta_n/np.math.factorial(n)*(2*pi*input_signal.timeFreq.f)**(n+2)
       
        
        #Pre-calculate effect of dispersion and loss as it's the same everywhere
        disp_and_loss=np.exp(fiber.dz*(1j*dispterm-fiber.alpha_Np_per_m/2))
        disp_and_loss_half_step = disp_and_loss**0.5
        #Precalculate constants for nonlinearity
        
        NL_function = NL_simple
        
        if fiber.ramanModel != "None":
            NL_function = NL_full
        
        
        
        #Initialize arrays to store temporal profile and spectrum while calculating SSFM
        
        spectrum = np.copy(input_signal.spectrum )*disp_and_loss_half_step
        pulse    = getPulseFromSpectrum(input_signal.timeFreq.f, spectrum)
        
    

        print(f"Running SSFM with nsteps = {fiber.numberOfSteps}")
        updates = 0
        for z_step_index in range(fiber.numberOfSteps):   
            pulse*=NL_function(fiber,input_signal.timeFreq,pulse,fiber.dz) #Apply nonlinearity
            
            spectrum = getSpectrumFromPulse(t, pulse)*(disp_and_loss) #Go to spectral domain and apply disp and loss
            
            
            pulse=getPulseFromSpectrum(f, spectrum) #Return to time domain 
            
            
            #Store results and repeat
            ssfm_result.spectrumMatrix[z_step_index+1,:]=spectrum*disp_and_loss_half_step
            ssfm_result.pulseMatrix[z_step_index+1,:]=getPulseFromSpectrum(f, ssfm_result.spectrumMatrix[z_step_index+1,:])
            
            


            finished = 100*(z_step_index/fiber.numberOfSteps)
            if divmod(finished, 10)[0] > updates and showProgressFlag == True:
                updates += 1
                print(f"SSFM progress through fiber number {fiber_index+1} = {np.floor(finished):.2f}%")
            
            
        #Append list of output results
        
        ssfm_result_list.append(ssfm_result)
        
        #Take pulse at output of this fiber and feed it into the next one
        current_input_signal.amplitude =np.copy(ssfm_result.pulseMatrix[z_step_index+1,:])
        current_input_signal.Pmax = np.max(getPower(current_input_signal.amplitude))
        

    print("Finished running SSFM!!!")
    
    #Exit current output directory and return to base directory.
    os.chdir(base_dir)
        
        
    return ssfm_result_list


def saveplot(basename):
    """ 
    Helper function for adding file type suffix to name of plot
    
    Helper function for adding file type suffix to name of plot
    
    Parameters:
        basename (str): Name to which a file extension is to be appended if not already present. 
        
    Returns:
        
    
    """
    
    if basename.lower().endswith(('.pdf','.png','.jpg')) == False:
        basename+='.png'
        
    plt.savefig(basename, bbox_inches='tight', pad_inches=0)


def unpackZvals(ssfm_result_list):
    """ 
    Unpacks z_values of individual fibers in ssfm_result_list into single array
    
    For a span of 5 fibers with 100 steps each, this function concatenates the
    arrays like this:
        
        Merge A[0:98] with (B[0:98] + A[end])      to create z_temp
        Merge z_temp with  (C[0:98] + z_temp[end]) to create z_temp
        Merge z_temp with  (D[0:98] + z_temp[end]) to create z_temp
        Merge z_temp with  (E[0:99] + z_temp[end]) to create z_temp
        
    Note that the final entry in the first 4 arrays are discarded as they are
    identical to the first element in the next one.
    
    
    Parameters:
        ssfm_result_list (list): List of ssmf_result_class objects corresponding to each fiber segment
        
    Returns:
        nparray: z_values for each fiber concatenated together.  
    
    """    
    if len(ssfm_result_list)==1:
        return ssfm_result_list[0].fiber.z_array
    
    zvals =np.array([])
    number_of_fibers = len(ssfm_result_list)
    previous_length = 0
    for i, ssfm_result in enumerate(ssfm_result_list):
        
    
        if i==0:
            zvals = np.copy(ssfm_result.fiber.z_array[0:-1])
            
        elif  (i>0) and (i< number_of_fibers-1):
            zvals = np.append(zvals,ssfm_result.fiber.z_array[0:-1]+previous_length) 
            
            
        elif i==number_of_fibers-1:
            zvals = np.append(zvals,ssfm_result.fiber.z_array+previous_length) 

       
        previous_length += ssfm_result.fiber.Length    
        
    return zvals

def unpackMatrix(ssfm_result_list,zvals,timeFreq,pulse_or_spectrum):
    """ 
    Unpacks pulseMatrix or spectrumMatrix for individual fibers in ssfm_result_list into single array
    
    For a span of 5 fibers with 100 steps each, this function concatenates the
    arrays like this:
        
        Merge A[0:98,:] with B[0:98,:]  to create mat_temp
        Merge mat_temp with  C[0:98,:]  to create mat_temp
        Merge mat_temp with  D[0:98,:]  to create mat_temp
        Merge mat_temp with  E[0:99,:]  to create mat_temp
        
    Note that the final entry in the first 4 arrays are discarded as they are
    identical to the first element in the next one.
    
    
    Parameters:
        ssfm_result_list (list): List of ssmf_result_class objects corresponding to each fiber segment
        zvals (nparray) : Array of unpacked z_values from unpackZvals. Needed for pre-allocating returned matrix
        timeFreq (timeFreq_class): timeFreq for simulation. Needed for pre-allocation
        pulse_or_spectrum (str) : Indicates if we want to unpack pulseMatrix or spectrumMatrix
        
    Returns:
        nparray: Array of size (n_z_steps,n_time_steps) describing pulse amplitude or spectrum for whole fiber span.
    
    """  
    number_of_fibers = len(ssfm_result_list)
    
    
    
    matrix=np.zeros( ( len(zvals), len(  timeFreq.t )  ) )*(1+0j)
    
    starting_row  = 0
    
    for i, ssfm_result in enumerate(ssfm_result_list):
        
        if pulse_or_spectrum.lower()=="pulse":
            sourceMatrix = ssfm_result.pulseMatrix
        elif pulse_or_spectrum.lower()=="spectrum":
            sourceMatrix = ssfm_result.spectrumMatrix
        else:
            print("ERROR: Please set pulse_or_spectrum to either 'pulse' or 'spectrum'!!!")
            return
        
        if number_of_fibers == 1:
            return sourceMatrix
        
        
        if i==0:
            matrix[0: len(ssfm_result.fiber.z_array)-1, :] = sourceMatrix[0: len(ssfm_result.fiber.z_array)-1, :]
            
        elif  (i>0) and (i< number_of_fibers-1):

            matrix[starting_row : starting_row + len(ssfm_result.fiber.z_array)-1, :] = sourceMatrix[0: len(ssfm_result.fiber.z_array)-1, :]

        elif i==number_of_fibers-1:

            matrix[starting_row : starting_row + len(ssfm_result.fiber.z_array), :] = sourceMatrix[0:len(ssfm_result.fiber.z_array), :]
            
        starting_row +=len(ssfm_result.fiber.z_array)-1
    
    
    return matrix        

          
def plotFirstAndLastPulse(ssfm_result_list, nrange:int, dB_cutoff,**kwargs):
    """ 
    Plots input pulse and output pulse of simulation
    
    Line plot of input pulse and output pulse of SSFM run centered in the middle of the time array and with nrange points on either side
    
    Parameters:
        ssfm_result_list (list): List of ssmf_result_class objects corresponding to each fiber segment
        nrange (int): Determines how many points on either side of the center we wish to plot  
        dB_cutoff : Lowest y-value in plot is this many dB smaller than the peak power
        **kwargs: If firstandlastpulsescale=='log' is contained in keyword args, set y-scale to log
        
    Returns:
    """    

    
    timeFreq = ssfm_result_list[0].input_signal.timeFreq
    
    Nmin = np.max([int(timeFreq.number_of_points/2-nrange),0])
    Nmax = np.min([int(timeFreq.number_of_points/2+nrange),timeFreq.number_of_points-1])   
     
    zvals = unpackZvals(ssfm_result_list)
    

    t=timeFreq.t[Nmin:Nmax]*1e12



    P_initial=getPower(ssfm_result_list[0].pulseMatrix[0,Nmin:Nmax])
    P_final=getPower(ssfm_result_list[-1].pulseMatrix[-1,Nmin:Nmax])
    
    scalingFactor,prefix=getUnitsFromValue(np.max(zvals))
    
    os.chdir(ssfm_result_list[0].dirs[1])
    fig, ax = plt.subplots(dpi=200)
    ax.set_title("Initial pulse and final pulse")
    ax.plot(t,P_initial,label=f"Initial Pulse at z = 0{prefix}m")
    ax.plot(t,P_final,label=f"Final Pulse at z = {zvals[-1]/scalingFactor}{prefix}m")
    
    ax.set_xlabel("Time [ps]")
    ax.set_ylabel("Power [W]")
    
    
    for kw, value in kwargs.items():
        if kw.lower()=='firstandlastpulsescale' and value.lower()=='log':
            ax.set_yscale('log')

    ax.legend(bbox_to_anchor=(1.15,0.8))
    saveplot('first_and_last_pulse')
    plt.show() 
    os.chdir(ssfm_result_list[0].dirs[0])


def plotPulseMatrix2D(ssfm_result_list, nrange:int, dB_cutoff):
    """ 
    Plots amplitude calculated by SSFM as colour surface
    
    2D colour plot of signal amplitude in time domain throughout entire fiber span
    normalized to the highest peak power throughout. 
    
    Parameters:
        ssfm_result_list (list): List of ssmf_result_class objects corresponding to each fiber segment
        nrange (int): Determines how many points on either side of the center we wish to plot  
        dB_cutoff : Lowest y-value in plot is this many dB smaller than the peak power
        
    Returns:
    """   
    
    timeFreq = ssfm_result_list[0].input_signal.timeFreq
    
    Nmin = np.max([int(timeFreq.number_of_points/2-nrange),0])
    Nmax = np.min([int(timeFreq.number_of_points/2+nrange),timeFreq.number_of_points-1])   
    
   
     
    zvals = unpackZvals(ssfm_result_list)
    
    matrix = unpackMatrix(ssfm_result_list,zvals,timeFreq,"pulse")
    

    #Plot pulse evolution throughout fiber in normalized log scale
    os.chdir(ssfm_result_list[0].dirs[1])
    fig, ax = plt.subplots(dpi=200)
    ax.set_title('Pulse Evolution (dB scale)')
    t = timeFreq.t[Nmin:Nmax]*1e12
    z = zvals
    T, Z = np.meshgrid(t, z)
    P=getPower(matrix[:,Nmin:Nmax]  )/np.max(getPower(matrix[:,Nmin:Nmax]))
    P[P<1e-100]=1e-100
    P = 10*np.log10(P)
    P[P<dB_cutoff]=dB_cutoff
    surf=ax.contourf(T, Z, P,levels=40, cmap="jet")
    ax.set_xlabel('Time [ps]')
    ax.set_ylabel('Distance [m]')
    cbar=fig.colorbar(surf, ax=ax)
    saveplot('pulse_evo_2D') 
    plt.show()
    os.chdir(ssfm_result_list[0].dirs[0])

def plotPulseMatrix3D(ssfm_result_list, nrange:int, dB_cutoff):
    """ 
    Plots amplitude calculated by SSFM as 3D colour surface
    
    3D colour plot of signal amplitude in time domain throughout entire fiber span
    normalized to the highest peak power throughout. 
    
    Parameters:
        ssfm_result_list (list): List of ssmf_result_class objects corresponding to each fiber segment
        nrange (int): Determines how many points on either side of the center we wish to plot  
        dB_cutoff : Lowest y-value in plot is this many dB smaller than the peak power
        
    Returns:
    """   
    
    timeFreq = ssfm_result_list[0].input_signal.timeFreq   
    
    Nmin = np.max([int(timeFreq.number_of_points/2-nrange),0])
    Nmax = np.min([int(timeFreq.number_of_points/2+nrange),timeFreq.number_of_points-1])   
    
 
    
    zvals = unpackZvals(ssfm_result_list)
    matrix = unpackMatrix(ssfm_result_list,zvals,timeFreq,"pulse")
  
    #Plot pulse evolution in 3D
    os.chdir(ssfm_result_list[0].dirs[1])
    fig, ax = plt.subplots(1,1, figsize=(10,7),subplot_kw={"projection": "3d"})
    plt.title("Pulse Evolution (dB scale)")

    t = timeFreq.t[Nmin:Nmax]*1e12
    z = zvals
    T_surf, Z_surf = np.meshgrid(t, z)
    P_surf=getPower(matrix[:,Nmin:Nmax]  )/np.max(getPower(matrix[:,Nmin:Nmax]))
    P_surf[P_surf<1e-100]=1e-100
    P_surf = 10*np.log10(P_surf)
    P_surf[P_surf<dB_cutoff]=dB_cutoff
    # Plot the surface.
    surf = ax.plot_surface(T_surf, Z_surf, P_surf, cmap=cm.jet,
                            linewidth=0, antialiased=False)
    ax.set_xlabel('Time [ps]')
    ax.set_ylabel('Distance [m]')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    saveplot('pulse_evo_3D')
    plt.show()
    os.chdir(ssfm_result_list[0].dirs[0])


def plotPulseChirp2D(ssfm_result_list, nrange:int, dB_cutoff,**kwargs):
    """ 
    Plots local chirp throughout entire fiber span.
    
    2D colour plot of local chirp throughout entire fiber span with red indicating
    lower frequencies and blue indicating higher ones. 
    
    Parameters:
        ssfm_result_list (list): List of ssmf_result_class objects corresponding to each fiber segment
        nrange (int): Determines how many points on either side of the center we wish to plot  
        dB_cutoff : Lowest y-value in plot is this many dB smaller than the peak power
        **kwargs : If chirpPlotRange=(fmin,fmax) is contained in **kwargs, use these values to set color scale. 
        
    Returns:
    """     
    
    timeFreq = ssfm_result_list[0].input_signal.timeFreq   
    
    Nmin = np.max([int(timeFreq.number_of_points/2-nrange),0])
    Nmax = np.min([int(timeFreq.number_of_points/2+nrange),timeFreq.number_of_points-1])   
      
    
    zvals = unpackZvals(ssfm_result_list)
    matrix = unpackMatrix(ssfm_result_list,zvals,timeFreq,"pulse")

    #Plot pulse evolution throughout fiber  in normalized log scale
    os.chdir(ssfm_result_list[0].dirs[1])
    fig, ax = plt.subplots(dpi=200)
    ax.set_title('Pulse Chirp Evolution')
    t = timeFreq.t[Nmin:Nmax]*1e12
    z = zvals
    T, Z = np.meshgrid(t, z)
    
    
    Cmatrix=np.ones( (len(z),len(t))  )*1.0

    for i in range(len(zvals)):
        Cmatrix[i,:]=getChirp(t/1e12,matrix[i,Nmin:Nmax])/1e9

    
    chirpplotrange_set_flag = False
    for kw, value in kwargs.items():
        if kw.lower()=='chirpplotrange' and type(value)==tuple:
            Cmatrix[Cmatrix<value[0]]=value[0]
            Cmatrix[Cmatrix>value[1]]=value[1]
            chirpplotrange_set_flag = True

    if chirpplotrange_set_flag == False:
        Cmatrix[Cmatrix<-50]=-50 #Default fmin = -50GHz
        Cmatrix[Cmatrix> 50]=50  #Default fmax = -50GHz
        
    surf=ax.contourf(T, Z, Cmatrix,levels=40,cmap='RdBu')
    
    ax.set_xlabel('Time [ps]')
    ax.set_ylabel('Distance [m]')
    cbar=fig.colorbar(surf, ax=ax)
    cbar.set_label('Chirp [GHz]')
    saveplot('chirp_evo_2D') 
    plt.show()
    os.chdir(ssfm_result_list[0].dirs[0])



def plotEverythingAboutPulses(ssfm_result_list, 
                              nrange:int, 
                              dB_cutoff, **kwargs):
    """ 
    Generates all plots of pulse amplitudes throughout fiber span
    
    Calls plotFirstAndLastPulse, plotPulseMatrix2D, plotPulseMatrix3D and plotPulseChirp2D
    sequentially and saves the plots in the appropriate directory
    
    Parameters:
        ssfm_result_list (list): List of ssmf_result_class objects corresponding to each fiber segment
        nrange (int): Determines how many points on either side of the center we wish to plot  
        dB_cutoff : Lowest y-value in plot is this many dB smaller than the peak power
        **kwargs (optional):     
    
    Returns:

    
    """  
    print('  ')
    plotFirstAndLastPulse(ssfm_result_list, nrange, dB_cutoff,**kwargs)
    plotPulseMatrix2D(ssfm_result_list,nrange,dB_cutoff)
    plotPulseChirp2D(ssfm_result_list,nrange,dB_cutoff,**kwargs) 
    plotPulseMatrix3D(ssfm_result_list,nrange,dB_cutoff)
    print('  ')



def plotFirstAndLastSpectrum(ssfm_result_list, nrange:int, dB_cutoff):
    """ 
    Plots input spectrum and output spectrum of simulation
    
    Line plot of input spectrum and output spectrum of SSFM run centered in 
    the middle of the time array and with nrange points on either side
    
    Parameters:
        ssfm_result_list (list): List of ssmf_result_class objects corresponding to each fiber segment
        nrange (int): Determines how many points on either side of the center we wish to plot  
        dB_cutoff : Lowest y-value in plot is this many dB smaller than the peak power
        
    Returns:
    """    
    timeFreq = ssfm_result_list[0].input_signal.timeFreq
    center_freq_Hz = timeFreq.centerFrequency
    Nmin = np.max([int(timeFreq.number_of_points/2-nrange),0])
    Nmax = np.min([int(timeFreq.number_of_points/2+nrange),timeFreq.number_of_points-1])   
    
  
    
    zvals = unpackZvals(ssfm_result_list)
    
    
    
    
    P_initial=getPower(ssfm_result_list[0].spectrumMatrix[0,Nmin:Nmax])
    P_final=getPower(ssfm_result_list[-1].spectrumMatrix[-1,Nmin:Nmax])



    Pmax_initial = np.max(P_initial)
    Pmax_final = np.max(P_final)
    Pmax=np.max([Pmax_initial,Pmax_final]) 

    f=(timeFreq.f[Nmin:Nmax])/1e12#+center_freq_Hz
    
    scalingFactor,prefix=getUnitsFromValue(np.max(zvals))
    os.chdir(ssfm_result_list[0].dirs[1])
    fig,ax = plt.subplots(dpi=200)
    ax.set_title("Initial spectrum and final spectrum")
    ax.plot(f,P_initial,label=f"Initial Spectrum at {zvals[0]}{prefix}m")
    ax.plot(f,P_final,label=f"Final Spectrum at {zvals[-1]/scalingFactor}{prefix}m")
    ax.set_xlabel("Freq. [THz]")
    ax.set_ylabel("PSD [W/GHz]")
    ax.set_yscale('log')
    ax.set_ylim(Pmax/(10**(-dB_cutoff/10)),2*Pmax)
    fig.legend(bbox_to_anchor=(0.95,0.8))
    saveplot('first_and_last_spectrum')
    plt.show()
    os.chdir(ssfm_result_list[0].dirs[0])


def plotSpectrumMatrix2D(ssfm_result_list, nrange:int, dB_cutoff):
    """ 
    Plots spectrum calculated by SSFM as colour surface
    
    2D colour plot of spectrum in freq domain throughout entire fiber span
    normalized to the highest peak power throughout. 
    
    Parameters:
        ssfm_result_list (list): List of ssmf_result_class objects corresponding to each fiber segment
        nrange (int): Determines how many points on either side of the center we wish to plot  
        dB_cutoff : Lowest y-value in plot is this many dB smaller than the peak power
        
    Returns:
    """     
    timeFreq = ssfm_result_list[0].input_signal.timeFreq   
    zvals = unpackZvals(ssfm_result_list)
    matrix = unpackMatrix(ssfm_result_list,zvals,timeFreq,"spectrum")
    
    Nmin = np.max([int(timeFreq.number_of_points/2-nrange),0])
    Nmax = np.min([int(timeFreq.number_of_points/2+nrange),timeFreq.number_of_points-1])   
    center_freq_Hz = timeFreq.centerFrequency


    #Plot pulse evolution throughout fiber in normalized log scale
    os.chdir(ssfm_result_list[0].dirs[1])
    fig, ax = plt.subplots(dpi=200)
    ax.set_title('Spectrum Evolution (dB scale)')
    f = (timeFreq.f[Nmin:Nmax]+center_freq_Hz)/1e12 
    z = zvals
    F, Z = np.meshgrid(f, z)
    Pf=getPower(matrix[:,Nmin:Nmax]  )/np.max(getPower(matrix[:,Nmin:Nmax]))
    Pf[Pf<1e-100]=1e-100
    Pf = 10*np.log10(Pf)
    Pf[Pf<dB_cutoff]=dB_cutoff
    surf=ax.contourf(F, Z, Pf,levels=40)
    ax.set_xlabel('Freq. [THz]')
    ax.set_ylabel('Distance [m]')
    cbar=fig.colorbar(surf, ax=ax) 
    saveplot('spectrum_evo_2D') 
    plt.show()
    os.chdir(ssfm_result_list[0].dirs[0])

def plotSpectrumMatrix3D(ssfm_result_list, nrange:int, dB_cutoff):
    """ 
    Plots spectrum calculated by SSFM as 3D colour surface
    
    3D colour plot of signal spectrum in freq domain throughout entire fiber span
    normalized to the highest peak power throughout. 
    
    Parameters:
        ssfm_result_list (list): List of ssmf_result_class objects corresponding to each fiber segment
        nrange (int): Determines how many points on either side of the center we wish to plot  
        dB_cutoff : Lowest y-value in plot is this many dB smaller than the peak power
        
    Returns:
    """    
    timeFreq = ssfm_result_list[0].input_signal.timeFreq   
    zvals = unpackZvals(ssfm_result_list)
    matrix = unpackMatrix(ssfm_result_list,zvals,timeFreq,"spectrum")
    
    Nmin = np.max([int(timeFreq.number_of_points/2-nrange),0])
    Nmax = np.min([int(timeFreq.number_of_points/2+nrange),timeFreq.number_of_points-1])     
    center_freq_Hz = timeFreq.centerFrequency


    #Plot pulse evolution in 3D
    os.chdir(ssfm_result_list[0].dirs[1])
    fig, ax = plt.subplots(1,1, figsize=(10,7),subplot_kw={"projection": "3d"})
    plt.title("Spectrum Evolution (dB scale)")
      
    f = (timeFreq.f[Nmin:Nmax]+center_freq_Hz)/1e12 
    z = zvals
    F_surf, Z_surf = np.meshgrid(f, z)
    P_surf=getPower(matrix[:,Nmin:Nmax]  )/np.max(getPower(matrix[:,Nmin:Nmax]))
    P_surf[P_surf<1e-100]=1e-100
    P_surf = 10*np.log10(P_surf)
    P_surf[P_surf<dB_cutoff]=dB_cutoff
    # Plot the surface.
    surf = ax.plot_surface(F_surf, Z_surf, P_surf, cmap=cm.viridis,
                          linewidth=0, antialiased=False)
    ax.set_xlabel('Freq. [GHz]')
    ax.set_ylabel('Distance [m]')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    saveplot('spectrum_evo_3D') 
    plt.show()
    os.chdir(ssfm_result_list[0].dirs[0])


def plotEverythingAboutSpectra(ssfm_result_list,
                               nrange:int, 
                               dB_cutoff):
    """ 
    Generates all plots of pulse amplitudes throughout fiber span
    
    Calls plotFirstAndLastSpectrum, plotSpectrumMatrix2D and plotSpectrumMatrix3D
    sequentially and saves the plots in the appropriate directory
    
    Parameters:
        ssfm_result_list (list): List of ssmf_result_class objects corresponding to each fiber segment
        nrange (int): Determines how many points on either side of the center we wish to plot  
        dB_cutoff : Lowest y-value in plot is this many dB smaller than the peak power
    
    Returns:

    
    """   

    print('  ')  
    plotFirstAndLastSpectrum(ssfm_result_list, nrange, dB_cutoff)
    plotSpectrumMatrix2D(ssfm_result_list, nrange, dB_cutoff)
    plotSpectrumMatrix3D(ssfm_result_list, nrange, dB_cutoff)
    print('  ')  

    




from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.legend import LineCollection
from matplotlib.colors import LinearSegmentedColormap

def makeChirpGif(ssfm_result_list,nrange:int,chirpRange=[-20,20],framerate=30):
    """ 
    Animates pulse evolution and shows local chirp 
    
    Animates pulse power evolution and shows local chirp by changing line color.
    Saves result as .gif file.
    Note: Producing the animation can take a several minutes on a regular 
    PC, so please be patient.
    
    Parameters:
        ssfm_result_list (list): List of ssmf_result_class objects corresponding to each fiber segment
        nrange (int): Determines how many points on either side of the center we wish to plot  
        chirpRange=[-20,20] (list) (optional): Min and Max frequency values in GHz to determine line color
        framerate=30 (int) (optional): Framerate of .gif animation. May want to reduce this number for simulations with few steps.     
        
    """      
    print("Making .gif anination of pulse evolution. This may take a while, so please be patient.")
    
    os.chdir(ssfm_result_list[0].dirs[1])
    
    print(f"The .gif animation will be saved in {os.getcwd()}")
    
    timeFreq = ssfm_result_list[0].input_signal.timeFreq   
    zvals = unpackZvals(ssfm_result_list)
    matrix = unpackMatrix(ssfm_result_list,zvals,timeFreq,"pulse")
    scalingFactor, letter =  getUnitsFromValue(np.max(zvals))
    
    Nmin = np.max([int(timeFreq.number_of_points/2-nrange),0])
    Nmax = np.min([int(timeFreq.number_of_points/2+nrange),timeFreq.number_of_points-1])    
    
    Tmin = timeFreq.t[Nmin]
    Tmax = timeFreq.t[Nmax]
    
    points = np.array( [timeFreq.t*1e12 ,  getPower(matrix[len(zvals)-1,Nmin:Nmax])   ] ,dtype=object ).T.reshape(-1,1,2)
    segments = np.concatenate([points[0:-1],points[1:]],axis=1)
    
    
    
    
    #Make custom colormap
    colors = ["red" ,"gray", "blue"]
    cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)
    
    #Initialize color normalization function
    norm = plt.Normalize(chirpRange[0],chirpRange[1])
    
    #Initialize line collection to be plotted
    lc=LineCollection(segments,cmap=cmap1,norm=norm)
    lc.set_array( getChirp(timeFreq.t[Nmin:Nmax],matrix[len(zvals)-1,Nmin:Nmax])/1e9 )
    
    #Initialize figure
    fig, ax = plt.subplots(dpi=150)
    line = ax.add_collection(lc)
    fig.colorbar(line,ax=ax, label = 'Chirp [GHz]')
    
    Pmax = np.max( np.abs(matrix) )**2
    

    
    
    #Function for specifying axes
    def init():
      
    
      ax.set_xlim([Tmin*1e12,Tmax*1e12])
      ax.set_ylim([0,1.05*Pmax])
      
      ax.set_xlabel('Time [ps]')
      ax.set_ylabel('Power [W]')
    #Function for updating the plot in the .gif
    def update(i):
      ax.clear() #Clear figure 
      init()     #Reset axes  {num:{1}.{5}}  np.round(,2)
      ax.set_title(f'Pulse evolution, z = {zvals[i]/scalingFactor:.2f}{letter}m')
      
      #Make collection of points from pulse power
      points = np.array( [timeFreq.t[Nmin:Nmax]*1e12 ,  getPower(matrix[i,Nmin:Nmax])   ] ,dtype=object ).T.reshape(-1,1,2)
      
      #Make collection of lines from points
      segments = np.concatenate([points[0:-1],points[1:]],axis=1)
      lc=LineCollection(segments,cmap=cmap1,norm=norm)
    
      #Activate norm function based on local chirp
      
      lc.set_array( getChirp(timeFreq.t[Nmin:Nmax],matrix[i,Nmin:Nmax])/1e9 )
      #Plot line
      line = ax.add_collection(lc)



    #Make animation
    ani = FuncAnimation(fig,update,range(len(zvals)),init_func=init)
    plt.show()
    
    #Save animation as .gif

    writer = PillowWriter(fps=framerate)
    ani.save(f'{ssfm_result_list[0].experimentName}_fps={framerate}.gif',writer=writer)

    os.chdir(ssfm_result_list[0].dirs[0])


def getAverageTimeOrFreq(time_or_freq,pulse_or_spectrum):
    """ 
    Computes central time (or frequency) of pulse (spectrum) 
    
    Computes central time (or frequency) of pulse (spectrum) by calculating
    the 'expectation value'. 
    
    Parameters:
        time_or_freq (nparray): Time range in seconds or freq. range in Hz
        pulse_or_spectrum (nparray): Temporal or spectral amplitude
        
    Returns:
        float: average time or frequency 
    
    """    
    E=getEnergy(time_or_freq,pulse_or_spectrum)    
    meanValue = np.trapz( time_or_freq*getPower(pulse_or_spectrum),time_or_freq )/E   
    return meanValue
    
def getVarianceTimeOrFreq(time_or_freq,pulse_or_spectrum):
    """ 
    Computes variance of pulse or spectrum 
    
    Computes variance of pulse in time domain or freq domain via
    <x**2>-<x>**2
    
    Parameters:
        time_or_freq (nparray): Time range in seconds or freq. range in Hz
        pulse_or_spectrum (nparray): Temporal or spectral amplitude
        
    Returns:
        float: variance in time or frequency domains 
    
    """ 
    E=getEnergy(time_or_freq,pulse_or_spectrum)
    variance = np.trapz( time_or_freq**2*getPower(pulse_or_spectrum),time_or_freq )/E  - (getAverageTimeOrFreq(time_or_freq,pulse_or_spectrum))**2
    return variance   
    
def getStDevTimeOrFreq(time_or_freq,pulse_or_spectrum):
    """ 
    Computes standard deviation of pulse or spectrum 
    
    Computes std of pulse in time domain or freq domain via
    sqrt(<x**2>-<x>**2)
    
    Parameters:
        time_or_freq (nparray): Time range in seconds or freq. range in Hz
        pulse_or_spectrum (nparray): Temporal or spectral amplitude
        
    Returns:
        float: Stdev in time or frequency domains 
    
    """ 
    return np.sqrt(getVarianceTimeOrFreq(time_or_freq,pulse_or_spectrum))


def plotAverageAndStdTimeAndFreq(ssfm_result_list):
    """ 
    Plots how spectral and temporal width of signal change with distance
    
    Uses getAverageTimeOrFreq and getStDevTimeOrFreq to create dual-axis 
    line plot of temporal and spectral center and widths throughout fiber span.
    Saves plot in appropriate folder.
    
    Parameters:
        ssfm_result_list (list): List of ssmf_result_class objects corresponding to each fiber segment
        
    Returns:
    
    """    
    timeFreq = ssfm_result_list[0].input_signal.timeFreq   
    center_freq_Hz = timeFreq.centerFrequency
    zvals = unpackZvals(ssfm_result_list)
    
    pulseMatrix = unpackMatrix(ssfm_result_list,zvals,timeFreq,"pulse")
    spectrumMatrix = unpackMatrix(ssfm_result_list,zvals,timeFreq,"spectrum")
    
    meanTimeArray = np.zeros( len(zvals) )*1.0
    meanFreqArray = np.copy( meanTimeArray )
    stdTimeArray  = np.copy( meanTimeArray )
    stdFreqArray  = np.copy( meanTimeArray )
    
    
    i = 0
    for pulse, spectrum in zip(pulseMatrix,spectrumMatrix):
        
        meanTimeArray[i] = getAverageTimeOrFreq(timeFreq.t,pulse)
        meanFreqArray[i] = getAverageTimeOrFreq(timeFreq.f,spectrum)
        
        stdTimeArray[i]  = getStDevTimeOrFreq(timeFreq.t,pulse)
        stdFreqArray[i]  = getStDevTimeOrFreq(timeFreq.f,spectrum)
        
        
        i+=1

    scalingFactor_Z,prefix_Z=getUnitsFromValue(np.max(zvals))
    maxCenterTime = np.max( np.abs(meanTimeArray)  )
    maxStdTime    = np.max(stdTimeArray)

    
    scalingFactor_pulse,prefix_pulse=getUnitsFromValue( np.max(  [ maxCenterTime,maxStdTime] ) )
    scalingFactor_spectrum,prefix_spectrum=getUnitsFromValue( np.max(  [ meanFreqArray,stdFreqArray] ) )


    os.chdir(ssfm_result_list[0].dirs[1])
    fig,ax = plt.subplots(dpi=200)
    plt.title("Evolution of temporal/spectral widths and centers")
    ax.plot(zvals/scalingFactor_Z,meanTimeArray/scalingFactor_pulse, 'C0-',label = "Pulse Center")
    ax.plot(zvals/scalingFactor_Z,stdTimeArray/scalingFactor_pulse, 'C0--',label =  "Pulse Width")
    ax.set_xlabel(f'Distance [{prefix_Z}m]')
    ax.set_ylabel(f'Time [{prefix_pulse}s]',color = 'C0')
    ax.tick_params(axis='y',labelcolor='C0')
    
    ax2=ax.twinx()
    ax2.plot(zvals/scalingFactor_Z,meanFreqArray/scalingFactor_spectrum,'C1-', label= f"Spectrum Center rel. to $f_c$={center_freq_Hz/1e12:.5}THz ")
    ax2.plot(zvals/scalingFactor_Z,stdFreqArray/scalingFactor_spectrum,'C1--',  label= "Spectrum Width")
    
    ax2.set_ylabel(f'Freq. [{prefix_spectrum}Hz]',color = 'C1')
    ax2.tick_params(axis='y',labelcolor='C1')
    fig.legend(bbox_to_anchor=(1.55,0.8))
    
    saveplot('Width_evo') 
    plt.show()
    os.chdir(ssfm_result_list[0].dirs[0])
    

def plotEverythingAboutResult(ssfm_result_list, 
                              nrange_pulse, 
                              dB_cutoff_pulse, 
                              nrange_spectrum, 
                              dB_cutoff_spectrum,
                              skip_3D_plot_flag = False,
                              skip_chirp_plot_flag = False,
                              **kwargs):
    """ 
    Generates all plots of pulse amplitudes, spectra etc. throughout fiber span
    
    Calls   plotAverageAndStdTimeAndFreq, plotEverythingAboutPulses and 
    plotEverythingAboutSpectra sequentially, saving plots in the appropriate directory
    
    Parameters:
        ssfm_result_list (list): List of ssmf_result_class objects corresponding to each fiber segment
        nrange_pulse (int): For pulse plots, determines how many points on either side of the center we wish to plot  
        dB_cutoff_pulse   : For pulse plots, lowest y-value in plot is this many dB smaller than the peak power
        nrange_spectrum (int): For spectrum plots, determines how many points on either side of the center we wish to plot  
        dB_cutoff_spectrum   : For spectrum plots, lowest y-value in plot is this many dB smaller than the peak power
        **kwargs (optional):     
    
    Returns:

    """  
    plotAverageAndStdTimeAndFreq(ssfm_result_list)
    
    plotEverythingAboutPulses(ssfm_result_list, 
                                  nrange_pulse, 
                                  dB_cutoff_pulse, **kwargs)
    
    plotEverythingAboutSpectra(ssfm_result_list,
                                   nrange_spectrum, 
                                   dB_cutoff_spectrum)






from scipy import signal

def waveletTest(M,s):
    
    w=1
    x = np.arange(0, M) - (M - 1.0) / 2
    x = x / s
    wavelet = np.exp(1j * w * x) * np.exp(-0.5 * x**2) * np.pi**(-0.25)
    output = np.sqrt(1/s) * wavelet
    return output  

def waveletTransform(timeFreq:timeFreq_class,
                     pulse, 
                     nrange_pulse,
                     nrange_spectrum,
                     dB_cutoff ):
    
    Nmin_pulse = np.max([int(timeFreq.number_of_points/2-nrange_pulse),0])
    Nmax_pulse = np.min([int(timeFreq.number_of_points/2+nrange_pulse),timeFreq.number_of_points-1])    
    

    Tmax = timeFreq.t[Nmax_pulse]
    
   

    
    t = timeFreq.t[Nmin_pulse:Nmax_pulse]

    wavelet_durations = np.linspace((t[1]-t[0])*10,Tmax,1000)
    
    print((t[1]-t[0])*100,Tmax)
    print(1/Tmax/1e9,1/((t[1]-t[0])*100)/1e9)
    
    dt_wavelet = wavelet_durations[1]-wavelet_durations[0]
    
    
    plt.figure()
    plt.plot(t,np.real(pulse[Nmin_pulse:Nmax_pulse]))
    plt.plot(t,np.imag(pulse[Nmin_pulse:Nmax_pulse]))
    plt.show()
    
            
    plt.figure()
    plt.plot(t,getChirp(t, pulse[Nmin_pulse:Nmax_pulse])/1e9)
    plt.ylabel('Chirp [GHz]')
    plt.show()
    
    cwtmatr = signal.cwt(pulse[Nmin_pulse:Nmax_pulse], signal.morlet2, wavelet_durations,dtype=complex)
    

    
    
    
      
    Z = np.abs(cwtmatr)**2
    print(np.max(Z))
    Z /= np.max(Z)
    
    Z[Z<10**(dB_cutoff/10)] = 10**(dB_cutoff/10)
    

    fig, ax = plt.subplots(dpi=200)
    ax.set_title('Wavelet transform of final pulse')
    T, F = np.meshgrid(t, 1/wavelet_durations)
    
    surf=ax.contourf(T/1e-12,F/1e9, Z,levels=40)
    ax.set_xlabel('Time. [ps]')
    ax.set_ylabel('Freq. [GHz]')
    cbar=fig.colorbar(surf, ax=ax) 
    saveplot('wavelet_final') 
    plt.show()
    
    
    
    


def wavelengthToFreq(wavelength_m):
    """ 
    Converts wavelength in m to frequency in Hz
    
    Converts wavelength in m to frequency in Hz using f=c/lambda
    
    Parameters:
        wavelength_m (float): Wavelength in m
        
    Returns:
        float: Frequency in Hz 
    """     
    return c/wavelength_m

def freqToWavelength(freq_Hz):
    """ 
    Converts frequency in Hz to wavelength in m
    
    Converts frequency in Hz to wavelength in m using lambda=c/f
    
    Parameters:
        freq_Hz (float): Frequency in Hz
        
    Returns:
        float: Wavelength in m
    """  
    return c/freq_Hz

def wavelengthBWtoFreqBW(wavelength_m,wavelengthBW_m):
    """ 
    Converts bandwidth in m to bandwidth in Hz
    
    A signal centered at lambda_0 with bandwidth specified in terms of wavelength
    will have a frequency bandwidth of c*lambda_BW/lambda**2
    
    Parameters:
        wavelength_m   (float): Wavelength in m
        wavelengthBW_m (float): Wavelength bandwidth in m
        
    Returns:
        float: Frequency bandwidth in Hz
    """  
    return c*wavelengthBW_m/wavelength_m**2

def freqBWtoWavelengthBW(freq_Hz,freqBW_Hz):
    """ 
    Converts bandwidth in Hz to bandwidth in m
    
    A signal centered at f_0 with bandwidth specified in terms of frequency
    will have a wavelength bandwidth of c*freq_BW/freq**2
    
    Parameters:
        freq_Hz   (float): Frequency in Hz
        freqBW_Hz (float): Frequency bandwidth in Hz
        
    Returns:
        float: Wavelength bandwidth in m
    """  
    return c*freqBW_Hz/freq_Hz**2


def getGammaFromFiberParams(wavelength_m,n2_m2_W,coreDiameter_m):
    return 2*pi/wavelength_m*n2_m2_W/( pi* coreDiameter_m**2/4  )
  

if __name__ == "__main__":
    
    os.chdir(os.path.realpath(os.path.dirname(__file__)))
    
    
    N  = 2**16 #Number of points
    dt = 1e-15 #Time resolution [s] 
    
    centerWavelength = 426.9e-9 #laser wl in m  
    centerFreq_test=wavelengthToFreq(centerWavelength)
    
    timeFreq_test=timeFreq_class(N,dt,centerFreq_test)
    
    beta_list = [1e-30] #Dispersion in units of s^(entry+2)/m    
    #beta_list = [0,0,1e-12*1e-24*1e-11] 
    
    fiber_diameter = 9e-6 #m
    n2_silica=2.2e-20 #m**2/W
    
    gamma_test = getGammaFromFiberParams(centerWavelength,n2_silica,fiber_diameter)
    
    #  Initialize fibers
    alpha_test = 0.0
    
    numberOfSteps_test = 2**8 
    
    fiber_test = fiber_class(1000, numberOfSteps_test, gamma_test,   beta_list,    alpha_test  )
   
    
    
    fiber_list = [fiber_test,fiber_test]
    fiber_span = fiber_span_class(fiber_list)
    
    
    
    
    testAmplitude = np.sqrt(1)                    #Amplitude in units of sqrt(W)
    testDuration  =1e-12   #Pulse 1/e^2 duration [s]
    testTimeOffset    = 0                       #Time offset
    testFreqOffset    = 0                       #Time offset
    
    testChirp = 0
    testPulseType='gaussian' 
    testOrder = 1
    testNoiseAmplitude = 0
    

    testInputSignal = input_signal_class(timeFreq_test, 
                                          testAmplitude ,
                                          testDuration,
                                          testTimeOffset,
                                          testFreqOffset,                                        
                                          testChirp,
                                          testPulseType,
                                          testOrder,
                                          testNoiseAmplitude)
    
    #testInputSignal.amplitude = np.cos(2*pi*5e9*testInputSignal.timeFreq.t)*testInputSignal.amplitude
    #testInputSignal.spectrum  = getSpectrumFromPulse(testInputSignal.timeFreq.t, testInputSignal.amplitude) 
      
    
    
    

       





    expName="Raman_development"
    #Run SSFM
    ssfm_result_list = SSFM(fiber_span,
                            testInputSignal,
                            showProgressFlag=True,
                            experimentName=expName)
    
    
    
    #Plot pulses
    nrange_test_pulse=1000
    cutoff_test_pulse=-60

    #Plot pulses
    nrange_test_spectrum=int(1000)
    cutoff_test_spectrum=-60
    
    #plotFirstAndLastPulse(ssfm_result_list,nrange_test_pulse, cutoff_test_pulse)
    #plotFirstAndLastSpectrum(ssfm_result_list, nrange_test_spectrum,cutoff_test_spectrum)
    
    # plotEverythingAboutSpectra(ssfm_result_list,
    #                                 nrange_test_spectrum, 
    #                                 cutoff_test_spectrum)
    
    
    plotEverythingAboutResult(ssfm_result_list,
                              nrange_test_pulse,
                              cutoff_test_pulse,
                              nrange_test_spectrum,
                              cutoff_test_spectrum,
                              )
    
    #makeChirpGif(ssfm_result_list,nrange_test_pulse,chirpRange=[-20,20],framerate=30)
    
    # waveletTransform(ssfm_result_list[0].input_signal.timeFreq,
    #                   ssfm_result_list[0].pulseMatrix[0,:], 
    #                   nrange_test_pulse,
    #                   nrange_test_spectrum,
    #                   cutoff_test_pulse )
    
    