# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 10:23:48 2022

@author: okrarup
"""

import numpy as np
from scipy.fftpack import fft, ifft, fftshift, ifftshift, fftfreq


import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import cm

import os

from datetime import datetime


global pi; pi=np.pi 

#Define frequency range for spectrum based on time basis. 
def getFreqRangeFromTime(time):
    return fftshift(fftfreq(len(time), d=time[1]-time[0]))

#Calcualte phase by getting the complex angle of the pulse, unwrapping it and centering on middle entry.
def getPhase(pulse):
    phi=np.unwrap(np.angle(pulse)) #Get phase starting from 1st entry
    phi=phi-phi[int(len(phi)/2)]   #Center phase on middle entry
    return phi    

#Calculate local chirp as the (negative) derivative of phase w.r.t. time. 
def getChirp(time,pulse):
    phi=getPhase(pulse)
    dphi=np.diff(phi ,prepend = phi[0] - (phi[1]  - phi[0]  ),axis=0) #Change in phase. Prepend to ensure consistent array size 
    dt  =np.diff(time,prepend = time[0]- (time[1] - time[0] ),axis=0) #Change in time.  Prepend to ensure consistent array size

    return -1.0/(2*pi)*dphi/dt #Chirp = - 1/(2pi) * d(phi)/dt
    
    

#Class for storing info about the time axis and frequency axis. 
class timeFreq_class:
    def __init__(self,N,dt):
        self.number_of_points=N
        self.time_step=dt
        t=np.linspace(0,N*dt,N)
        self.t=t-np.mean(t)
        self.tmin=self.t[0]
        self.tmax=self.t[-1]
        
        self.f=getFreqRangeFromTime(self.t)
        self.fmin=self.f[0]
        self.fmax=self.f[-1]
        self.freq_step=self.f[1]-self.f[0]

        self.describe_config()
        
    def describe_config(self,destination = None):
        print(" ### timeFreq Configuration Parameters ###" , file = destination)
        print(f"  Number of points \t\t= {self.number_of_points}", file = destination)
        print(f"  Start time, tmin \t\t= {self.tmin*1e12:.3f}ps", file = destination)
        print(f"  Stop time, tmax \t\t= {self.tmax*1e12:.3f}ps", file = destination)
        print(f"  Time resolution \t\t= {self.time_step*1e12:.3f}ps", file = destination)
        print("  ", file = destination)
        print(f"  Start frequency\t\t= {self.fmin/1e12:.3f}THz", file = destination)
        print(f"  Stop frequency \t\t= {self.fmax/1e12:.3f}THz", file = destination)
        print(f"  Frequency resolution \t\t= {self.freq_step/1e6:.3f}MHz", file = destination)
        print( "   ", file = destination)
        

    def saveTimeFreq(self):
        timeFreq_df = pd.DataFrame(columns=['number_of_points', 'dt_s',])

        timeFreq_df.loc[  len(timeFreq_df.index) ] = [self.number_of_points,
                                                  self.time_step]
        
        timeFreq_df.to_csv("timeFreq.csv")  
        
    
#Loads timeFreq from previous run
def load_timeFreq(path:str): 
    df = pd.read_csv(path+'\\timeFreq.csv')
    number_of_points = df['number_of_points']
    dt_s = df['dt_s']
    
    return timeFreq_class(int(number_of_points[0]), dt_s[0])
    

        

#Function returns pulse power or spectrum PSD
def getPower(amplitude):
    return np.abs(amplitude)**2  

#Function gets the energy of a pulse pulse or spectrum by integrating the power
def getEnergy(time_or_frequency,amplitude):
    return np.trapz(getPower(amplitude),time_or_frequency)

#Gaussian pulse
def GaussianPulse(time,amplitude,duration,offset,chirp,order,carrier_freq_Hz):
    assert 1 <= order, f"Error: Order of gaussian pulse is {order}. Must be >=1"
    return amplitude*np.exp(- (1+1j*chirp)/2*((time-offset)/(duration))**(2*np.floor(order)))*np.exp(-1j*2*pi*carrier_freq_Hz*time)

#Square pulse approximated with high-order super-Gaussian
def squarePulse(time,amplitude,duration,offset,chirp,carrier_freq_Hz):
    return GaussianPulse(time,amplitude,duration,offset,chirp,100,carrier_freq_Hz)


#Sech pulse
def sechPulse(time,amplitude,duration,offset,chirp,carrier_freq_Hz):
    return amplitude/np.cosh((time-offset)/duration)*np.exp(- (1j*chirp)/2*((time-offset)/(duration))**2)*np.exp(-1j*2*pi*carrier_freq_Hz*time)


#Function for adding white noise
def noise_ASE(time,amplitude):
    randomAmplitudes=np.random.normal(loc=0.0, scale=amplitude, size=len(time))*(1+0j)
    randomPhases = np.random.uniform(-pi,pi, len(time))
    return randomAmplitudes*np.exp(1j*randomPhases)   


#Returns a pulse of certain type based on specified parameters. 
def getPulse(time,amplitude,duration,offset,chirp,carrier_freq_Hz,pulseType,order,noiseAmplitude):
    
    noise = noise_ASE(time,noiseAmplitude)
    
    if pulseType.lower()=="gaussian":
        return GaussianPulse(time,amplitude,duration,offset,chirp,order,carrier_freq_Hz)+noise
    
    if pulseType.lower()=="sech":
        return sechPulse(time,amplitude,duration,offset,chirp,carrier_freq_Hz)+noise
    
    if pulseType.lower()=="square":
        return squarePulse(time,amplitude,duration,offset,chirp,carrier_freq_Hz)+noise
    
    if pulseType.lower()=="custom":
        return noise


#Computes FFT of pulse to get its spectrum and verifies that energy is conserved. 
def getSpectrumFromPulse(time,pulse_amplitude):
    pulseEnergy=getEnergy(time,pulse_amplitude) #Get pulse energy
    f=getFreqRangeFromTime(time) 
    dt=time[1]-time[0]
    
    spectrum_amplitude=fftshift(fft(pulse_amplitude))*dt #Take FFT and do shift
    spectrumEnergy=getEnergy(f, spectrum_amplitude) #Get spectrum energy
    
    err=np.abs((pulseEnergy/spectrumEnergy-1))
    
    assert( err<1e-7 ), f'ERROR = {err}: Energy changed when going from Pulse to Spectrum!!!' 
    
    return spectrum_amplitude



#Function for getting time base from frequency range
def getTimeFromFrequency(frequency):  
    return fftshift(fftfreq(len(frequency), d=frequency[1]-frequency[0]))


#Compute iFFT of spectrum to get pulse and verifies that energy is conserved
def getPulseFromSpectrum(frequency,spectrum_amplitude):
    
    spectrumEnergy=getEnergy(frequency, spectrum_amplitude)
    
    time = getTimeFromFrequency(frequency)
    dt = time[1]-time[0]
     
    pulse = ifft(ifftshift(spectrum_amplitude))/dt
    pulseEnergy = getEnergy(time, pulse)
    
    err=np.abs((pulseEnergy/spectrumEnergy-1))

    assert( err<1e-7   ), f'ERROR = {err}: Energy changed when going from Spectrum to Pulse!!!' 
    
    return pulse

#Generates a Gaussian spectrum
def GaussianSpectrum(frequency,amplitude,bandwidth,carrier_freq_Hz):
    time = getTimeFromFrequency(frequency)
    return getSpectrumFromPulse(time, GaussianPulse(time, amplitude, 1/bandwidth, 0,0,1,carrier_freq_Hz))


#Class for holding info about individual fibers
class fiber_class:
    def __init__(self,L,gamma,beta2,alpha_dB_per_m):
      self.Length=L
      self.gamma=gamma
      self.beta2=beta2
      self.alpha_dB_per_m=alpha_dB_per_m
      self.alpha_Np_per_m = self.alpha_dB_per_m*np.log(10)/10.0 #Loss coeff is usually specified in dB/km, but Nepers/km is more useful for calculations
      self.total_loss =  alpha_dB_per_m*self.Length
      #TODO: Make alpha frequency dependent.  
      self.describe_fiber()
      
    def describe_fiber(self,destination = None):
        print(' ### Characteristic parameters of fiber: ###', file = destination)
        print(f'Fiber Length [km] \t= {self.Length/1e3} ', file = destination)
        print(f'Fiber gamma [1/W/m] \t= {self.gamma} ', file = destination)
        print(f'Fiber beta2 [s^2/m] \t= {self.beta2} ', file = destination)
        print(f'Fiber alpha_dB_per_m \t= {self.alpha_dB_per_m} ', file = destination)
        print(f'Fiber alpha_Np_per_m \t= {self.alpha_Np_per_m} ', file = destination)
        print(f'Fiber total loss [dB] \t= {self.total_loss} ', file = destination)
        print(' ', file = destination)


#Class for holding info about span of concatenated fibers. 
class fiber_span_class:
    
    def __init__(self,fiber_list):
        self.fiber_list=fiber_list
        self.number_of_fibers_in_span=len(fiber_list)
        
    def saveFiberSpan(self):
        
        fiber_df = pd.DataFrame(columns=['Length_m', 
                                         'gamma_per_W_per_m',
                                         'beta2_s2_per_m',
                                         'alpha_dB_per_m',
                                         'alpha_Np_per_m'])
                                         
        for fiber in self.fiber_list:
            fiber_df.loc[  len(fiber_df.index) ] = [fiber.Length,fiber.gamma,fiber.beta2,fiber.alpha_dB_per_m,fiber.alpha_Np_per_m ]
        
        fiber_df.to_csv("Fiber_span.csv")


#Load fiber span from previous run
def load_fiber_span(path:str):
    
    df = pd.read_csv(path+'\\Fiber_span.csv')
    Length_m = df['Length_m']
    gamma_per_W_per_m = df['gamma_per_W_per_m']
    beta2_s2_per_m = df['beta2_s2_per_m']
    alpha_dB_per_m = df['alpha_dB_per_m']
    
    
    fiber_list=[]
    
    for i in range(len(Length_m)):   
        fiber_list.append( fiber_class(  Length_m[i], gamma_per_W_per_m[i], beta2_s2_per_m[i], alpha_dB_per_m[i] ) )    
    
    return fiber_span_class(fiber_list)


#Class for holding input signal sent into fiber. 
class input_signal_class:
    def __init__(self,timeFreq:timeFreq_class,peak_amplitude,duration,offset,chirp,carrier_freq_Hz,pulseType,order,noiseAmplitude):


        self.Amax = peak_amplitude
        self.Pmax= self.Amax**2
        self.duration=duration
        self.offset=offset
        self.chirp=chirp
        self.carrier_freq_Hz=carrier_freq_Hz
        self.pulseType=pulseType
        self.order=order
        self.noiseAmplitude=noiseAmplitude        

        self.timeFreq=timeFreq
        self.number_of_points=timeFreq.number_of_points
        self.dt=timeFreq.time_step
        
        
        self.amplitude = getPulse(self.timeFreq.t,peak_amplitude,duration,offset,chirp,carrier_freq_Hz,pulseType,order,noiseAmplitude)
        

        if getEnergy(self.timeFreq.t, self.amplitude) == 0.0:
            self.spectrum = np.copy(self.amplitude)  
        else:
            self.spectrum = getSpectrumFromPulse(self.timeFreq.t,self.amplitude)   
        
        
        
        self.describe_input_signal()
        
    def describe_input_signal(self,destination = None):
        print(" ### Input Signal Parameters ###" , file = destination)
        print(f"  Pmax \t\t\t\t= {self.Pmax:.3f} W", file = destination)
        print(f"  Duration \t\t\t= {self.duration*1e12:.3f} ps", file = destination)
        print(f"  Offset \t\t\t= {self.offset*1e12:.3f} ps", file = destination)
        print(f"  Chirp \t\t\t= {self.chirp:.3f}", file = destination)
        print(f"  Carrier_freq \t\t\t= {self.carrier_freq_Hz/1e12} THz", file = destination)
        print(f"  pulseType \t\t\t= {self.pulseType}", file = destination)
        print(f"  order \t\t\t= {self.order}", file = destination)
        print(f"  noiseAmplitude \t\t= {self.noiseAmplitude:.3f} sqrt(W)", file = destination)
        
        print( "   ", file = destination)

    def saveInputSignal(self):
 
        #Initialize dataframe
        signal_df = pd.DataFrame(columns=['Amax_sqrt(W)',
                                          'Pmax_W',
                                          'duration_s',
                                          'offset_s',
                                          'chirp',
                                          'carrier_freq_Hz',
                                          'pulseType',
                                          'order',
                                          'noiseAmplitude_sqrt(W)'])
                                         
         
        
        #Fill it with values used for generating input signal
        signal_df.loc[  len(signal_df.index) ] = [self.Amax,
                                                  self.Pmax,
                                                  self.duration,
                                                  self.offset,
                                                  self.chirp,
                                                  self.carrier_freq_Hz,
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
      

#Load input signal used in previous run        
def load_InputSignal(path):    

    #Open dataframe with pulse parameters
    df = pd.read_csv(path+'\\Input_signal.csv')
    
    Amax_sqrt_W             = df['Amax_sqrt(W)'][0]
    duration_s              = df['duration_s'][0]
    offset_s                = df['offset_s'][0]
    chirp                   = df['chirp'][0]
    carrier_freq_Hz         = df['carrier_freq_Hz'][0]
    pulseType               = df['pulseType'][0]
    order                   = int(df['order'][0])
    noiseAmplitude_sqrt_W   = df['noiseAmplitude_sqrt(W)'][0]
    
    #Load timeFreq 
    timeFreq = load_timeFreq( path )
    
    #Initialize class for loaded signal
    loaded_input_signal = input_signal_class(timeFreq,Amax_sqrt_W,duration_s,offset_s,chirp,carrier_freq_Hz,pulseType,order, noiseAmplitude_sqrt_W)
    
    #If signal type is "custom", load the raw amplitude values
    if pulseType == "custom":
        df_custom = pd.read_csv( path + '\\Custom_input_signal.csv' )
        
        A_real = np.array(df_custom["amplitude_sqrt_W_real"])
        A_imag = np.array(df_custom["amplitude_sqrt_W_imag"])
        A = A_real+1j*A_imag
        
        loaded_input_signal.amplitude = A
        
    
    return loaded_input_signal  


#Decide which approach to use for computing z-steps. 
def zstep_NL(z,fiber:fiber_class, input_signal:input_signal_class,stepmode,stepSafetyFactor):
    
    #TODO: Do we also need to check if alpha = 0?
    if fiber.gamma == 0.0:  
        return fiber.Length
    
    if fiber.beta2 == 0.0:
        return fiber.Length
    
    
    
    if stepmode.lower()=="cautious":
        return np.abs(fiber.beta2)*pi/(fiber.gamma*input_signal.Pmax*input_signal.duration)**2*np.exp(2*fiber.alpha_Np_per_m*z)/stepSafetyFactor
    
    if stepmode.lower()=="approx":
        return np.abs(fiber.beta2)*pi/(fiber.gamma*input_signal.Pmax)**2/(input_signal.duration*input_signal.timeFreq.time_step)*np.exp(2*fiber.alpha_Np_per_m*z)/stepSafetyFactor    


    else:
        return 1.0



#Function computes z-steps if we use a variable method (larger steps when power is low)
def getVariableZsteps( fiber:fiber_class, input_signal:input_signal_class,stepmode,stepSafetyFactor):    
    
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

#Compute z-steps taken by the simulation
def getZsteps(fiber:fiber_class,input_signal:input_signal_class,stepConfig_list,fiber_index=""):
    
    current_dir=os.getcwd()+'\\'
    
    newFolderName = "Z-step-graphs\\"
    zStepFolder = current_dir + newFolderName
    
    os.makedirs(zStepFolder,exist_ok=True)
    os.chdir(zStepFolder)
    
    
    stepMode=stepConfig_list[0]
    stepApproach=stepConfig_list[1]       
    stepSafetyFactor=stepConfig_list[2]
    
    zinfo = (np.array([0,fiber.Length]),fiber.Length)
    


    if stepMode.lower() == "fixed":
        
        if type(stepApproach) == str:
        
            dz=zstep_NL(0,fiber, input_signal,stepApproach,stepSafetyFactor)
            z_array=np.arange(0,fiber.Length,dz)
            
            if z_array[-1] != fiber.Length:
                z_array=np.append(z_array,fiber.Length)
            
            dz_array = np.diff( z_array)
            
            
        else:
            stepApproach=int(stepApproach)
            z_array=np.linspace(0,fiber.Length,stepApproach+1)
            dz_array=np.ones( stepApproach)*(z_array[1]-z_array[0])            

            
        zinfo   =(z_array,dz_array)
        
        
        
        
    else:
        zinfo = getVariableZsteps(fiber,input_signal,stepApproach,stepSafetyFactor)
        
    fig,ax = plt.subplots(dpi=200)
    ax.set_title(f"Fiber number = {fiber_index}, \n Stepmode = ({stepConfig_list[0]},{stepConfig_list[1]}), stepSafetyFactor = {stepConfig_list[2]}")
    ax.plot(zinfo[0]/1e3,'b.',label = f"z-locs ({len(zinfo[0])})")

    ax.set_xlabel('Entry')
    ax.set_ylabel('z-location [km]')
    ax.tick_params(axis='y',labelcolor='b')
    
    ax2=ax.twinx()
    ax2.plot(zinfo[1]/1e3,'r.',label = f"$\Delta$z-steps ({len(zinfo[1])})")
    ax2.set_ylabel('$\Delta$z [km]')
    ax2.tick_params(axis='y',labelcolor='r')
    
    fig.legend(bbox_to_anchor=(1.3,0.8))
    
    plt.savefig(f'Z-step_chart_{fiber_index}.png', 
                bbox_inches ="tight",
                pad_inches = 1,
                orientation ='landscape')
    plt.show()
    
    
    os.chdir(current_dir)
    
    return zinfo
   


        
#Class for holding result of SSFM simulation
class ssfm_result_class:
    def __init__(self, input_signal:input_signal_class, fiber:fiber_class,stepConfig,zinfo,experimentName,directories):
        
        self.input_signal = input_signal
        self.fiber = fiber
        self.stepConfig = stepConfig
        self.zinfo = zinfo
        self.experimentName=experimentName
        self.dirs = directories

        self.pulseMatrix = np.zeros((len(self.zinfo[0]),input_signal.timeFreq.number_of_points ) )*(1+0j)
        self.spectrumMatrix = np.copy(self.pulseMatrix)
        
        self.pulseMatrix[0,:]=np.copy(input_signal.amplitude)   
        self.spectrumMatrix[0,:] = np.copy(input_signal.spectrum)
        
        

#Helper function for getting SI prefix (k, M, G, T, etc.) and scaling factor when plotting and reporting results
def getUnitsFromValue(value):
    
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

    

#Print and plot different characteristic lengths (L_eff, L_D, L_NL etc.)
def describe_sim_parameters(fiber:fiber_class,input_signal:input_signal_class,zinfo,fiber_index,destination=None):    
    
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

    
    if fiber.beta2 != 0.0:
        Length_disp = input_signal.duration**2/np.abs(fiber.beta2)
    else:
        Length_disp=1e100
    print(f"  Length_disp \t= {Length_disp/scalingfactor:.2e} {prefix}m", file = destination)  
    
    length_list=np.append(length_list,Length_disp)
    
    if destination != None:
        ax.barh("Dispersion Length",Length_disp/scalingfactor, color ='C2')
    
    
    if fiber.gamma !=0.0:
        Length_NL = 1/fiber.gamma/input_signal.Pmax   
        N_soliton=np.sqrt(Length_disp/Length_NL)
    else:
        Length_NL=1e100
        N_soliton=np.NaN
    
    length_list=np.append(length_list,Length_NL)
    
    if destination != None:
        ax.barh("Nonlinear Length",Length_NL/scalingfactor, color ='C3')
    
    print(f"  Length_NL \t= {Length_NL/scalingfactor:.2e} {prefix}m", file = destination)
    print(f"  N_soliton \t= {N_soliton:.2e}", file = destination)
    print(f"  N_soliton^2 \t= {N_soliton**2:.2e}", file = destination)


    if fiber.beta2<0:
        
        z_soliton = pi/2*Length_disp
        length_list=np.append(length_list,z_soliton)
        if destination != None:
            ax.barh("Soliton Length",z_soliton/scalingfactor, color ='C4')
        
        print(' ', file = destination)
        print(f'  sign(beta2) \t= {np.sign(fiber.beta2)}, so Solitons and Modulation Instability may occur ', file = destination)
        print(f"   z_soliton \t= {z_soliton/scalingfactor:.2e} {prefix}m", file = destination)
        print(f"   N_soliton \t= {N_soliton:.2e}", file = destination)
        print(f"   N_soliton^2 \t= {N_soliton**2:.2e}", file = destination)
        

        print(" ", file = destination)
        
        # https://prefetch.eu/know/concept/modulational-instability/
        f_MI=np.sqrt(2*fiber.gamma*input_signal.Pmax/np.abs(fiber.beta2))/2/np.pi    
        gain_MI=2*fiber.gamma*input_signal.Pmax
        print(f"   Freq. w. max MI gain = {f_MI/1e9:.2e}GHz", file = destination)
        print(f"   Max MI gain \t\t= {gain_MI*scalingfactor:.2e} /{prefix}m ", file = destination)
        print(f"   Min MI gain distance = {1/(gain_MI*scalingfactor):.2e} {prefix}m ", file = destination)
        print(' ', file = destination)
        length_list=np.append(length_list,1/gain_MI)
        if destination != None:
            ax.barh("MI gain Length",1/(gain_MI*scalingfactor), color ='C5')
        
    elif fiber.beta2>0:           
        #https://prefetch.eu/know/concept/optical-wave-breaking/
        Nmin_OWB = np.exp(3/4)/2 #Minimum N-value of Optical Wave breaking with Gaussian pulse
        
        N_ratio = N_soliton/Nmin_OWB
        if N_ratio<=1:
            Length_wave_break = 1e100
        else:
            Length_wave_break = Length_disp/np.sqrt(N_soliton**2/Nmin_OWB**2-1)  #Characteristic length for Optical Wave breaking with Gaussian pulse
        length_list=np.append(length_list,Length_wave_break)
        print(' ', file = destination)
        print(f'   sign(beta2) \t\t\t\t= {np.sign(fiber.beta2)}, so Optical Wave Breaking may occur ', file = destination)
        print( "   Nmin_OWB (cst.) \t\t\t= 0.5*exp(3/4) (assuming Gaussian pulses)", file = destination)
        print(f"   N_ratio = N_soliton/Nmin_OWB \t= {N_ratio:.2e}", file = destination)
        print(f"   Length_wave_break \t\t\t= {Length_wave_break/scalingfactor:.2e} {prefix}m", file = destination)    
    
        if destination != None:
            ax.barh("OWB Length",Length_wave_break/scalingfactor, color ='C6')
    
    if destination != None:
        ax.barh("Maximum $\Delta$z",np.max(zinfo[1])/scalingfactor, color ='C7')
        ax.barh("Minimum $\Delta$z",np.min(zinfo[1])/scalingfactor, color ='C8')
        length_list=np.append(length_list,np.max(zinfo[1]))
        length_list=np.append(length_list,np.min(zinfo[1]))
            
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

   

#Print results to file and to console
def describe_run( current_time, current_fiber:fiber_class,  current_input_signal:input_signal_class,current_stepConfig, zinfo,fiber_index=""  ,destination = None):
  
    print("Info about fiber",file = destination )
    current_fiber.describe_fiber(destination = destination)
    print(' ', file = destination)
    
    
    describe_sim_parameters(current_fiber,current_input_signal,zinfo,fiber_index,destination=destination)
    
    
    print(' ', file = destination)
    print(f"Stepmode = ({current_stepConfig[0]},{current_stepConfig[1]}), stepSafetyFactor = {current_stepConfig[2]}", file = destination)
    print(' ', file = destination)


#Describe the input of the SSFM    
def describeInputConfig(current_time, fiber:fiber_class,  input_signal:input_signal_class,stepConfig, zinfo,fiber_index=""):
    
    with open(f"input_config_description_{fiber_index}.txt","w") as output_file:
            #Print info to terminal

            describe_run( current_time, fiber,  input_signal,stepConfig, zinfo,fiber_index=fiber_index)
            
            #Print info to file
            describe_run( current_time, fiber,  input_signal,stepConfig,zinfo, fiber_index=fiber_index  ,destination = output_file)    


def createOutputDirectory(experimentName):
    base_dir=os.getcwd()+'\\'
    os.chdir(base_dir)
    

    current_dir = "Simulation Results\\most_recent_run\\"
    current_time = datetime.now()
    
    if experimentName == "most_recent_run":
        current_dir = "Simulation Results\\most_recent_run\\"
        overwrite_folder_flag = True  
    else: 
        
        current_dir =base_dir+ f"Simulation Results\\{experimentName}\\{current_time.year}_{current_time.month}_{current_time.day}_{current_time.hour}_{current_time.minute}_{current_time.second}\\"
        overwrite_folder_flag = False 
        
    os.makedirs(current_dir,exist_ok=overwrite_folder_flag)
    os.chdir(current_dir)
    
    print(f"Current time is {current_time}")
    print("Current dir is "+current_dir)
    
    return (base_dir,current_dir) , current_time



#Save step config in folder
def saveStepConfig(stepConfig):

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

#Load step config from previous run
def load_StepConfig(path):
    
    df = pd.read_csv(path+'\\stepConfig.csv')
    
    stepmode = df['stepmode'][0]
    stepNumber_or_stepApproach = df['stepNumber_or_stepApproach'][0]
    SafetyFactor = df['SafetyFactor'][0]
    
    return (stepmode,stepNumber_or_stepApproach,SafetyFactor)
    
    
#Load all relevant info about previous run.
def load_previous_run(basePath):
    
    print(f"Loading run in {basePath}")
    
    fiber_span      = load_fiber_span(basePath+'\\input_info\\')
    input_signal    = load_InputSignal(basePath+'\\input_info\\')
    stepConfig      = load_StepConfig( basePath+'\\input_info\\')
    
    print(f"Successfully loaded run in {basePath}")
    
    return fiber_span, input_signal, stepConfig


#Run SSFM
def SSFM(fiber_span:fiber_span_class,input_signal:input_signal_class,stepConfig=("fixed","cautious",10.0),experimentName ="most_recent_run"):
    print("########### Initializing SSFM!!! ###########")
    
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
    
    saveStepConfig(stepConfig)
    
    
    
    #Return to main output directory
    os.chdir(current_dir)
    
    
    current_input_signal = input_signal
    
    ssfm_result_list = []
    
    print(f"Starting SSFM loop over {len(fiber_span.fiber_list)} fibers")
    
    for i, fiber in enumerate(fiber_span.fiber_list):
    
        print(f"Propagating through fiber number {i+1} out of {fiber_span.number_of_fibers_in_span}")
 
        
        #Get z-steps and z-locations throughout fiber and save plots of these values to new folder
        zinfo = getZsteps(fiber,current_input_signal,stepConfig,fiber_index=str(i))
        
        
        #Initialize arrays to store pulse and spectrum throughout fiber
        ssfm_result = ssfm_result_class(current_input_signal,fiber,stepConfig,zinfo,experimentName,dirs)

    
        newFolderName = "Length_info\\"
        newFolderPath = newFolderName
        os.makedirs(newFolderPath,exist_ok=True)
        os.chdir(newFolderPath)

        #Print simulation info to both terminal and .txt file in output folder
        describeInputConfig(current_time, fiber,  current_input_signal,stepConfig, zinfo,fiber_index=str(i))
        
        #Return to main output directory
        os.chdir(current_dir)
        

        
    
        
        
        print(f"Running SSFM with nsteps = {len(zinfo[1])}")
        
       
        #Pre-calculate effect of dispersion and loss as it's the same everywhere
        disp_and_loss=np.exp((1j*fiber.beta2/2*(2*pi*input_signal.timeFreq.f)**2-fiber.alpha_Np_per_m/2))
        
        #Precalculate constants for nonlinearity
        nonlinearity=1j*fiber.gamma
        
        #Initialize arrays to store temporal profile and spectrum while calculating SSFM
        pulse    = np.copy(input_signal.amplitude )
        spectrum = np.copy(input_signal.spectrum )
        
        
        for n, dz in enumerate(zinfo[1]):   
            pulse*=np.exp(nonlinearity*getPower(pulse)*dz) #Apply nonlinearity
            
            spectrum = getSpectrumFromPulse(input_signal.timeFreq.t, pulse)*(disp_and_loss**dz) #Go to spectral domain and apply disp and loss
            
            
            pulse=getPulseFromSpectrum(input_signal.timeFreq.f, spectrum) #Return to time domain 
            
            
            #Store results and repeat
            ssfm_result.pulseMatrix[n+1,:]=pulse
            ssfm_result.spectrumMatrix[n+1,:]=spectrum
    
        #Append list of output results
        
        ssfm_result_list.append(ssfm_result)
        
        current_input_signal.amplitude =np.copy(pulse)
        current_input_signal.Pmax = np.max(getPower(pulse))
        #Exit current output directory and return to base directory.
    
    print("Finished running SSFM!!!")
    os.chdir(base_dir)
        
        
    return ssfm_result_list




#Function for optionally saving plots
def saveplot(basename):

    
    if basename.lower().endswith(('.pdf','.png','.jpg')) == False:
        basename+='.png'
        
    plt.savefig(basename, bbox_inches='tight', pad_inches=0)


#If SSFM run involves multiple concatenated fibers, this function stitches the z-steps together into one array
def unpackZvals(ssfm_result_list):
    
    if len(ssfm_result_list)==1:
        return ssfm_result_list[0].zinfo[0]
    
    zvals =np.array([])
    number_of_fibers = len(ssfm_result_list)
    previous_length = 0
    for i, ssfm_result in enumerate(ssfm_result_list):
        
    
        if i==0:
            zvals = np.copy(ssfm_result.zinfo[0][0:-1])
            
        elif  (i>0) and (i< number_of_fibers-1):
            zvals = np.append(zvals,ssfm_result.zinfo[0][0:-1]+previous_length) 
            
        elif i==number_of_fibers-1:
            zvals = np.append(zvals,ssfm_result.zinfo[0]+previous_length) 
       
        previous_length += ssfm_result.fiber.Length    
        
    return zvals

#If SSFM run involves multiple concatenated fibers, this function stitches the matrices together into one array
def unpackMatrix(ssfm_result_list,zvals,timeFreq,pulse_or_spectrum):
    
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
            matrix[0: len(ssfm_result.zinfo[0])-1, :] = sourceMatrix[0: len(ssfm_result.zinfo[0])-1, :]
            
        elif  (i>0) and (i< number_of_fibers-1):

            matrix[starting_row : starting_row + len(ssfm_result.zinfo[0])-1, :] = sourceMatrix[0: len(ssfm_result.zinfo[0])-1, :]

        elif i==number_of_fibers-1:

            matrix[starting_row : starting_row + len(ssfm_result.zinfo[0]), :] = sourceMatrix[0:len(ssfm_result.zinfo[0]), :]
            
        starting_row +=len(ssfm_result.zinfo[0])-1
    

    assert 1==2
    
    return matrix        

          
def plotFirstAndLastPulse(ssfm_result_list, nrange:int, dB_cutoff,**kwargs):
    

    
    timeFreq = ssfm_result_list[0].input_signal.timeFreq
    
    Nmin = int(timeFreq.number_of_points/2-nrange)
    Nmax = int(timeFreq.number_of_points/2+nrange)
    
    Tmin = timeFreq.t[Nmin]
    Tmax = timeFreq.t[Nmax]    
     
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
    
    timeFreq = ssfm_result_list[0].input_signal.timeFreq
    
    Nmin = int(timeFreq.number_of_points/2-nrange)
    Nmax = int(timeFreq.number_of_points/2+nrange)
    
   
     
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
   
    timeFreq = ssfm_result_list[0].input_signal.timeFreq   
    
    Nmin = int(timeFreq.number_of_points/2-nrange)
    Nmax = int(timeFreq.number_of_points/2+nrange)
    
 
    
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
    
    
    timeFreq = ssfm_result_list[0].input_signal.timeFreq   
    
    Nmin = int(timeFreq.number_of_points/2-nrange)
    Nmax = int(timeFreq.number_of_points/2+nrange)
      
    
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


    for kw, value in kwargs.items():
        if kw.lower()=='chirpplotrange' and type(value)==tuple:
            Cmatrix[Cmatrix<value[0]]=value[0]
            Cmatrix[Cmatrix>value[1]]=value[1]


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
  
    
    print('  ')
    plotFirstAndLastPulse(ssfm_result_list, nrange, dB_cutoff,**kwargs)
    plotPulseMatrix2D(ssfm_result_list,nrange,dB_cutoff)
    plotPulseChirp2D(ssfm_result_list,nrange,dB_cutoff,**kwargs) 
    plotPulseMatrix3D(ssfm_result_list,nrange,dB_cutoff)
    print('  ')

    
    
        


def plotFirstAndLastSpectrum(ssfm_result_list, nrange:int, dB_cutoff):

    timeFreq = ssfm_result_list[0].input_signal.timeFreq
    
    Nmin = int(timeFreq.number_of_points/2-nrange)
    Nmax = int(timeFreq.number_of_points/2+nrange)
    
  
    
    zvals = unpackZvals(ssfm_result_list)
    
    
    P_initial=getPower(ssfm_result_list[0].spectrumMatrix[0,Nmin:Nmax])
    P_final=getPower(ssfm_result_list[-1].spectrumMatrix[-1,Nmin:Nmax])


    
    Pmax_initial = np.max(P_initial)
    Pmax_final = np.max(P_final)
    Pmax=np.max([Pmax_initial,Pmax_final]) 

    f=timeFreq .f[Nmin:Nmax]/1e12
    
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
    
    timeFreq = ssfm_result_list[0].input_signal.timeFreq   
    zvals = unpackZvals(ssfm_result_list)
    matrix = unpackMatrix(ssfm_result_list,zvals,timeFreq,"spectrum")
    
    Nmin = int(timeFreq.number_of_points/2-nrange)
    Nmax = int(timeFreq.number_of_points/2+nrange)
    
    #Plot pulse evolution throughout fiber in normalized log scale
    os.chdir(ssfm_result_list[0].dirs[1])
    fig, ax = plt.subplots(dpi=200)
    ax.set_title('Spectrum Evolution (dB scale)')
    f = timeFreq.f[Nmin:Nmax]/1e12 
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
    
    timeFreq = ssfm_result_list[0].input_signal.timeFreq   
    zvals = unpackZvals(ssfm_result_list)
    matrix = unpackMatrix(ssfm_result_list,zvals,timeFreq,"spectrum")
    
    Nmin = int(timeFreq.number_of_points/2-nrange)
    Nmax = int(timeFreq.number_of_points/2+nrange)
    
    #Plot pulse evolution in 3D
    os.chdir(ssfm_result_list[0].dirs[1])
    fig, ax = plt.subplots(1,1, figsize=(10,7),subplot_kw={"projection": "3d"})
    plt.title("Spectrum Evolution (dB scale)")
      
    f = timeFreq.f[Nmin:Nmax]/1e9 
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
  

    print('  ')  
    plotFirstAndLastSpectrum(ssfm_result_list, nrange, dB_cutoff)
    plotSpectrumMatrix2D(ssfm_result_list, nrange, dB_cutoff)
    plotSpectrumMatrix3D(ssfm_result_list, nrange, dB_cutoff)
    print('  ')  

    




from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.legend import LineCollection
from matplotlib.colors import LinearSegmentedColormap

def makeChirpGif(ssfm_result_list,nrange:int,framerate=30):
    
    
    
    print("Making .gif anination of pulse evolution. This may take a while, so please be patient.")
    
    os.chdir(ssfm_result_list[0].dirs[1])
    
    print(f"The .gif animation will be saved in {os.getcwd()}")
    
    timeFreq = ssfm_result_list[0].input_signal.timeFreq   
    zvals = unpackZvals(ssfm_result_list)
    matrix = unpackMatrix(ssfm_result_list,zvals,timeFreq,"pulse")
    scalingFactor, letter =  getUnitsFromValue(np.max(zvals))
    
    Nmin = int(timeFreq.number_of_points/2-nrange)
    Nmax = int(timeFreq.number_of_points/2+nrange)
    
    Tmin = timeFreq.t[Nmin]
    Tmax = timeFreq.t[Nmax]
    
    points = np.array( [timeFreq.t*1e12 ,  getPower(matrix[len(zvals)-1,Nmin:Nmax])   ] ,dtype=object ).T.reshape(-1,1,2)
    segments = np.concatenate([points[0:-1],points[1:]],axis=1)
    
    
    
    
    #Make custom colormap
    colors = ["red" ,"gray", "blue"]
    cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)
    
    #Initialize color normalization function
    norm = plt.Normalize(-20,20)
    
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
      init()     #Reset axes
      ax.set_title(f'Pulse evolution, z = {np.round(zvals[i]/scalingFactor,1)}{letter}m')
      
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


#Calculate central time or frequency of pulse or spectrum
def getAverageTimeOrFreq(time_or_freq,pulse_or_spectrum):
    
    E=getEnergy(time_or_freq,pulse_or_spectrum)    
    meanValue = np.trapz( time_or_freq*getPower(pulse_or_spectrum),time_or_freq )/E   
    return meanValue
    
#Calculate variance of pulse or spectrum
def getVarianceTimeOrFreq(time_or_freq,pulse_or_spectrum):
    E=getEnergy(time_or_freq,pulse_or_spectrum)
    variance = np.trapz( time_or_freq**2*getPower(pulse_or_spectrum),time_or_freq )/E  - (getAverageTimeOrFreq(time_or_freq,pulse_or_spectrum))**2
    return variance   
    
#Calculate standard deviation of pulse or spectrum (pulse duration or spectral width)
def getStDevTimeOrFreq(time_or_freq,pulse_or_spectrum):
    return np.sqrt(getVarianceTimeOrFreq(time_or_freq,pulse_or_spectrum))


#Plot how central time, duration, central freq. and spectral width change with z. 
def plotAverageAndStdTimeAndFreq(ssfm_result_list):
    
    timeFreq = ssfm_result_list[0].input_signal.timeFreq   
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
    ax.set_ylabel(f'Time [{prefix_pulse}s]')
    ax.set_xlabel(f'Distance [{prefix_Z}m]')
    ax.tick_params(axis='y',labelcolor='C0')
    
    ax2=ax.twinx()
    ax2.plot(zvals/scalingFactor_Z,meanFreqArray/scalingFactor_spectrum,'C1-', label= "Spectrum Center")
    ax2.plot(zvals/scalingFactor_Z,stdFreqArray/scalingFactor_spectrum,'C1--',  label= "Spectrum Width")
    
    ax2.set_ylabel(f'Freq. [{prefix_spectrum}Hz]')
    ax2.tick_params(axis='y',labelcolor='C1')
    fig.legend(bbox_to_anchor=(1.35,0.8))
    
    saveplot('Width_evo') 
    plt.show()
    os.chdir(ssfm_result_list[0].dirs[0])
    

#Call all the plotting functions in one go
def plotEverythingAboutResult(ssfm_result_list, 
                              nrange_pulse, 
                              dB_cutoff_pulse, 
                              nrange_spectrum, 
                              dB_cutoff_spectrum, 
                              **kwargs):
    
    plotAverageAndStdTimeAndFreq(ssfm_result_list)
    
    plotEverythingAboutPulses(ssfm_result_list, 
                                  nrange_pulse, 
                                  dB_cutoff_pulse, **kwargs)
    
    plotEverythingAboutSpectra(ssfm_result_list,
                                   nrange_spectrum, 
                                   dB_cutoff_spectrum)

if __name__ == "__main__":
    
    os.chdir(os.path.realpath(os.path.dirname(__file__)))
    
    
    N  = 2**16 #Number of points
    dt = 1e-12 #Time resolution [s] 
    
    
    timeFreq_test=timeFreq_class(N,dt)
    
    testAmplitude = np.sqrt(1)                    #Amplitude in units of sqrt(W)
    testDuration  =400*dt   #Pulse 1/e^2 duration [s]
    testOffset    = 0                       #Time offset
    testChirp = 0
    testCarrierFreq=0
    testPulseType='gaussian' 
    testOrder = 1
    testNoiseAmplitude = 0
    

    testInputSignal = input_signal_class(timeFreq_test, 
                                          testAmplitude ,
                                          testDuration,
                                          testOffset,
                                          testChirp,
                                          testCarrierFreq,
                                          testPulseType,
                                          testOrder,
                                          testNoiseAmplitude)
    
    testInputSignal.amplitude = np.cos(2*np.pi*5e9*testInputSignal.timeFreq.t)*testInputSignal.amplitude
    testInputSignal.spectrum  = getSpectrumFromPulse(testInputSignal.timeFreq.t, testInputSignal.amplitude) 
      
    #  Initialize fibers
    fiber_disp_positive = fiber_class(1000, 1e-100,   300e3*1e-30,    0  )
    fiber_gamma_only    = fiber_class(1000, 1e-4,   300e3*1e-300,    0  )
    fiber_test          = fiber_class(1000, 8e-3,   -30e3*1e-300,    0  )
    
    
    fiber_list = [fiber_test ]
    fiber_span = fiber_span_class(fiber_list)
    
    

    
    testSafetyFactor = 10
    testStepConfig=("fixed",2**5,testSafetyFactor)




    expName = 'FWM'
    #Run SSFM
    ssfm_result_list = SSFM(fiber_span,testInputSignal,stepConfig=testStepConfig,experimentName=expName)
    
    
    
    #Plot pulses
    nrange_test_pulse=800
    cutoff_test_pulse=-60

    #Plot pulses
    nrange_test_spectrum=2400
    cutoff_test_spectrum=-60

    plotEverythingAboutResult(ssfm_result_list,
                              nrange_test_pulse,
                              cutoff_test_pulse,
                              nrange_test_spectrum,
                              cutoff_test_spectrum,
                              )
    
    #makeChirpGif(ssfm_result_list,nrange_test_pulse,framerate=30)
    
    
    
    