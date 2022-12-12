# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 10:23:48 2022

@author: okrarup
"""

import numpy as np
from scipy.fftpack import fft, ifft, fftshift, ifftshift, fftfreq

import matplotlib.pyplot as plt
from matplotlib import cm

global pi; pi=np.pi 


def getFreqRangeFromTime(time):
    return fftshift(fftfreq(len(time), d=time[1]-time[0]))

def getPhase(pulse):
    phi=np.unwrap(np.angle(pulse)) #Get phase starting from 1st entry
    phi=phi-phi[int(len(phi)/2)]   #Center phase on middle entry
    return phi    


def getChirp(time,pulse):
    phi=getPhase(pulse)
    dphi=np.diff(phi ,prepend = phi[0] - (phi[1]  - phi[0]  ),axis=0) #Change in phase. Prepend to ensure consistent array size 
    dt  =np.diff(time,prepend = time[0]- (time[1] - time[0] ),axis=0) #Change in time.  Prepend to ensure consistent array size

    return -1.0/(2*pi)*dphi/dt #Chirp = - 1/(2pi) * d(phi)/dt
    
    


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
        
    def describe_config(self):
        print("### Configuration Parameters ###")
        print(f" Number of points = {self.number_of_points}")
        print(f" Start time, tmin = {self.tmin*1e12}ps")
        print(f" Stop time, tmax = {self.tmax*1e12}ps")
        print(f" Time resolution, dt = {self.time_step*1e12}ps")
        print("  ")
        print(f" Start frequency= {self.fmin/1e12}THz")
        print(f" Stop frequency = {self.fmax/1e12}THz")
        print(f" Frequency resolution= {self.freq_step/1e6}MHz")
        print( "   ")
        


        
    
        
#Function returns pulse power or spectrum PSD
def getPower(amplitude):
    return np.abs(amplitude)**2  

#Function gets the energy of a pulse pulse or spectrum by integrating the power
def getEnergy(time_or_frequency,amplitude):
    return np.trapz(getPower(amplitude),time_or_frequency)

#TODO: Add support for different carrier frequencies. Hint: Multiply by complex exponential!
#TODO: Add support for pre-chirped pulses. 
def GaussianPulse(time,amplitude,duration,offset,chirp,order,carrier_freq_Hz):
    assert 1 <= order, f"Error: Order of gaussian pulse is {order}. Must be >=1"
    return amplitude*np.exp(- (1+1j*chirp)/2*((time-offset)/(duration))**(2*np.floor(order)))*np.exp(-1j*2*pi*carrier_freq_Hz*time)

def squarePulse(time,amplitude,duration,offset,chirp,carrier_freq_Hz):
    return GaussianPulse(time,amplitude,duration,offset,chirp,100,carrier_freq_Hz)


#Define sech pulse
def sechPulse(time,amplitude,duration,offset,chirp,carrier_freq_Hz):
    return amplitude/np.cosh((time-offset)/duration)*np.exp(- (1j*chirp)/2*((time-offset)/(duration))**2)*np.exp(-1j*2*pi*carrier_freq_Hz*time)


#Define function for adding white noise
def noise_ASE(time,amplitude):
    randomAmplitudes=np.random.normal(loc=0.0, scale=amplitude, size=len(time))
    randomPhases = np.random.uniform(-pi,pi, len(time))
    return randomAmplitudes*np.exp(1j*randomPhases)   


def getPulse(time,amplitude,duration,offset,chirp,carrier_freq_Hz,pulseType,order=1,noiseAmplitude=0.0):
    
    
    noise = noise_ASE(time,noiseAmplitude)
    
    if pulseType.lower()=="gaussian":
        return GaussianPulse(time,amplitude,duration,offset,chirp,order,carrier_freq_Hz)+noise
    
    if pulseType.lower()=="sech":
        return sechPulse(time,amplitude,duration,offset,chirp,carrier_freq_Hz)+noise
    
    if pulseType.lower()=="square":
        return squarePulse(time,amplitude,duration,offset,chirp,carrier_freq_Hz)+noise
    
    if pulseType.lower()=="custom":
        return noise


def getSpectrumFromPulse(time,pulse_amplitude):
    pulseEnergy=getEnergy(time,pulse_amplitude) #Get pulse energy
    f=getFreqRangeFromTime(time) 
    dt=time[1]-time[0]
    
    spectrum_amplitude=fftshift(fft(pulse_amplitude))*dt #Take FFT and do shift
    spectrumEnergy=getEnergy(f, spectrum_amplitude) #Get spectrum energy
    
    err=np.abs((pulseEnergy/spectrumEnergy-1))
    
    assert( err<1e-7 ), f'ERROR = {err}: Energy changed when going from Pulse to Spectrum!!!' 
    
    return spectrum_amplitude



#Equivalent function for getting time base from frequency range
def getTimeFromFrequency(frequency):  
    return fftshift(fftfreq(len(frequency), d=frequency[1]-frequency[0]))


#Equivalent function for getting pulse from spectrum
def getPulseFromSpectrum(frequency,spectrum_amplitude):
    
    spectrumEnergy=getEnergy(frequency, spectrum_amplitude)
    
    time = getTimeFromFrequency(frequency)
    dt = time[1]-time[0]
     
    pulse = ifft(ifftshift(spectrum_amplitude))/dt
    pulseEnergy = getEnergy(time, pulse)
    
    err=np.abs((pulseEnergy/spectrumEnergy-1))

    assert( err<1e-7   ), f'ERROR = {err}: Energy changed when going from Spectrum to Pulse!!!' 
    
    return pulse

#Equivalent function for generating a Gaussian spectrum
def GaussianSpectrum(frequency,amplitude,bandwidth,carrier_freq_Hz):
    time = getTimeFromFrequency(frequency)
    return getSpectrumFromPulse(time, GaussianPulse(time, amplitude, 1/bandwidth, 0,0,1,carrier_freq_Hz))


class Fiber_class:
  def __init__(self,L,gamma,beta2,alpha_dB_per_m):
      self.Length=L
      self.gamma=gamma
      self.beta2=beta2
      self.alpha_dB_per_m=alpha_dB_per_m
      self.alpha_Np_per_m = self.alpha_dB_per_m*np.log(10)/10.0 #Loss coeff is usually specified in dB/km, but Nepers/km is more useful for calculations
      #TODO: Make alpha frequency dependent.  
      
class input_signal_class:
    def __init__(self,timeFreq:timeFreq_class,peak_amplitude,duration,offset,chirp,carrier_freq_Hz,pulseType,order=1,noiseAmplitude=0):

        
        self.timeFreq=timeFreq
        self.amplitude = getPulse(self.timeFreq.t,peak_amplitude,duration,offset,chirp,carrier_freq_Hz,pulseType,order=1,noiseAmplitude=0)
        

        if getEnergy(self.timeFreq.t, self.amplitude) == 0.0:
            self.spectrum = np.copy(self.amplitude)  
        else:
            self.spectrum = getSpectrumFromPulse(self.timeFreq.t,self.amplitude)   
        
        self.Pmax=np.max(getPower(self.amplitude))
        self.duration=duration
        self.offset=offset
        self.chirp=chirp
        self.carrier_freq_Hz=carrier_freq_Hz
        self.pulseType=pulseType
        self.order=order
        self.noiseAmplitude=noiseAmplitude
        



def zstep_NL(z,fiber:Fiber_class, input_signal:input_signal_class,stepmode,stepSafetyFactor):
    
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




def getZsteps( fiber:Fiber_class, input_signal:input_signal_class,stepmode,stepSafetyFactor):    
    
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




class ssfm_output_class:
    def __init__(self, input_signal:input_signal_class, fiber:Fiber_class,zinfo):
        self.zvals=zinfo[0]
        self.zsteps=zinfo[1]
        
        self.n_z_locs=len(zinfo[0])
        self.n_z_steps=len(zinfo[1])
        
        plt.figure()
        plt.title('zvals')
        plt.plot(self.zvals,'.')
        plt.show()
        
        plt.figure()
        plt.title(f'zsteps {self.n_z_steps}')
        plt.plot(self.zsteps,'.')
        plt.show()        
        
        self.pulseMatrix = np.zeros((self.n_z_locs,input_signal.timeFreq.number_of_points ) )*(1+0j)
        self.spectrumMatrix = np.copy(self.pulseMatrix)
        
        self.pulseMatrix[0,:]=np.copy(input_signal.amplitude)   
        self.spectrumMatrix[0,:] = np.copy(input_signal.spectrum)
        
        self.fiber=fiber
        self.timeFreq=input_signal.timeFreq







    
    

      
def SSFM(fiber:Fiber_class,input_signal:input_signal_class,stepConfig=("fixed","cautious"),stepSafetyFactor=1.0):
    
    if fiber.beta2 != 0.0:
        
        Length_disp = input_signal.duration**2/np.abs(fiber.beta2)
    else:
        Length_disp=np.inf
        
    
    if fiber.gamma !=0.0:
        Length_NL = 1/fiber.gamma/input_signal.Pmax   
        testN=np.sqrt(Length_disp/Length_NL)
    else:
        Length_NL=np.inf
        testN=np.NaN

    Nmin = np.sqrt(0.25*np.exp(3/2)) #Minimum N-value of Optical Wave breaking with Gaussian pulse
    Length_wave_break = Length_disp/np.sqrt(testN**2/Nmin**2-1)  #Characteristic length for Optical Wave breaking with Gaussian pulse

    print(f"testN={testN}")
    print(f"Length_disp={Length_disp/1e3}km")
    print(f"Length_NL={Length_NL/1e3}km")
    print(f"Length_wave_break = {Length_wave_break/1e3} km")

    
    
    print("Calculating zinfo")
    print(f"Stepmode = {stepConfig}, stepSafetyFactor = {stepSafetyFactor}")
    if stepConfig[0].lower() == "fixed":
        
        if type(stepConfig[1]) == str:
        
            dz=zstep_NL(0,fiber, input_signal,stepConfig[1],stepSafetyFactor)
            z_array=np.arange(0,fiber.Length,dz)
            
            if z_array[-1] != fiber.Length:
                z_array=np.append(z_array,fiber.Length)
            
            dz_array = np.diff( z_array)
            
            
        elif type(stepConfig[1]) == int:
            z_array=np.linspace(0,fiber.Length,stepConfig[1]+1)
            dz_array=np.ones( stepConfig[1])*(z_array[1]-z_array[0])            

            
        zinfo    =(z_array,dz_array)
        
    else:
        zinfo = getZsteps(fiber,input_signal,stepConfig[1],stepSafetyFactor)
        
    
    print(f"Running SSFM with nsteps = {len(zinfo[1])}")
    
    #Initialize arrays to store pulse and spectrum throughout fiber
    ssfm_result = ssfm_output_class(input_signal,fiber,zinfo)
    
    
    #Pre-calculate effect of dispersion and loss as it's the same everywhere
    disp_and_loss=np.exp((1j*fiber.beta2/2*(2*pi*input_signal.timeFreq.f)**2-fiber.alpha_Np_per_m/2))
    
    #Precalculate constants for nonlinearity
    nonlinearity=1j*fiber.gamma
    pulse    = np.copy(input_signal.amplitude )
    spectrum = np.copy(input_signal.spectrum )
    
    for n, dz in enumerate(zinfo[1]):   
        pulse*=np.exp(nonlinearity*getPower(pulse)*dz) #Apply nonlinearity
        
        spectrum = getSpectrumFromPulse(input_signal.timeFreq.t, pulse)*(disp_and_loss**dz) #Go to spectral domain and apply disp and loss
        
        
        pulse=getPulseFromSpectrum(input_signal.timeFreq.f, spectrum) #Return to time domain 
        
        
        #Store results and repeat
        ssfm_result.pulseMatrix[n+1,:]=pulse
        ssfm_result.spectrumMatrix[n+1,:]=spectrum

        #print(f"Finished {np.round(n/len(zinfo[1])*100,3)}% of simulation")
    #Return results
    print("Finished running SSFM!!!")
    return ssfm_result




#Function for optionally saving plots
def saveplot(basename,**kwargs):
  for kw, value in kwargs.items():
    if kw.lower()=='savename' and type(value)==str:
      savestring=basename+'_'+value
      if value.lower().endswith(('.pdf','.png','.jpg')) == False:
        savestring+='.png'
      plt.savefig(savestring,
                  bbox_inches='tight', 
                  transparent=True,
                  pad_inches=0)


#Function for optionally deleting plots     
import os
from IPython.lib.display import isdir
def removePlots(filetypes):
  dir_name = os.getcwd()
  filelist = os.listdir(dir_name)
  for item in filelist:
    
    for filetype in filetypes:
      if isdir(item):
        continue

      if item.endswith(filetype):
          print("Removed:"+item)
          os.remove(item)
          
def plotFirstAndLastPulse(matrix,fiber:Fiber_class,sim:timeFreq_class,zvals, nrange:int, dB_cutoff,**kwargs):
  t=sim.t[int(sim.number_of_points/2-nrange):int(sim.number_of_points/2+nrange)]*1e12
  
  P_initial=getPower(matrix[0,int(sim.number_of_points/2-nrange):int(sim.number_of_points/2+nrange)])
  P_final=getPower(matrix[-1,int(sim.number_of_points/2-nrange):int(sim.number_of_points/2+nrange)])
  
  
  Pmax_initial = np.max(P_initial)
  Pmax_final = np.max(P_final)
  Pmax=np.max([Pmax_initial,Pmax_final])
 
  plt.figure()
  plt.title("Initial pulse and final pulse")
  plt.plot(t,P_initial,label="Initial Pulse at z = 0")
  plt.plot(t,P_final,label=f"Final Pulse at z = {zvals[-1]/1e3}km")
  plt.xlabel("Time [ps]")
  plt.ylabel("Power [W]")
  plt.ylim(Pmax/(10**(-dB_cutoff/10)),1.05*Pmax)
  #plt.yscale('log')
  
  plt.legend()
  saveplot('first_and_last_pulse',**kwargs)
  plt.show()  


def plotPulseMatrix2D(matrix,fiber:Fiber_class,sim:timeFreq_class,zvals, nrange:int, dB_cutoff,**kwargs):
  #Plot pulse evolution throughout fiber in normalized log scale
  fig, ax = plt.subplots()
  ax.set_title('Pulse Evolution (dB scale)')
  t = sim.t[int(sim.number_of_points/2-nrange):int(sim.number_of_points/2+nrange)]*1e12
  z = zvals
  T, Z = np.meshgrid(t, z)
  P=getPower(matrix[:,int(sim.number_of_points/2-nrange):int(sim.number_of_points/2+nrange)]  )/np.max(getPower(matrix[:,int(sim.number_of_points/2-nrange):int(sim.number_of_points/2+nrange)]))
  P[P<1e-100]=1e-100
  P = 10*np.log10(P)
  P[P<dB_cutoff]=dB_cutoff
  surf=ax.contourf(T, Z, P,levels=40, cmap="jet")
  ax.set_xlabel('Time [ps]')
  ax.set_ylabel('Distance [m]')
  cbar=fig.colorbar(surf, ax=ax)
  saveplot('pulse_evo_2D',**kwargs) 
  plt.show()

def plotPulseMatrix3D(matrix,fiber:Fiber_class,sim:timeFreq_class,zvals, nrange:int, dB_cutoff,**kwargs):
  #Plot pulse evolution in 3D
  fig, ax = plt.subplots(1,1, figsize=(10,7),subplot_kw={"projection": "3d"})
  plt.title("Pulse Evolution (dB scale)")

  t = sim.t[int(sim.number_of_points/2-nrange):int(sim.number_of_points/2+nrange)]*1e12
  z = zvals
  T_surf, Z_surf = np.meshgrid(t, z)
  P_surf=getPower(matrix[:,int(sim.number_of_points/2-nrange):int(sim.number_of_points/2+nrange)]  )/np.max(getPower(matrix[:,int(sim.number_of_points/2-nrange):int(sim.number_of_points/2+nrange)]))
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
  saveplot('pulse_evo_3D',**kwargs)
  plt.show()


def plotPulseChirp2D(matrix,fiber:Fiber_class,sim:timeFreq_class,zvals, nrange:int,**kwargs):
  #Plot pulse evolution throughout fiber in normalized log scale
  fig, ax = plt.subplots()
  ax.set_title('Pulse Chirp Evolution')
  t = sim.t[int(sim.number_of_points/2-nrange):int(sim.number_of_points/2+nrange)]*1e12
  z = zvals
  T, Z = np.meshgrid(t, z)
  
  
  Cmatrix=np.ones( (len(z),len(t))  )*1.0

  for i in range(len(zvals)):
    Cmatrix[i,:]=getChirp(t/1e12,matrix[i,int(sim.number_of_points/2-nrange):int(sim.number_of_points/2+nrange)])/1e9


  for kw, value in kwargs.items():
    if kw.lower()=='chirpplotrange' and type(value)==tuple:
      Cmatrix[Cmatrix<value[0]]=value[0]
      Cmatrix[Cmatrix>value[1]]=value[1]
  

  surf=ax.contourf(T, Z, Cmatrix,levels=40,cmap='RdBu')
  
  ax.set_xlabel('Time [ps]')
  ax.set_ylabel('Distance [m]')
  cbar=fig.colorbar(surf, ax=ax)
  cbar.set_label('Chirp [GHz]')
  saveplot('chirp_evo_2D',**kwargs) 
  plt.show()


def plotEverythingAboutPulses(ssfm_result:ssfm_output_class, 
                              nrange:int, 
                              dB_cutoff,
                              **kwargs):
  
  
    
  print('  ')
  plotFirstAndLastPulse(ssfm_result.pulseMatrix,ssfm_result.fiber,ssfm_result.timeFreq,ssfm_result.zvals, nrange, dB_cutoff,**kwargs)
  plotPulseMatrix2D(ssfm_result.pulseMatrix,ssfm_result.fiber,ssfm_result.timeFreq,ssfm_result.zvals,nrange,dB_cutoff,**kwargs)
  plotPulseChirp2D(ssfm_result.pulseMatrix,ssfm_result.fiber,ssfm_result.timeFreq,ssfm_result.zvals,nrange,**kwargs) 
  plotPulseMatrix3D(ssfm_result.pulseMatrix,ssfm_result.fiber,ssfm_result.timeFreq,ssfm_result.zvals,nrange,dB_cutoff,**kwargs)
  print('  ')  


def plotFirstAndLastSpectrum(matrix,fiber:Fiber_class,sim:timeFreq_class, nrange:int, dB_cutoff,**kwargs):

    P_initial=getPower(matrix[0,int(sim.number_of_points/2-nrange):int(sim.number_of_points/2+nrange)])*1e9
    P_final=getPower(matrix[-1,int(sim.number_of_points/2-nrange):int(sim.number_of_points/2+nrange)])*1e9
    
    
    Pmax_initial = np.max(P_initial)
    Pmax_final = np.max(P_final)
    Pmax=np.max([Pmax_initial,Pmax_final]) 

    f=sim.f[int(sim.number_of_points/2-nrange):int(sim.number_of_points/2+nrange)]/1e9
    plt.figure()
    plt.title("Initial spectrum and final spectrum")
    plt.plot(f,P_initial,label="Initial Spectrum")
    plt.plot(f,P_final,label="Final Spectrum")
    plt.xlabel("Freq. [GHz]")
    plt.ylabel("PSD [W/GHz]")
    plt.yscale('log')
    plt.ylim(Pmax/(10**(-dB_cutoff/10)),1.05*Pmax)
    plt.legend()
    saveplot('first_and_last_spectrum',**kwargs)
    plt.show()

def plotSpectrumMatrix2D(matrix,fiber:Fiber_class,sim:timeFreq_class,zvals, nrange:int, dB_cutoff,**kwargs):
  #Plot pulse evolution throughout fiber in normalized log scale
  fig, ax = plt.subplots()
  ax.set_title('Spectrum Evolution (dB scale)')
  f = sim.f[int(sim.number_of_points/2-nrange):int(sim.number_of_points/2+nrange)]/1e9 
  z = zvals
  F, Z = np.meshgrid(f, z)
  Pf=getPower(matrix[:,int(sim.number_of_points/2-nrange):int(sim.number_of_points/2+nrange)]  )/np.max(getPower(matrix[:,int(sim.number_of_points/2-nrange):int(sim.number_of_points/2+nrange)]))
  Pf[Pf<1e-100]=1e-100
  Pf = 10*np.log10(Pf)
  Pf[Pf<dB_cutoff]=dB_cutoff
  surf=ax.contourf(F, Z, Pf,levels=40)
  ax.set_xlabel('Freq. [GHz]')
  ax.set_ylabel('Distance [m]')
  cbar=fig.colorbar(surf, ax=ax) 
  saveplot('spectrum_evo_2D',**kwargs) 
  plt.show()

def plotSpectrumMatrix3D(matrix,fiber:Fiber_class,sim:timeFreq_class,zvals, nrange:int, dB_cutoff,**kwargs):
  #Plot pulse evolution in 3D
  fig, ax = plt.subplots(1,1, figsize=(10,7),subplot_kw={"projection": "3d"})
  plt.title("Spectrum Evolution (dB scale)")

  f = sim.f[int(sim.number_of_points/2-nrange):int(sim.number_of_points/2+nrange)]/1e9 
  z = zvals
  F_surf, Z_surf = np.meshgrid(f, z)
  P_surf=getPower(matrix[:,int(sim.number_of_points/2-nrange):int(sim.number_of_points/2+nrange)]  )/np.max(getPower(matrix[:,int(sim.number_of_points/2-nrange):int(sim.number_of_points/2+nrange)]))
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
  saveplot('spectrum_evo_3D',**kwargs) 
  plt.show()


def plotEverythingAboutSpectra(ssfm_result:ssfm_output_class,
                               nrange:int, 
                               dB_cutoff,
                               **kwargs):
  
  print('  ')  
  plotFirstAndLastSpectrum(ssfm_result.spectrumMatrix,ssfm_result.fiber,ssfm_result.timeFreq, nrange, dB_cutoff,**kwargs)
  plotSpectrumMatrix2D(ssfm_result.spectrumMatrix,ssfm_result.fiber,ssfm_result.timeFreq,ssfm_result.zvals,nrange,dB_cutoff,**kwargs)
  plotSpectrumMatrix3D(ssfm_result.spectrumMatrix,ssfm_result.fiber,ssfm_result.timeFreq,ssfm_result.zvals,nrange,dB_cutoff,**kwargs)
  print('  ')  





if __name__ == "__main__":
    
    N  = 2**12 #Number of points
    dt = 0.1e-12 #Time resolution [s] 
    
    
    timeFreq_test=timeFreq_class(N,dt)
    
    #Define fiberulation parameters
    Length          = 53*1.5      #Fiber length in m
    #nsteps          = 2**8     #Number of steps we divide the fiber into
    
    gamma           = 400e-3     #Nonlinearity parameter in 1/W/m 
    beta2           = 100e3    #Dispersion in fs^2/m (units typically used when referring to beta2) 
    beta2          *= (1e-30)  #Convert fs^2 to s^2 so everything is in SI units
    alpha_dB_per_m  = 0.0e-3   #Power attenuation coeff in decibel per m. Usual value at 1550nm is 0.2 dB/km
    
    #Note:  beta2>0 is normal dispersion with red light pulling ahead, 
    #       causing a negative leading chirp
    #       
    #       beta2<0 is anormalous dispersion with blue light pulling ahead, 
    #       causing a positive leading chirp.
    
      
    #  Initialize class
    fiber=Fiber_class(Length, gamma, beta2, alpha_dB_per_m)
    
    
    #Initialize Gaussian pulse

    
    testAmplitude = np.sqrt(1)                    #Amplitude in units of sqrt(W)
    testDuration  =100*timeFreq_test.time_step   #Pulse 1/e^2 duration [s]
    testOffset    = 0                       #Time offset
    testChirp = 0
    testCarrierFreq=0
    testPulseType='gaussian' 
    testOrder = 1
    testNoiseAmplitude = 0.0
    
    
    
    
    
    Length_disp = testDuration**2/np.abs(beta2)
    Length_NL = 1/gamma/testAmplitude**2   
    

    Nmin = np.sqrt(0.25*np.exp(3/2))

    
    testN=np.sqrt(Length_disp/Length_NL)


    
    #https://prefetch.eu/know/concept/optical-wave-breaking/
    
    Length_wave_break = Length_disp/np.sqrt(testN**2/Nmin**2-1) 
    
    print(f"testN={testN}, Length_disp={Length_disp/1e3}km, Length_NL={Length_NL/1e3}km, Length_wave_break = {Length_wave_break/1e3} km ")
    
    
    
    testInputSignal = input_signal_class(timeFreq_test, 
                                         testAmplitude ,
                                         testDuration,
                                         testOffset,
                                         testChirp,
                                         testCarrierFreq,
                                         testPulseType,
                                         testOrder,
                                         testNoiseAmplitude)
    
    
    if testPulseType.lower() == "custom":
        
        testInputSignal.amplitude +=  getPulse(testInputSignal.timeFreq.t,testAmplitude/100,testDuration*5,0,0,-0.5e9,"square")
        testInputSignal.amplitude +=  getPulse(testInputSignal.timeFreq.t,testAmplitude,testDuration/5,0  ,0,0.5e9,"gaussian")
        testInputSignal.Pmax=np.max(getPower(testInputSignal.amplitude))
        testInputSignal.duration=testDuration/5
        
        testInputSignal.spectrum=getSpectrumFromPulse(testInputSignal.timeFreq.t, testInputSignal.amplitude)
    
    testStepConfig=("fixed",2**11)
    testSafetyFactor = 40
    
    #Run SSFM
    ssfm_result_test = SSFM(fiber,testInputSignal,stepConfig=testStepConfig,stepSafetyFactor=testSafetyFactor)
    
    #Plot pulses
    nrange_test=400
    cutoff_test=-30
    plotEverythingAboutPulses(ssfm_result_test,nrange_test,cutoff_test,chirpPlotRange=(-60,60))
    
 
    nrange_test=200
    cutoff_test=-60    
    
    #Plot spectra
    plotEverythingAboutSpectra(ssfm_result_test,nrange_test,cutoff_test)


