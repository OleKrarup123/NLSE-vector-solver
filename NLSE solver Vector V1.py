# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 13:28:33 2020

@author: Bruger
"""




#In this script, I try to model the effect of multiples frequencies of light inside the NL medium. 
#For random birefringence, I reuse modified code taken from Agrawall, specifically chapter 7.4. 


#Import Math libraries
import numpy as np
import pandas as pd
import scipy as sp
import scipy.fftpack
import scipy.optimize
from scipy.fftpack import fft, ifft, fftshift,ifftshift

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import cm
#Define pi for convenience
global pi; pi=np.pi 


# import py_pol as ppp
# from py_pol import degrees, np
# from py_pol.stokes import Stokes
# from py_pol.mueller import Mueller
# from py_pol.drawings import draw_poincare_sphere, draw_on_poincare

import random


#Import the custom gif library to make animations of results
#import gif
###


#Specify custom style for plots
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'lines.linewidth': 5})

def f2lBW(DeltaF,lamb_c):
    return DeltaF*lamb_c**2/c

def l2fBW(DeltaL,freq_c):
    return freq_c**2/c*DeltaL



def getPhase(a): #Returns the local phase
    
            
    phi= np.angle(a)
    phi=(phi + 2 * np.pi) % (2 * np.pi)
    phi=np.unwrap(phi)
                
            
    #phi=np.cumsum(phi)/2/pi
    phi=phi-phi[int(np.floor(len(phi)/2))]
    return (phi)

def getChirp(t,a): #Note: Time must be in s
    
    phase=getPhase(a)
    step=np.diff(t)
    dummy=np.diff(phase)
    dummy=-1.0*np.append(dummy[0]-(dummy[1]-dummy[0]),dummy)/(step[0])/2/pi #Return the local chirp in Hz
    
    return dummy

#Define function to obtain the frequency spectrum of a pulse
#Define function to obtain the frequency spectrum of a pulse
def getSpectrumFromPulse(t,a):
    
    Energy1=np.round(pulseEnergy(t,a),5) #Find energy of pulse
    n=len(t) #length of time vector
    dt=t[1]-t[0] #Length of time step
    f = fftshift(scipy.fftpack.fftfreq(n, d=dt)) #Generate frequencies
    A0ff0=fftshift((fft(a[0])))*(dt) #Take FFT of pulse to get frequencies
    A0ff1=fftshift((fft(a[1])))*(dt) #Take FFT of pulse to get frequencies
    
    output=np.array([A0ff0,A0ff1])
    Energy2=np.round(spectralEnergy(f,output),5) #Determine total energy of spectrum

    if Energy1 != Energy2: #Verify that energy has not been lost
        print("Warning energy was lost when converting from pulse to spectrum")
        print('Spectral energy ='+str(Energy2))
        print('Pulse energy ='+str(Energy1))
        
        assert(Energy1==Energy2)    
    assert(Energy1==Energy2)
    
    return f,output
    
#Define a function to get the pulse corresponding to a certain spectrum
def getPulseFromSpectrum(f,af):
    
    Energy2=np.round(spectralEnergy(f,af),5) #Energy of spectrum
    
    n=len(f) #Length of list of requencies
    df=f[1]-f[0] #Frequency step
    t = ifftshift(scipy.fftpack.fftfreq(n, d=df)) #generate time trace
    A0=scipy.fftpack.ifft(ifftshift((af[0])))/(t[1]-t[0])# Take FFT of spectrum to get pulse (NEED TO BE iFFT?)
    A1=scipy.fftpack.ifft(ifftshift((af[1])))/(t[1]-t[0])# Take FFT of spectrum to get pulse (NEED TO BE iFFT?)
    
    #A = np.divide(A[-np.arange(A.shape[0])],s.shape[-1])
    output=np.array([A0,A1])
    Energy1=np.round(pulseEnergy(t,output),5) #Energy of pulse
    
    #Compare energies
    if Energy1 != Energy2:
        print("   ")
        print("Warning energy was lost when converting from spectrum to pulse")
        print('Spectral energy ='+str(Energy2))
        print('Pulse energy ='+str(Energy1))
        print("   ")
        
        assert(Energy1==Energy2)
    assert(Energy1==Energy2)
    return t,output    #Return times and pulse   [::-1]
    
#Define a box of height 1 centered at t0 and width w
def box(t,t0,w):
    out=np.ones_like(t)*1e-100
    
    i=0
    for ts in t:
        if np.abs(ts-t0)<=w/2:
            out[i]=1
        i=i+1    
    return out

def pulsePower(a):
    return np.abs(a[0])**2+np.abs(a[1])**2

def pulseEnergy(t,a): #Determines the energy of the input pulse in [J] if given a time vector in [s] and pulse in units of sqrt(W)
    return np.trapz(pulsePower(a),t)

def spectralPower(af):
    return np.abs(af[0])**2+np.abs(af[1])**2

def spectralEnergy(f,af): #Takes a list of frequencies [Hz] and a spectrum in sqrt(W/Hz) and returns total energy
    
    return np.trapz(spectralPower(af),f)
   


def GaussianPulse(t,A,T0,s,c,pol): #Function used to define a Gaussian pulse
    
    
    amplitude=A*np.exp(-(t-T0)**2/(2*s**2))+c
    return np.array([pol[0]*amplitude,pol[1]*amplitude])

def GaussianSpectrum(omega,Aw,w0,sw,cw): #Function used to define a Gaussian spectral line.
    return Aw*np.exp(-(omega-w0)**2/(2*sw**2))+cw


def getSideband(freq,spectrum,fs,fp,n):
    fd=np.abs(fp-fs)
    boxfilt=box(freq,fp+(n+1)*fd,fd/5)
    return np.array([spectrum[0]*boxfilt,spectrum[1]*boxfilt])


def getStokes(pulse):
    S0=pulsePower(pulse)
    S1=(np.abs(pol2[0])**2-np.abs(pol2[1])**2)*np.ones_like(pulse[0])/S0
    S2=2*np.real(pol2[0]*np.conj(pol2[1]))*np.ones_like(pulse[0])/S0
    S3=-2*np.imag(pol2[0]*np.conj(pol2[1]))*np.ones_like(pulse[0])/S0
    
    return np.array([S0/np.max(S0),S1,S2,S3])


###### Define plot parameters #####
    
global gifsize
gifsize=75


############### Define constants #######################
c=3e8 #Speed of light

rounding=3 #Number of decimals we want to round to


############### Define simulation parameters ###############
global nt
nt=2024*100*4 #FFT points

print("   ")
print("For the FFT, we will use a total of nt="+str(nt)+' points.')
print("   ")

global step_num
step_num=round(100) #No. of z-steps

dispersion_on=0#5e8
NL_on=1
Biref_on=0
random.seed(126)


sidebandnumber=1

print("   ")
print("The number of steps in solving the NLSE will be "+str(step_num))
print("   ")
 


############### Fiber parameters #######################
global distance
distance=2000 #Distance in meters
gamma=1*0.001#Gamma in /W/m
alpha=0 #Loss in db/LD
Pcr=1/(gamma*distance)
print("The critical power of the fiber P_cr="+str(Pcr)+'W')

global deltaz
deltaz=distance/step_num



print("   ")
print("The fiber is divided into N="+str(step_num)+' segments.')
print("The length of the fiber is "+str(distance/1e3)+'km')
print("Each step has a length of "+str(np.round(deltaz,rounding))+' m')
print("   ")

################  Spectral components  ############################


#Peak powers of pulses in units of [W]
power1=1.84*Pcr  #Pump corresponding to n=-1
power2=0.10*Pcr  #Signal corresponding to n=0
power3=0.0
power4=0.0
power5=0.0
power6=0.0
power7=0.0
power8=0.0
power9=0.0

powerlist=np.array([power1,power2,power3,power4,power5,power6,power7,power8,power9])
print("   ")
print("The pulse powers are:")
print(str(powerlist)+'[W]')
print("   ")
print("The pulse power relative to P_cr ([dBc]) are:")
print(str(10*np.log10(powerlist/Pcr)))


#Polraization angle
theta1=45 #X/Y pol of pump
theta2= 0 #X/Y pol of signal
theta3=45 
theta4=45 
theta5=45 
theta6=45 
theta7=45 
theta8=45 
theta9=45 


#Phase delay between x and y comp for (phi=90, theta=45 => circular polarization) 
phi1= 0 #90  
phi2= 0 #90  
phi3=90 #90  
phi4=90 #90  
phi5=90 #90  
phi6=90 #90  
phi7=90 #90  
phi8=90 #90  
phi9=90 #90  

#Overall phase delay
delta1=0
delta2=0 
delta3=180 
delta4=180 
delta5=180 
delta6=180 
delta7=180 
delta8=180 
delta9=180 

degrees=pi/180

#Jones vectors
pol1=np.array([np.cos(theta1*degrees),np.sin(theta1*degrees)*np.exp(1j*phi1*degrees)])*np.exp(1j*delta1*degrees)
pol2=np.array([np.cos(theta2*degrees),np.sin(theta2*degrees)*np.exp(1j*phi2*degrees)])*np.exp(1j*delta2*degrees)
pol3=np.array([np.cos(theta3*degrees),np.sin(theta3*degrees)*np.exp(1j*phi3*degrees)])*np.exp(1j*delta3*degrees)
pol4=np.array([np.cos(theta4*degrees),np.sin(theta4*degrees)*np.exp(1j*phi4*degrees)])*np.exp(1j*delta4*degrees)
pol5=np.array([np.cos(theta5*degrees),np.sin(theta5*degrees)*np.exp(1j*phi5*degrees)])*np.exp(1j*delta5*degrees)
pol6=np.array([np.cos(theta6*degrees),np.sin(theta6*degrees)*np.exp(1j*phi6*degrees)])*np.exp(1j*delta6*degrees)
pol7=np.array([np.cos(theta7*degrees),np.sin(theta7*degrees)*np.exp(1j*phi7*degrees)])*np.exp(1j*delta7*degrees)
pol8=np.array([np.cos(theta8*degrees),np.sin(theta8*degrees)*np.exp(1j*phi8*degrees)])*np.exp(1j*delta8*degrees)
pol9=np.array([np.cos(theta9*degrees),np.sin(theta9*degrees)*np.exp(1j*phi9*degrees)])*np.exp(1j*delta9*degrees)



#Field amplitudes of pulses in units of [sqrt(W)]
ampllist=np.sqrt(powerlist)
print("   ")
print("The field amplitudes are:")
print(str(np.round(ampllist,rounding))+'[sqrt(W)]')



#Wavelengths of spectral components:

lambda1=(1550+0.07)*1e-9# Wavelength in m #Pump
lambda2=(1550-0.07)*1e-9# Wavelength in m #Signal

lambda3=1300.0e-9#1272.6e-9 #Wavelength in m
lambda4=1300.0e-9#1272.6e-9 #Wavelength in m
lambda5=1300.0e-9#1272.6e-9 #Wavelength in m
lambda6=1300.0e-9#1272.6e-9 #Wavelength in m
lambda7=1300.0e-9#1272.6e-9 #Wavelength in m
lambda8=1300.0e-9#1272.6e-9 #Wavelength in m
lambda9=1300.5e-9#1273.3e-9 #Wavelength in m

lambdalist=np.array([lambda1,lambda2,lambda3,lambda4,lambda5,lambda6,lambda7,lambda8,lambda9])
print("   ")
print('The wavelengths are')
print(str(np.round(lambdalist*1e9,rounding))+'[nm]')



#Define angular frequencies
omegalist=c*2*pi/lambdalist
print("   ")
print('The angular frequencies are')
print(str(np.round(omegalist/1e12,rounding))+'[2pi*THz]')


wm=np.abs(omegalist[0]+omegalist[1])/2
ws=np.abs(omegalist[0]-omegalist[1])

fm=wm/2/pi
fs=ws/2/pi



#Define actual frequencies
freqlist=omegalist/2/pi
print("   ")
print('The frequencies are')
print(str(np.round((freqlist+fm)/1e12,rounding))+'[THz]')


OmegaDiffMax=np.max(omegalist)-np.min(omegalist)


#Define spectral widths in units of [Hz] of field components 

sw1=0.01e9
sw2=0.01e9  
sw3=0.01e9 
sw4=0.01e9 
sw5=0.01e9 
sw6=0.01e9 
sw7=0.01e9 
sw8=0.01e9 
sw9=0.01e9 

swlist=np.array([sw1,sw2,sw3,sw4,sw5,sw6,sw7,sw8,sw9])
print("   ")
print('The spectral widths are')
print(str(np.round(swlist/1e6,rounding))+'[MHz]')

lwlist=f2lBW(swlist,lambdalist)
print("   ")
print('The wavelength widths are')
print(str(np.round(lwlist*1e12,rounding))+'[pm]')



#Gaussian pulse widths in [s]
Tlist=1/2/pi/swlist
print("   ")
print('The pulse widths are')
print(str(np.round(Tlist*1e9,rounding))+'[ns]')

Awlist=ampllist*np.sqrt(2*pi)*Tlist

print("   ")
print("The spectral amplitudes are:")
print(str(np.round(Awlist*1e6*1e3,rounding))+'[sqrt(uW/THz)]')
print("   ")

print("   ")
print("The spectral peaks are:")
print(str(np.round(Awlist**2*1e12*1e6,rounding))+'[uW/THz]')
print("   ")




t1=0
t2=0
t3=0
t4=0
t5=0
t6=0
t7=0
t8=0
t9=0

tlist=np.array([t1,t2,t3,t4,t5,t6,t7,t8,t9])

print("   ")
print("The starting times for the pulses are")
print(str(np.round(tlist*1e9,rounding))+'[ns]')






#### Determining the dispersion ###

#We do this by opening a separate data file I generated to get the beta_2 parameter for a regular SMF.
colnames=['lspace', 'b2'] 

df=pd.read_csv('lspaceVSbeta2.csv', names=colnames, header=1) 
lspace=df["lspace"]
b2=df["b2"]

lspace=lspace/1e6 #Convert into m
b2=b2*1e-24 #Convert in to s^2/m

wspace=c*2*pi/lspace
    

f3l = interp1d(lspace, b2, kind='cubic') #Functions for obtaining beta2 when given wavelength of angular freq.
f3w = interp1d(wspace, b2, kind='cubic')

lambdaspace=np.linspace(1200e-9,1300e-9,10000)
pp=f3l(lambdaspace)
i0=np.argmin(np.abs(pp))
l0=lambdaspace[i0]

plt.figure()
plt.title('Dispersion vs. Wavelength')
plt.plot(lspace*1e9,b2*1e24)
#plt.plot(lambdalist[0]*1e9,f3l(lambdalist[0])*1e24,'r.')
plt.plot(lambdalist*1e9,f3l(np.array(lambdalist))*1e24,'r.',markersize=12)
plt.xlabel('Wavelength [nm]')
plt.ylabel('beta2 [ps^2/m]')
plt.grid()
plt.axis([1200,1500,-50,5])
#plt.axis([np.min(lambdalist)*1e9-0.5,np.max(lambdalist)*1e9+0.5,np.min(f3l(lambdalist)*1e24)-0.5,np.max(f3l(lambdalist)*1e24)+0.5])
plt.show()

#w1=c*2*pi/wl2
#w2=c*2*pi/wl1

plt.figure()
plt.title('Dispersion vs. Frequency')
plt.plot(wspace/2/pi/1e12,b2*1e24)
plt.plot(wspace/2/pi/1e12,f3w(wspace)*1e24)
plt.plot((omegalist+wm)/2/pi/1e12,f3w(omegalist+wm)*1e24,'r.',markersize=12)
plt.xlabel('Frequency [THz]')
plt.ylabel('beta2 [ps^2/m]')
plt.grid()
plt.axis([np.min((omegalist+wm)/2/pi)/1e12*0.9999,np.max((omegalist+wm)/2/pi)/1e12*1.0001,np.min(f3w(omegalist+wm))*1e24*0.9999,np.max(f3w(omegalist+wm))*1e24*0.9998])
plt.show()



Ld=Tlist**2/np.abs(f3w(omegalist+wm)*dispersion_on)

print("   ")
print("The dispersion lengths for the pulses are")
print(str(np.round(Ld/1e3,rounding))+'[km]')
print("   ")

LNL=1/(gamma*powerlist*NL_on)
print("   ")
print("The NL lengths for the pulses are")
print(str(np.round(LNL/1e3,rounding))+'[km]')
print("   ")




#Define the angular frequencies used for the FFT
mx=np.max((omegalist))
mi=np.min((omegalist))
omega=np.linspace(-20*ws,20*ws,nt) #Defines a list of frequencies centered between the two main frequencies
global freq
freq=omega/2/pi
llambda=c*2*pi/omega

fplot=(freq)/1e12

dw=omega[1]-omega[0]
df=dw/2/pi


print("   ")
print("The list of frequencies extends from "+str(np.min(freq)/1e12)+'[THz] to '+str(np.max(freq)/1e12)+'[THz].')
print("That is a range of "+str((np.max(freq)-np.min(freq))/1e9)+'[GHz]')
print("The frequency resolution is "+str(df/1e6)+'[MHz]')
print("   ")
print("In terms of wavelengths:")
print("The list of wavelengths extends from "+str(np.min(llambda)*1e9)+'[nm] to '+str(np.max(llambda)*1e9)+'[nm].')
print("That is a range of "+str((np.max(llambda)-np.min(llambda))*1e9)+'[nm]')
print("The wavelength resolution is "+str((llambda[0]-llambda[1])*1e15)+'[fm]')
print("   ")

Tmax=1/df
global dt
dt=Tmax/nt
global tau
tau=np.arange(-nt/2,nt/2)*dt

print("   ")
print("The time range extends from "+str(np.min(tau)*1e9)+'[ns] to '+str(np.max(tau)*1e9)+'[ns].')
print("That is a range of "+str((np.max(tau)*1e9-np.min(tau)*1e9))+'[ns]')
print("And just to confirm, Tmax=1/df="+str(Tmax*1e9)+'[ns]')
print("The time resolution is dt="+str(dt*1e12)+'[ps]')

print("  ")
print("The time range is "+str(Tmax*1e9)+'ns')
print("The time resolution is "+str(dt*1e9)+'ns')
print("   ")

print("Generating spectra for each pulse")

### Generate the spectra for the various frequency components ###
#Spec1=GaussianSpectrum(freq,Awlist[0],freqlist[0],swlist[0],0)
#Spec2=GaussianSpectrum(freq,Awlist[1],freqlist[1],swlist[1],0)
#Spec3=GaussianSpectrum(freq,Awlist[2],freqlist[2],swlist[2],0)
#Spec4=GaussianSpectrum(freq,Awlist[3],freqlist[3],swlist[3],0)
#Spec5=GaussianSpectrum(freq,Awlist[4],freqlist[4],swlist[4],0)
#Spec6=GaussianSpectrum(freq,Awlist[5],freqlist[5],swlist[5],0)
#Spec7=GaussianSpectrum(freq,Awlist[6],freqlist[6],swlist[6],0)
#Spec8=GaussianSpectrum(freq,Awlist[7],freqlist[7],swlist[7],0)
#Spec9=GaussianSpectrum(freq,Awlist[8],freqlist[8],swlist[8],0)
#
#print("  ")
#print("Adding up the spectra into a total spectrum")
#Add them all up to a total spectrum
#Spectot=Spec1+Spec2+Spec3+Spec4+Spec5+Spec6+Spec7+Spec8+Spec9
#Spectot0=Spec1+Spec2+Spec3+Spec4+Spec5+Spec6+Spec7+Spec8+Spec9 #Copy of initial spectrum

print("  ")
print("Generating pulses from individual spectra")
### Generate pulses for each spectrum ###
#dasd,pulse1=getPulseFromSpectrum(freq,Spec1)
#dasd,pulse2=getPulseFromSpectrum(freq,Spec2)
#dasd,pulse3=getPulseFromSpectrum(freq,Spec3)
#dasd,pulse4=getPulseFromSpectrum(freq,Spec4)
#dasd,pulse5=getPulseFromSpectrum(freq,Spec5)
#dasd,pulse6=getPulseFromSpectrum(freq,Spec6)
#dasd,pulse7=getPulseFromSpectrum(freq,Spec7)
#dasd,pulse8=getPulseFromSpectrum(freq,Spec8)
#dasd,pulse9=getPulseFromSpectrum(freq,Spec9)

pulse1=GaussianPulse(tau,ampllist[0],tlist[0],Tlist[0],0,pol1)*np.exp(1j*((omegalist[0]-wm)*tau))
pulse2=GaussianPulse(tau,ampllist[1],tlist[1],Tlist[1],0,pol2)*np.exp(1j*((omegalist[1]-wm)*tau))
pulse3=GaussianPulse(tau,ampllist[2],tlist[2],Tlist[2],0,pol3)*np.exp(1j*(-(omegalist[2]-wm)*tau))
pulse4=GaussianPulse(tau,ampllist[3],tlist[3],Tlist[3],0,pol4)*np.exp(1j*(-(omegalist[3]-wm)*tau))
pulse5=GaussianPulse(tau,ampllist[4],tlist[4],Tlist[4],0,pol5)*np.exp(1j*(-(omegalist[4]-wm)*tau))
pulse6=GaussianPulse(tau,ampllist[5],tlist[5],Tlist[5],0,pol6)*np.exp(1j*(-(omegalist[5]-wm)*tau))
pulse7=GaussianPulse(tau,ampllist[6],tlist[6],Tlist[6],0,pol7)*np.exp(1j*(-(omegalist[6]-wm)*tau))
pulse8=GaussianPulse(tau,ampllist[7],tlist[7],Tlist[7],0,pol8)*np.exp(1j*(-(omegalist[7]-wm)*tau))
pulse9=GaussianPulse(tau,ampllist[8],tlist[8],Tlist[8],0,pol9)*np.exp(1j*(-(omegalist[8]-wm)*tau))

print("  ")
print("Adding them up to a total pulse")
### Generate total pulse from total spectrum ###
#dasd,Pulsetot=getPulseFromSpectrum(freq,Spectot)
#dasd,Pulsetot0=getPulseFromSpectrum(freq,Spectot)

Pulsetot =pulse1+pulse2+pulse3+pulse4+pulse5+pulse6+pulse7+pulse8+pulse9
Pulsetot0=pulse1+pulse2+pulse3+pulse4+pulse5+pulse6+pulse7+pulse8+pulse9

vv,Spectot=getSpectrumFromPulse(tau,Pulsetot)
vv,Spectot0=getSpectrumFromPulse(tau,Pulsetot0)


#Pulsetot=Pulsetot*np.exp(-1j*np.angle(Pulsetot))
#Pulsetot0=Pulsetot0*np.exp(-1j*np.angle(Pulsetot0))


E1=pulseEnergy(tau,pulse1)
E2=pulseEnergy(tau,pulse2)
E3=pulseEnergy(tau,pulse3)
E4=pulseEnergy(tau,pulse4)
E5=pulseEnergy(tau,pulse5)
E6=pulseEnergy(tau,pulse6)
E7=pulseEnergy(tau,pulse7)
E8=pulseEnergy(tau,pulse8)
E9=pulseEnergy(tau,pulse9)

Elist=np.array([E1,E2,E3,E4,E5,E6,E7,E8,E9])
print("   ")
print("The energies of the pulses are")
print(str(np.round(Elist*1e9,rounding))+'[nJ]')

Etot=pulseEnergy(tau,Pulsetot0)

print("  ")
print("The total pulse energy is Etot="+str(Etot*1e9)+' [nJ]')
print("Just to check, we also sum the individual energies="+str(np.sum(Elist)*1e9)+' [nJ]')
print(" ")


#plt.figure()
#plt.title('Testing spectrum for individual freq. comp.')
#plt.plot(freq/1e12,np.abs(Spec1)**2*1e12,label='E1='+str(np.round(spectralEnergy(freq,Spec1)*1e9,rounding))+'nJ.')
#plt.plot(freq/1e12,np.abs(Spec2)**2*1e12,label='E2='+str(np.round(spectralEnergy(freq,Spec2)*1e9,rounding))+'nJ.')
#plt.plot(freq/1e12,np.abs(Spec3)**2*1e12,label='E3='+str(np.round(spectralEnergy(freq,Spec3)*1e9,rounding))+'nJ.')
#plt.plot(freq/1e12,np.abs(Spec4)**2*1e12,label='E4='+str(np.round(spectralEnergy(freq,Spec4)*1e9,rounding))+'nJ.')
#plt.plot(freq/1e12,np.abs(Spec5)**2*1e12,label='E5='+str(np.round(spectralEnergy(freq,Spec5)*1e9,rounding))+'nJ.')
#plt.plot(freq/1e12,np.abs(Spec6)**2*1e12,label='E6='+str(np.round(spectralEnergy(freq,Spec6)*1e9,rounding))+'nJ.')
#plt.plot(freq/1e12,np.abs(Spec7)**2*1e12,label='E7='+str(np.round(spectralEnergy(freq,Spec7)*1e9,rounding))+'nJ.')
#plt.plot(freq/1e12,np.abs(Spec8)**2*1e12,label='E8='+str(np.round(spectralEnergy(freq,Spec8)*1e9,rounding))+'nJ.')
#plt.plot(freq/1e12,np.abs(Spec9)**2*1e12,label='E9='+str(np.round(spectralEnergy(freq,Spec9)*1e9,rounding))+'nJ.')
#plt.xlabel('Frequency [THz]')
#plt.ylabel('Power density [W/THz]')
#plt.yscale('log')
#plt.grid()
#plt.axis([np.min(freq/1e12),np.max(freq/1e12),np.min(np.abs(Awlist)**2*1e12)*1e-6,np.max(np.abs(Awlist)**2*1e12)*10])
#plt.legend(frameon=False)
#plt.show()


plt.figure()
plt.title('Testing trace for individual freq component')
plt.plot(tau*1e9,pulsePower(pulse1),label='E1='+str(np.round(pulseEnergy(tau,pulse1)*1e9,rounding))+'nJ.')
plt.plot(tau*1e9,pulsePower(pulse2),label='E2='+str(np.round(pulseEnergy(tau,pulse2)*1e9,rounding))+'nJ.')
plt.plot(tau*1e9,pulsePower(pulse3),label='E3='+str(np.round(pulseEnergy(tau,pulse3)*1e9,rounding))+'nJ.')
plt.plot(tau*1e9,pulsePower(pulse4),label='E4='+str(np.round(pulseEnergy(tau,pulse4)*1e9,rounding))+'nJ.')
plt.plot(tau*1e9,pulsePower(pulse5),label='E5='+str(np.round(pulseEnergy(tau,pulse5)*1e9,rounding))+'nJ.')
plt.plot(tau*1e9,pulsePower(pulse6),label='E6='+str(np.round(pulseEnergy(tau,pulse6)*1e9,rounding))+'nJ.')
plt.plot(tau*1e9,pulsePower(pulse7),label='E7='+str(np.round(pulseEnergy(tau,pulse7)*1e9,rounding))+'nJ.')
plt.plot(tau*1e9,pulsePower(pulse8),label='E8='+str(np.round(pulseEnergy(tau,pulse8)*1e9,rounding))+'nJ.')
plt.plot(tau*1e9,pulsePower(pulse9),label='E9='+str(np.round(pulseEnergy(tau,pulse9)*1e9,rounding))+'nJ.')
plt.xlabel('Time [ns]')
plt.ylabel('Power [W]')
plt.grid()
plt.legend(frameon=False)
plt.show()



################ Define time and frequency arrays ###############





print("Plot input pulse and spectrum") ###############


plt.figure()
plt.title('Total input pulse')
plt.plot(tau*1e9,pulsePower(Pulsetot),label='E='+str(Etot*1e9)+'nJ.')
plt.xlabel('Time [ns]')
plt.ylabel('Power [W]')
plt.grid()
plt.legend(frameon=False)
plt.show()

plt.figure()
plt.title('Phase of Total input pulse')
plt.plot(tau*1e9,getPhase(Pulsetot0[0]),label='$\phi_x$')
plt.plot(tau*1e9,getPhase(Pulsetot0[1]),label='$\phi_y$')
plt.xlabel('Time [ns]')
plt.ylabel('Phase [rad]')
plt.grid()
plt.legend(frameon=False)
plt.show()

cx=getChirp(tau,Pulsetot0[0])
cy=getChirp(tau,Pulsetot0[1])


fig,ax=plt.subplots()
plt.title('Chirp of Total input pulse')
plt.plot(tau*1e9,cx/1e12,label='$c_x$')
plt.plot(tau*1e9,cy/1e12,label='$c_y$')
plt.xlabel('Time [ns]')
plt.ylabel('Local Chirp [THz]')
ax.set_xlim(-2*np.max(Tlist)*1e9,2*np.max(Tlist)*1e9)
plt.grid()
plt.legend(frameon=False)
plt.show()






plt.figure()
plt.title('Zoom on total input pulse')
plt.plot(tau*1e9,pulsePower(Pulsetot0),label='E='+str(Etot*1e9)+'nJ.')
plt.xlabel('Time [ns]')
plt.ylabel('Power [W]')
plt.grid()
plt.axis([-2*Tlist[0]*1e9,2*Tlist[0]*1e9,0,np.max(np.abs(Pulsetot0)**2)*2])
plt.legend(frameon=False)
plt.show()

plt.figure()
plt.title('Total input spectrum')
plt.plot(fplot,spectralPower(Spectot0)*1e12,label='E='+str(np.round(spectralEnergy(freq,Spectot0)*1e9,4))+'nJ.')
plt.xlabel('Frequency [THz]')
plt.ylabel('Power density [W/THz]')
plt.yscale('log')
plt.grid()
#plt.axis([np.min(freq/1e12),np.max(freq/1e12),np.min(np.abs(Awlist)**2*1e12)*1e-6,np.max(np.abs(Awlist)**2*1e12)*10])
plt.legend(frameon=False)
plt.show()


plt.figure()
plt.title('Zoom on total input spectrum')
plt.plot(fplot,spectralPower(Spectot0)*1e12,label='E='+str(np.round(spectralEnergy(freq,Spectot0)*1e9,4))+'nJ.')
plt.xlabel('Frequency [THz]')
plt.ylabel('Power density [W/THz]')
plt.yscale('log')
plt.grid()
plt.axis([omegalist[1]/2/pi/1e12,omegalist[0]/2/pi/1e12, 1e-34,1e0])
#plt.axis([omegalist[1]/1e12/2/pi-50*ws/1e12/2/pi,omegalist[0]/1e12/2/pi+50*ws/1e12/2/pi,np.min(np.abs(Awlist)**2*1e12)*1e-6,np.max(np.abs(Awlist)**2*1e12)*10])
plt.legend(frameon=False)
plt.show()


a,b=getSpectrumFromPulse(tau,Pulsetot0)
c,d=getPulseFromSpectrum(freq,b)


plt.figure()
plt.plot(freq/1e12,np.log10(spectralPower(b)))
#plt.plot(freq/1e12,np.log10(spectralPower(d))-10)
plt.axis([-0.02,0.02,-50,10])
plt.grid()
plt.show()

############## Calculate dispersive phase shift ###############

Pulsetot_NL=Pulsetot0*np.exp(1j*NL_on*gamma*distance*pulsePower(Pulsetot0))
s,Specfinal_NL=getSpectrumFromPulse(tau,Pulsetot_NL)

plt.figure()
plt.title('Expected final spectrum with only NL effect')
plt.plot(freq/1e12,spectralPower(Spectot0)*1e12,label='E='+str(np.round(spectralEnergy(freq,Spectot0)*1e9,4))+'nJ.')
plt.plot(freq/1e12,spectralPower(Specfinal_NL)*1e12,'r-',label='E='+str(np.round(spectralEnergy(freq,Specfinal_NL)*1e9,4))+'nJ.')
plt.xlabel('Frequency [THz]')
plt.ylabel('Power density [W/THz]')
plt.yscale('log')
plt.grid()
plt.axis([omegalist[1]/1e12/2/pi-5*ws/1e12/2/pi,omegalist[0]/1e12/2/pi+5*ws/1e12/2/pi,np.max(np.abs(Spectot)**2*1e12)*1e-11,np.max(np.abs(Spectot)**2*1e12)*10])
plt.legend(frameon=False)
plt.show()

dispersion=np.exp(dispersion_on*0.5*1j*f3w(wm)*(omega)**2*deltaz) 
disptot=np.exp(dispersion_on*0.5*1j*f3w(wm)*(omega)**2*distance)
Specfinal_disp=Spectot0*disptot

plt.figure()
plt.title('Expected final spectrum with only disp effect')
plt.plot(fplot,pulsePower(Spectot0)*1e12+1e-100)
plt.plot(fplot,pulsePower(Specfinal_disp)*1e12+1e-100,'r-')
plt.xlabel('Frequency [THz]')
plt.ylabel('Power density [W/THz]')
plt.yscale('log')
plt.grid()
#plt.axis([np.min(fplot),np.max(fplot),np.max(np.abs(Spectot)**2)*1e-9,np.max(np.abs(Spectot)**2)*10])
plt.legend(frameon=False)
plt.show()

ss,Pulsefinal_disp=getPulseFromSpectrum(freq,Specfinal_disp)



plt.figure()
plt.title('Expected final pulse with only disp effect')
plt.plot(tau*1e9,pulsePower(Pulsetot0),label='E='+str(np.round(pulseEnergy(tau,Pulsetot0)*1e9,rounding))+'nJ.')
plt.plot(tau*1e9,pulsePower(Pulsefinal_disp),'r-',label='E='+str(np.round(pulseEnergy(tau,Pulsefinal_disp)*1e9,rounding))+'nJ.')
plt.xlabel('Time [ns]')
plt.ylabel('Power density [W]')
plt.grid()
#plt.axis([np.min(fplot),np.max(fplot),np.max(np.abs(Spectot)**2)*1e-9,np.max(np.abs(Spectot)**2)*10])
plt.legend(frameon=False)
plt.show()


#Dispersion phase factor evaluated at mean frequency


hhz=NL_on*1j*gamma*deltaz #NL phase factor

boxfilt0=box(freq,omegalist[1]/2/pi+ws/2/pi,np.abs(omegalist[1]-omegalist[0])/10)
boxfiltm1=box(freq,omegalist[1]/2/pi+ws/2/pi,np.abs(omegalist[1]-omegalist[0])/10)



#pulsematrix=np.zeros([step_num+1,len(omega)])*1j
#spectrummatrix=np.zeros([step_num+1,len(omega)])*1j
#extractionmatrix=np.zeros([step_num+1,len(omega)])*1j
#extractionmatrixm1=np.zeros([step_num+1,len(omega)])*1j

#pulsematrix[0,:]=Pulsetot0
#spectrummatrix[0,:]=Spectot0
#extractionmatrix[0,:]=pulse2
#extractionmatrixm1[0,:]=pulse1


temp=(Pulsetot0*np.exp(pulsePower(Pulsetot0)*hhz/2)) #apply initial NL phase factor
for n in range(1,step_num+1):
    #print(n)
  
    s,f_temp=getSpectrumFromPulse(tau,temp) #Get the spectru from the pulse and apply dispersion    
    f_temp=f_temp*dispersion
    
    s,temp=getPulseFromSpectrum(freq,f_temp) #Convert back to a pulse
    #temp=sp.fftpack.fftshift(temp) #Note: Here I have to do an extra FFT shift for some reason. Not sure why, but it works!
    
    bireftheta=random.uniform(-pi,pi)*1.0*Biref_on
    birefphi=random.uniform(-pi/2,pi/2)*1.0*Biref_on
    
     
    temp=temp*np.exp(pulsePower(temp)*hhz) #Apply whole phase factor NL phase factor.  
   
    Rmatrix=np.array([[np.cos(bireftheta),np.sin(bireftheta)*np.exp(1j*birefphi)],[-np.sin(bireftheta)*np.exp(-1j*birefphi),np.cos(bireftheta)]])
    
    temp=np.dot(Rmatrix,temp)
    
    #Save pulses and spectra for visualization later
    dummypulse=temp*np.exp(pulsePower(temp)*hhz/2)
    #pulsematrix[n,:]=dummypulse
    s,dummyspec=getSpectrumFromPulse(tau,dummypulse)    
    #spectrummatrix[n,:]=dummyspec
        
#    dummyspecextraction=dummyspec*boxfilt0
#    dummyspecextractionm1=dummyspec*boxfiltm1
    
#    s,dummypulseextraction=getPulseFromSpectrum(freq,dummyspecextraction)        
#    s,dummypulseextractionm1=getPulseFromSpectrum(freq,dummyspecextractionm1)
#    extractionmatrix[n,:]=dummypulseextraction
#    extractionmatrixm1[n,:]=dummypulseextractionm1
        

    #print(np.round(n/step_num*100.0,4))    
    
    #print("NSLE solution step number "+str(n)+" out of "+str(step_num))
    if np.round(n/step_num*100.0,4)%10==0.0:
        print("Since np.round(n/step_num*100.0,4)%10==0.0 because n="+ str(n)+"we print graphs")        
        #temp=temp*np.exp(np.abs(temp)**2*hhz/2) #Apply final NL phase factor.     
        dummyspecextraction=getSideband(freq,dummyspec,(omegalist[1]/2/pi)-fm,omegalist[0]/2/pi-fm,sidebandnumber)
        dummyspecextractionm1=dummyspec*boxfiltm1
    
        s,dummypulseextraction=getPulseFromSpectrum(freq,dummyspecextraction)        
        s,dummypulseextractionm1=getPulseFromSpectrum(freq,dummyspecextractionm1)
         
        print(Rmatrix)
        
        plt.figure()
        plt.title('Output pulse. z='+str(n*deltaz/1e3)+'km')
        plt.plot(tau*1e9,pulsePower(Pulsetot0), linewidth=5,label='Input')#'E='+str(np.round(pulseEnergy(tau,pulsetot0)*1e9,rounding))+'nJ. E_th= '+str(Etot_th)+'nJ')
        plt.plot(tau*1e9,pulsePower(Pulsefinal_disp),'r-', linewidth=5,label='Only disp')#'E='+str(np.round(pulseEnergy(tau,pulse_final_only_disp)*1e9,rounding))+'nJ. Only disp')
        plt.plot(tau*1e9,pulsePower(dummypulse), linewidth=5,label='Output')#'E='+str(np.round(pulseEnergy(tau,dummypulse)*1e9,rounding))+'nJ. E_th= '+str(Etot_th)+'nJ')
        plt.xlabel('Time [ns]')
        plt.ylabel('Power [W]')
        plt.grid()
        plt.legend(frameon=False)
        plt.show()






        plt.figure()
        plt.plot(tau*1e9,pulsePower(dummypulseextraction))
        plt.grid()
        plt.show()
        
        S0=pulsePower(dummypulseextraction)
        S1=(np.abs(dummypulseextraction[0])**2-np.abs(dummypulseextraction[1])**2)/S0
        S2=2*np.real(dummypulseextraction[0]*np.conj(dummypulseextraction[1]))/S0
        S3=-2*np.imag(dummypulseextraction[0]*np.conj(dummypulseextraction[1]))/S0
        
        plt.figure()
        plt.plot(tau*1e9,S0/np.max(S0),label='S0')
        plt.plot(tau*1e9,S1**2*np.sign(S1),label='S1')
        plt.plot(tau*1e9,S2**2*np.sign(S2),label='S2')
        plt.plot(tau*1e9,S3**2*np.sign(S3),label='S3')
        plt.grid()
        plt.legend()
        plt.axis([-50,50,-1.05,1.05])
        plt.show()
    
#        plt.figure()
#        plt.title('Phase of output pulse. z='+str(n*deltaz/1e3)+'km')
#        plt.plot(tau*1e9,getPhase(Pulsetot0[0]), linewidth=5,label='Initial phase x')#'E='+str(np.round(pulseEnergy(tau,pulsetot0)*1e9,rounding))+'nJ. E_th= '+str(Etot_th)+'nJ')
#        plt.plot(tau*1e9,getPhase(Pulsetot0[1]), linewidth=5,label='Initial phase y')#'E='+str(np.round(pulseEnergy(tau,pulsetot0)*1e9,rounding))+'nJ. E_th= '+str(Etot_th)+'nJ')
#        
#        plt.plot(tau*1e9,getPhase(dummypulse[0]), linewidth=5,label='Output phase x')#'E='+str(np.round(pulseEnergy(tau,dummypulse)*1e9,rounding))+'nJ. E_th= '+str(Etot_th)+'nJ')
#        plt.plot(tau*1e9,getPhase(dummypulse[1]), linewidth=5,label='Output phase y')#'E='+str(np.round(pulseEnergy(tau,dummypulse)*1e9,rounding))+'nJ. E_th= '+str(Etot_th)+'nJ')
#       
#        plt.xlabel('Time [ns]')
#        plt.ylabel('Phase [rad]')
#        plt.grid()
#        #plt.axis([np.min(tau)*1e9,np.max(tau)*1e9,-0.5,0.5])
#        plt.legend(frameon=False)
#        plt.show()
        
#        plt.figure()
#        plt.title('Chirp of output pulse. z='+str(n*deltaz/1e3)+'km')
#        plt.plot(tau*1e9,getChirp(tau,Pulsetot0[0])/1e6, linewidth=5,label='Initial chirp x')#'E='+str(np.round(pulseEnergy(tau,pulsetot0)*1e9,rounding))+'nJ. E_th= '+str(Etot_th)+'nJ')
#        plt.plot(tau*1e9,getChirp(tau,Pulsetot0[1])/1e6, linewidth=5,label='Initial chirp y')#'E='+str(np.round(pulseEnergy(tau,pulsetot0)*1e9,rounding))+'nJ. E_th= '+str(Etot_th)+'nJ')
#
#        plt.plot(tau*1e9,getChirp(tau,dummypulse[0])/1e6, linewidth=5,label='Output chirp x')#'E='+str(np.round(pulseEnergy(tau,dummypulse)*1e9,rounding))+'nJ. E_th= '+str(Etot_th)+'nJ')        
#        plt.plot(tau*1e9,getChirp(tau,dummypulse[1])/1e6, linewidth=5,label='Output chirp y')#'E='+str(np.round(pulseEnergy(tau,dummypulse)*1e9,rounding))+'nJ. E_th= '+str(Etot_th)+'nJ')        
#        plt.xlabel('Time [ns]')
#        plt.ylabel('Local Chirp [MHz]')
#        plt.grid()
#        #ax.set_xlim(-5*np.max(Tlist)*1e9,5*np.max(Tlist)*1e9)
#        plt.axis([-5*np.max(Tlist)*1e9,5*np.max(Tlist)*1e9,-5/2/np.min(Tlist)/1e6,5/2/np.min(Tlist)/1e6])
#        plt.legend(frameon=False)
#        plt.show()
        
        

#        plt.figure()
#        plt.title('Output pulse for m=0. z='+str(n*deltaz/1e3)+'km')
#        plt.plot(tau*1e9,pulsePower(Pulsetot0), linewidth=5,label='Total Input')#'E='+str(np.round(pulseEnergy(tau,pulsetot0)*1e9,rounding))+'nJ. E_th= '+str(Etot_th)+'nJ')
#        plt.plot(tau*1e9,pulsePower(pulse2),'r-', linewidth=5,label='m=0 Input')#'E='+str(np.round(pulseEnergy(tau,pulse_final_only_disp)*1e9,rounding))+'nJ. Only disp')
#        plt.plot(tau*1e9,pulsePower(dummypulseextraction), linewidth=5,label='m=0 Output')#'E='+str(np.round(pulseEnergy(tau,dummypulse)*1e9,rounding))+'nJ. E_th= '+str(Etot_th)+'nJ')
#        plt.xlabel('Time [ns]')
#        plt.ylabel('Power [W]')
#        plt.yscale('log')
#        plt.grid()
#        plt.legend(frameon=False)
#        plt.show()
#        
#        plt.figure()
#        plt.title('Chirp of m=0 pulse. z='+str(n*deltaz/1e3)+'km')
#        plt.plot(tau*1e9,getChirp(tau,pulse2[0])/1e9, linewidth=5,label='Initial chirp x')#'E='+str(np.round(pulseEnergy(tau,pulsetot0)*1e9,rounding))+'nJ. E_th= '+str(Etot_th)+'nJ')
#        plt.plot(tau*1e9,getChirp(tau,pulse2[1])/1e9, linewidth=5,label='Initial chirp y')#'E='+str(np.round(pulseEnergy(tau,pulsetot0)*1e9,rounding))+'nJ. E_th= '+str(Etot_th)+'nJ')
#        
#        plt.plot(tau*1e9,getChirp(tau,dummypulseextraction[0])/1e9, linewidth=5,label='Output x')#'E='+str(np.round(pulseEnergy(tau,dummypulse)*1e9,rounding))+'nJ. E_th= '+str(Etot_th)+'nJ')
#        plt.plot(tau*1e9,getChirp(tau,dummypulseextraction[1])/1e9, linewidth=5,label='Output y')#'E='+str(np.round(pulseEnergy(tau,dummypulse)*1e9,rounding))+'nJ. E_th= '+str(Etot_th)+'nJ')
#
#        plt.xlabel('Time [ns]')
#        plt.ylabel('Local Chirp [GHz]')
#        
#        plt.grid()
#        plt.axis([-5*np.max(Tlist)*1e9,5*np.max(Tlist)*1e9,12,13])
#        
#        #plt.axis([-5*np.max(Tlist)*1e9,5*np.max(Tlist)*1e9,-1/4/np.min(Tlist)/1e6,1/4/np.min(Tlist)/1e6])
#        plt.legend(frameon=False)
#        plt.show()
        
        
#        plt.figure()
#        plt.title('Output pulse for m=0. z='+str(n*deltaz/1e3)+'km')
#        plt.plot(tau*1e9,10*np.log10(np.abs(Pulsetot0)**2/np.max(np.abs(Pulsetot0)**2)), linewidth=5,label='Total Input')#'E='+str(np.round(pulseEnergy(tau,pulsetot0)*1e9,rounding))+'nJ. E_th= '+str(Etot_th)+'nJ')
#        plt.plot(tau*1e9,10*np.log10((np.abs(pulse2)**2)/np.max(np.abs(pulse2)**2)),'r-', linewidth=5,label='m=0 Input')#'E='+str(np.round(pulseEnergy(tau,pulse_final_only_disp)*1e9,rounding))+'nJ. Only disp')
#        plt.plot(tau*1e9,10*np.log10(np.abs(dummypulseextraction)**2/np.max(np.abs(pulse2)**2)), linewidth=5,label='m=0 Output')#'E='+str(np.round(pulseEnergy(tau,dummypulse)*1e9,rounding))+'nJ. E_th= '+str(Etot_th)+'nJ')
#        plt.xlabel('Time [ns]')
#        plt.ylabel('Power rel. to max input [dB]')
#        plt.axis([np.min(tau*1e9),np.max(tau*1e9),-50,50])
#        plt.grid()
#        plt.legend(frameon=False)
#        plt.show()


###
#        plt.figure()
#        plt.title('Output pulse for m=-1. z='+str(n*deltaz/1e3)+'km')
#        plt.plot(tau*1e9,np.abs(Pulsetot0)**2, linewidth=5,label='Total Input')#'E='+str(np.round(pulseEnergy(tau,pulsetot0)*1e9,rounding))+'nJ. E_th= '+str(Etot_th)+'nJ')
#        plt.plot(tau*1e9,np.abs(pulse1)**2,'r-', linewidth=5,label='m=-1 Input')#'E='+str(np.round(pulseEnergy(tau,pulse_final_only_disp)*1e9,rounding))+'nJ. Only disp')
#        plt.plot(tau*1e9,np.abs(dummypulseextractionm1)**2, linewidth=5,label='m=-1 Output')#'E='+str(np.round(pulseEnergy(tau,dummypulse)*1e9,rounding))+'nJ. E_th= '+str(Etot_th)+'nJ')
#        plt.xlabel('Time [ns]')
#        plt.ylabel('Power [W]')
#        plt.yscale('log')
#        plt.grid()
#        plt.axis([np.min(tau)*1e9,np.max(tau)*1e9,1e-30,np.max(np.abs(Pulsetot0)**2)*2])
#        plt.legend(frameon=False)
#        plt.show()
#        
#        plt.figure()
#        plt.title('Output pulse for m=-1. z='+str(n*deltaz/1e3)+'km')
#        plt.plot(tau*1e9,10*np.log10(np.abs(Pulsetot0)**2/np.max(np.abs(Pulsetot0)**2)), linewidth=5,label='Total Input')#'E='+str(np.round(pulseEnergy(tau,pulsetot0)*1e9,rounding))+'nJ. E_th= '+str(Etot_th)+'nJ')
#        plt.plot(tau*1e9,10*np.log10((np.abs(pulse1)**2)/np.max(np.abs(pulse1)**2)),'r-', linewidth=5,label='m=-1 Input')#'E='+str(np.round(pulseEnergy(tau,pulse_final_only_disp)*1e9,rounding))+'nJ. Only disp')
#        plt.plot(tau*1e9,10*np.log10(np.abs(dummypulseextractionm1)**2/np.max(np.abs(pulse1)**2)), linewidth=5,label='m=-1 Output')#'E='+str(np.round(pulseEnergy(tau,dummypulse)*1e9,rounding))+'nJ. E_th= '+str(Etot_th)+'nJ')
#        plt.xlabel('Time [ns]')
#        plt.ylabel('Power rel. to max input [dB]')
#        plt.axis([np.min(tau*1e9),np.max(tau*1e9),-50,50])
#        plt.grid()
#        plt.legend(frameon=False)
#        plt.show()
####

        plt.figure()
        plt.title('Output Spectrum. z='+str(n*deltaz/1e3)+'km')
        plt.plot(fplot,spectralPower(Spectot0)*1e12, linewidth=5,label='Input')#'E='+str(np.round(spectralEnergy(freq,Spectot0)*1e9,4))+'nJ. E_th= '+str(E1_th+E2_th)+'nJ')
        #plt.plot(freq/1e12,np.abs(Specfinal_NL)**2*1e12,'r-', linewidth=5,label='Only NL')#label='E='+str(np.round(spectralEnergy(freq,specfinal)*1e9,4))+'nJ. E_th= '+str(E1_th+E2_th)+'nJ. Only NL')
        plt.plot(fplot,spectralPower(dummyspec)*1e12, linewidth=5,label='Output')#'E='+str(np.round(spectralEnergy(freq,dummyspec)*1e9,4))+'nJ. E_th= '+str(E1_th+E2_th)+'nJ')
        plt.plot(fplot,spectralPower(dummyspecextraction)*1e12, linewidth=5,label='n='+str(sidebandnumber))#'E='+str(np.round(spectralEnergy(freq,dummyspec)*1e9,4))+'nJ. E_th= '+str(E1_th+E2_th)+'nJ')
        plt.axis([fplot[int(len(fplot)/2-len(fplot)/10)],fplot[int(len(fplot)/2+len(fplot)/10)],1e-30,1e0])
        
        plt.plot((omegalist[0]+wm)/2/pi/1e12,1e-10,'r.',markersize=25)
        plt.plot((omegalist[1]+wm)/2/pi/1e12,1e-10,'m.',markersize=25)
        
        plt.xlabel('Frequency [THz]')
        plt.ylabel('Power density [W/THz]')
        plt.yscale('log')
        plt.grid()
        #plt.axis([fm/1e12-0.0005,fm/1e12+0.0005,1e-35,1e13])
        #plt.axis([omegalist[1]/1e12/2/pi-5*ws/2/pi/1e12,omegalist[0]/1e12/2/pi+5*ws/2/pi/1e12,np.max(np.abs(Spectot0)**2*1e12)*1e-15,np.max(np.abs(Spectot0)**2*1e12)*10])
        #plt.axis([omegalist[0]/2/pi/1e12-0.0005,omegalist[0]/2/pi/1e12+0.0005,1e-40,1e8])
        #plt.axis([np.min(freq/1e12),np.max(freq/1e12),np.max(np.abs(Spectot0)**2*1e12)*1e-9,np.max(np.abs(Spectot0)**2*1e12)*10])
        plt.legend(frameon=False)
        plt.show()
        print(np.round(n/step_num*100.0,4))
        
    
    


pulsetot=temp*np.exp(pulsePower(temp)*hhz/2) #Apply final phase factor
s,Spectot=getSpectrumFromPulse(tau,pulsetot)

plt.figure()
#plt.plot(tau*1e9,np.abs(pulsematrix[10,:])**2)
plt.plot(tau*1e9,pulsePower(pulsetot))

plt.figure()
plt.title('Output pulse. z='+str(distance/1e3)+'km')
plt.plot(tau*1e9,pulsePower(Pulsetot0), linewidth=5,label='Input')#label='E='+str(np.round(pulseEnergy(tau,pulsetot0)*1e9,rounding))+'nJ. E_th= '+str(Etot_th)+'nJ')
plt.plot(tau*1e9,pulsePower(Pulsefinal_disp),'r-', linewidth=5,label='only disp')#label='E='+str(np.round(pulseEnergy(tau,pulse_final_only_disp)*1e9,rounding))+'nJ.')
plt.plot(tau*1e9,pulsePower(pulsetot), linewidth=5,label='Output')#='E='+str(np.round(pulseEnergy(tau,pulsetot)*1e9,rounding))+'nJ. E_th= '+str(Etot_th)+'nJ')
plt.xlabel('Time [ns]')
plt.ylabel('Power [W]')
plt.grid()
plt.legend(frameon=False)
plt.show()




#plt.figure()
#plt.title('Chirp of Output pulse. z='+str(distance/1e3)+'km')
#plt.plot(tau*1e9,getChirp(tau,Pulsetot0)/1e12, linewidth=5,label='Input')#label='E='+str(np.round(pulseEnergy(tau,pulsetot0)*1e9,rounding))+'nJ. E_th= '+str(Etot_th)+'nJ')
#plt.plot(tau*1e9,getChirp(tau,Pulsefinal_disp)/1e12,'r-', linewidth=5,label='only disp')#label='E='+str(np.round(pulseEnergy(tau,pulse_final_only_disp)*1e9,rounding))+'nJ.')
#plt.plot(tau*1e9,getChirp(tau,pulsetot)/1e12, linewidth=5,label='Output')#='E='+str(np.round(pulseEnergy(tau,pulsetot)*1e9,rounding))+'nJ. E_th= '+str(Etot_th)+'nJ')
#plt.xlabel('Time [ns]')
#plt.ylabel('Chirp [THz]')
#plt.grid()
#plt.legend(frameon=False)
#plt.show()



plt.figure()
plt.title('Output Spectrum. z='+str(distance/1e3)+'km')
plt.plot(freq/1e12,spectralPower(Spectot0)*1e12, linewidth=5,label='Input')#'E='+str(np.round(spectralEnergy(freq,Spectot0)*1e9,4))+'nJ. E_th= '+str(E1_th+E2_th)+'nJ')
plt.plot(freq/1e12,spectralPower(Specfinal_NL)*1e12, 'r-',linewidth=5,label='Only NL')#'E='+str(np.round(spectralEnergy(freq,specfinal)*1e9,4))+'nJ. E_th= '+str(E1_th+E2_th)+'nJ')
plt.plot(freq/1e12,spectralPower(Spectot)*1e12, linewidth=5,label='Output')#'E='+str(np.round(spectralEnergy(freq,Spectot)*1e9,4))+'nJ. E_th= '+str(E1_th+E2_th)+'nJ')
plt.plot(freq/1e12,boxfilt0, linewidth=5,label='boxfilt0')#'E='+str(np.round(spectralEnergy(freq,Spectot)*1e9,4))+'nJ. E_th= '+str(E1_th+E2_th)+'nJ')
plt.plot(freq/1e12,boxfiltm1, linewidth=5,label='boxfiltm1')#'E='+str(np.round(spectralEnergy(freq,Spectot)*1e9,4))+'nJ. E_th= '+str(E1_th+E2_th)+'nJ')
plt.xlabel('Frequency [THz]')
plt.ylabel('Power density [W/THz]')
plt.yscale('log')
plt.grid()
plt.axis([-0.1,0.1,np.max(np.abs(Spectot0)**2*1e12)*1e-9,np.max(np.abs(Spectot0)**2*1e12)*10])
plt.legend(frameon=False)
plt.show()

minus2Spectrum=getSideband(freq,Spectot,omegalist[1]/2/pi-fm,omegalist[0]/2/pi-fm,sidebandnumber)

tt,minus2Pulse=getPulseFromSpectrum(freq,minus2Spectrum)




plt.figure()
plt.plot(tau*1e9,pulsePower(minus2Pulse))
plt.grid()
plt.show()

S0=pulsePower(minus2Pulse)
S1=(np.abs(minus2Pulse[0])**2-np.abs(minus2Pulse[1])**2)/S0
S2=2*np.real(minus2Pulse[0]*np.conj(minus2Pulse[1]))/S0
S3=-2*np.imag(minus2Pulse[0]*np.conj(minus2Pulse[1]))/S0

fig,ax=plt.subplots()
plt.title('Output Pol. for n='+str(sidebandnumber)+' sideband (Numerical)')
plt.plot(tau*1e9,pulsePower(pulse1)/np.max(pulsePower(pulse1)),'m--',label='Input Pulse')
plt.plot(tau*1e9,S0/np.max(S0),'--',label='S0')
plt.plot(tau*1e9,S1**2*np.sign(S1),'--',label='S1')
plt.plot(tau*1e9,S2**2*np.sign(S2),'--',label='S2')
plt.plot(tau*1e9,S3**2*np.sign(S3),'--',label='S3')
plt.grid()
plt.legend(loc='upper right')
ax.legend(loc='upper center', bbox_to_anchor=(1.25, 0.8),
          ncol=1, fancybox=True, shadow=True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.axis([-50,50,-1.05,1.05])
plt.show()


S1=(np.abs(pol1[0])**2-np.abs(pol1[1])**2)*np.ones_like(pulse1[0])
S2=2*np.real(pol1[0]*np.conj(pol1[1]))*np.ones_like(pulse1[0])
S3=-2*np.imag(pol1[0]*np.conj(pol1[1]))*np.ones_like(pulse1[0])

plt.figure()
plt.title('Input Polarization for signal (n=-1)')
plt.plot(tau*1e9,pulsePower(pulse1)/np.max(pulsePower(pulse1)),label='S0')
plt.plot(tau*1e9,np.abs(S1)**2*np.sign(S1),label='S1')
plt.plot(tau*1e9,np.abs(S2)**2*np.sign(S2),label='S2')
plt.plot(tau*1e9,np.abs(S3)**2*np.sign(S3),label='S3')
plt.grid()
plt.legend()
plt.axis([-50,50,-1.05,1.05])
plt.show()



S1=(np.abs(pol2[0])**2-np.abs(pol2[1])**2)*np.ones_like(pulse2[0])
S2=2*np.real(pol2[0]*np.conj(pol2[1]))*np.ones_like(pulse2[0])
S3=-2*np.imag(pol2[0]*np.conj(pol2[1]))*np.ones_like(pulse2[0])

plt.figure()
plt.title('Input Polarization for pump (n=0)')
plt.plot(tau*1e9,pulsePower(pulse2)/np.max(pulsePower(pulse2)),label='S0')
plt.plot(tau*1e9,np.abs(S1)**2*np.sign(S1),label='S1')
plt.plot(tau*1e9,np.abs(S2)**2*np.sign(S2),label='S2')
plt.plot(tau*1e9,np.abs(S3)**2*np.sign(S3),label='S3')
plt.grid()
plt.legend()
plt.axis([-50,50,-1.05,1.05])
plt.show()

#print("  ")
#print("Begin creating animation of the evolution of the spectrum")
#print("  ")
#@gif.frame
#def plot(i,Spectot0,Specfinal_NL,spectrummatrix):
#    
#    
#    plt.figure(figsize=(10, 6), dpi=gifsize)#
#    plt.title('z = '+str(np.round(i/10*distance/1e3,rounding))+'km')
#    
#    plt.plot(freq/1e12, np.abs(spectrummatrix[i*10,:])**2*1e12+1e-100,'r-',label='Output')    
#    plt.plot(freq/1e12, np.abs(Spectot0)**2*1e12+1e-100,'b-',alpha=0.7,label='Input')
#    #plt.plot(freq/1e12, np.abs(Specfinal_NL)**2*1e12, 'r-',label='Only NL')
#    plt.xlabel('Frequency [THz]')
#    plt.ylabel('Power density [W/THz]')
#    plt.yscale('log')
#    plt.grid()
#    #plt.axis([omegalist[1]/1e12/2/pi-5*ws/2/pi/1e12,omegalist[0]/1e12/2/pi+5*ws/2/pi/1e12,np.max(np.abs(Spectot0)**2*1e12)*1e-15,np.max(np.abs(Spectot0)**2*1e12)*10])
#    
#    #plt.axis([0.072,0.076,np.max(np.abs(Spectot0)**2*1e12)*1e-15,np.max(np.abs(Spectot0)**2*1e12)*10])
#    plt.legend(loc="upper right", bbox_to_anchor=(0.8,0.7),frameon=False)
#    #plt.show()

#frames = []
#print (frames)
#for i in range(0,11):
#    frame = plot(i,Spectot0,Specfinal_NL,spectrummatrix)
#    frames.append(frame)
#    print(np.round(i/(11)*100,rounding))
#
#frame = plot(10,Spectot0,Specfinal_NL,spectrummatrix)
#frames.append(frame)
#
#gif.save(frames, "C:\\Users\\Bruger\\Dropbox\\Canada\\PhD\\Research Questions\\High Ext pulses\\Resolution enhancement\\EvolvingSpectrum.gif", duration=700)
#
#print("  ")
#print("Finished the animation of the spectrum")
#print("  ")
#
#print("  ")
#print("Begin creating animation of the evolution of the pulse")
#print("  ")
#
#@gif.frame
#def plot2(i,pulsetot0,Pulsefinal_disp,pulsematrix):
#    
#    
#    plt.figure(figsize=(10, 6), dpi=gifsize)#
#    plt.title('z = '+str(np.round(i/10*distance/1e3,rounding))+'km')
#    plt.plot(tau*1e9, np.abs(pulsetot0)**2+1e-100,label='Input')
#    plt.plot(tau*1e9, np.abs(Pulsefinal_disp)**2+1e-100, 'r-',label='Only Disp')
#    plt.plot(tau*1e9, np.abs(pulsematrix[i*10,:])**2+1e-100,label='Output')    
#    
#    plt.xlabel('Time [ns]')
#    plt.ylabel('Power [W]')
#    plt.axis([-1,1,0,10*np.max(np.abs(Pulsetot0)**2)])
#    
#    #plt.axis([-10*np.max(Tlist)*1e9,10*np.max(Tlist)*1e9,0,10*np.max(np.abs(Pulsetot0)**2)])
#    plt.legend(loc="upper right", bbox_to_anchor=(0.8,0.7),frameon=False)
#    #plt.show()
#
#frames = []
#print (frames)
#for i in range(0,11):
#    frame = plot2(i,Pulsetot0,Pulsefinal_disp,pulsematrix)
#    frames.append(frame)
#    print(np.round(i/(11)*100,rounding))
#
#frame = plot2(10,Pulsetot0,Pulsefinal_disp,pulsematrix)
#frames.append(frame)
#
#gif.save(frames, "C:\\Users\\Bruger\\Dropbox\\Canada\\PhD\\Research Questions\\High Ext pulses\\Resolution enhancement\\EvolvingPulse.gif", duration=700)
#
#print("  ")
#print("Finished the animation of the pulse")
#print("  ")
#
#
#print("  ")
#print("Begin creating animation of the extracted pulse")
#print("  ")
#
#
#
#@gif.frame
#def plot3(i,pulse2,extractedPulses):
#    
#    
#    
#    plt.figure(figsize=(10, 6), dpi=gifsize)#
#    plt.title('z = '+str(np.round(i/10*distance/1e3,rounding))+'km')
#    plt.plot(tau*1e9, np.abs(pulse2)**2+1e-100,label='Input')
#    plt.plot(tau*1e9, np.abs(extractedPulses[i*10,:])**2+1e-100,label='Output. Ampl = +'+str(np.round( 10*np.log10(np.max(np.abs(extractedPulses[i*10,:])**2)/np.max(np.abs(pulse2)**2) ),rounding)) +'dB')   
#    
#    plt.xlabel('Time [ns]')
#    plt.ylabel('Power [W]')
#    plt.axis([np.min(tau)*1e9,np.max(tau)*1e9,1e-6,10*np.max(np.abs(extractedPulses[100,:])**2)])
#    plt.yscale('log')
#    plt.grid()
#    plt.legend(loc="upper right", bbox_to_anchor=(0.8,0.7),frameon=False)
#    #plt.show()
#
#extractedSpectra=np.multiply(spectrummatrix,boxfilt0)
#extractedPulses=np.zeros_like(extractedSpectra)
#for i in range(0,step_num+1):
#    #print(i)
#    ss, pulse=getPulseFromSpectrum(freq,extractedSpectra[i,:])
#    extractedPulses[i,:]=pulse
#
#
#frames = []
#print (frames)
#for i in range(0,11):
#    frame = plot3(i,pulse2,extractedPulses)
#    frames.append(frame)
#    print(np.round(i/(11)*100,rounding))
#
#frame = plot3(10,pulse2,extractedPulses)
#frames.append(frame)
#
#gif.save(frames, "C:\\Users\\Bruger\\Dropbox\\Canada\\PhD\\Research Questions\\High Ext pulses\\Resolution enhancement\\Evolving_m=0_Pulse.gif", duration=700)
#
#print("  ")
#print("Finished the animation of the extracted pulse")
#print("  ")
#
#frames=[]
#
#print("  ")
#print("Begin making surface plots of spectra and pulses")
#print("  ")
#
#
#fig, ax = plt.subplots()
#ax.set_title('Spectrum Evolution')
#x = fplot
#y = np.linspace(0,step_num*deltaz,step_num)/1e3  
#X, Y = np.meshgrid(x, y)
#Z=np.abs(spectrummatrix)**2*1e12+1e-100
#Z = np.log10(Z[:-1, :])
#Z[Z<-30]=-30
#surf=ax.contourf(X, Y, Z)
#ax.set_xlabel('Frequency [THz]')
#ax.set_ylabel('Distance [km]')
#ax.set_xlim(np.min(fplot),np.max(fplot))
#cbar=fig.colorbar(surf, ax=ax) 
#cbar.set_label("Intensity [W/THz]" )
#plt.savefig('SpectrumEvolution.pdf',bbox_inches='tight')
#plt.show()
#
#
#fig, ax = plt.subplots()
#ax.set_title('Pulse Evolution for m=0')
#x = tau*1e9
#y = np.linspace(0,step_num*deltaz,step_num)/1e3  
#X, Y = np.meshgrid(x, y)
#Z=(np.abs(extractedPulses)**2+1e-100)/(np.max(np.abs(extractedPulses[0,:])**2))
#Z = 10*np.log10(Z[:-1, :])
#Z[Z<-30]=-30
#surf=ax.contourf(X, Y, Z)
#ax.set_xlabel('Time [ns]')
#ax.set_ylabel('Distance [km]')
#ax.set_xlim(-40, 40)
#cbar=fig.colorbar(surf, ax=ax) 
#cbar.set_label("Amplification [dB]" )
#plt.savefig('PulseEvolution.pdf',bbox_inches='tight')
#plt.show()
#
#
#chirpzoom=2**3
#
#fig, ax = plt.subplots()
#ax.set_title('Pulse Chirp')
#x = tau[int(nt/2-nt/chirpzoom):int(nt/2+nt/chirpzoom)]*1e9
#y = np.linspace(0,step_num*deltaz,step_num)/1e3  
#X, Y = np.meshgrid(x, y)
#
#chirpmatrix=np.zeros_like(X)
#
#
#for i in range(0,len(pulsematrix[:,1])-1):
#    line=pulsematrix[i,int(nt/2-nt/chirpzoom):int(nt/2+nt/chirpzoom)]
#    c=getChirp(tau,line)
#    chirpmatrix[i,:]=c-np.mean(c[0:1000])
#    i=i+1
#
#Z=chirpmatrix/1e6
##Z =(Z[:-1, :])
##Z[Z<-300]=-300
#surf=ax.contourf(X, Y, Z,cmap='RdBu')#levels=np.linspace(-5/2*np.min(Tlist)/1e6,5/2*np.min(Tlist)/1e6,11)
#ax.set_xlabel('Time [ns]')
#ax.set_ylabel('Distance [km]')
##ax.set_xlim(-5*np.max(Tlist)*1e9,5*np.max(Tlist)*1e9)
#cbar=fig.colorbar(surf, ax=ax) 
#cbar.set_label("Chirp [MHz]" )
##plt.savefig('PulseChirp_wholePulse.pdf',bbox_inches='tight')
#plt.show()
#
#
#fig, ax = plt.subplots()
#ax.set_title('Chirp of m=0 pulse')
#x = tau[int(nt/2-nt/chirpzoom):int(nt/2+nt/chirpzoom)]*1e9
#y = np.linspace(0,step_num*deltaz,step_num)/1e3  
#X, Y = np.meshgrid(x, y)
#
#chirpmatrix=np.zeros_like(X)
#
#
#for i in range(0,len(extractedPulses[:,1])-1):
#    line=extractedPulses[i,int(nt/2-nt/chirpzoom):int(nt/2+nt/chirpzoom)]
#    c=getChirp(tau,line)
#    chirpmatrix[i,:]=c-np.mean(c[0:1000])
#    i=i+1
#
#Z=chirpmatrix/1e6
##Z =(Z[:-1, :])
##Z[Z<-300]=-300
#surf=ax.contourf(X, Y, Z,cmap='RdBu')#levels=np.linspace(-5/2*np.min(Tlist)/1e6,5/2*np.min(Tlist)/1e6,11)
#ax.set_xlabel('Time [ns]')
#ax.set_ylabel('Distance [km]')
##ax.set_xlim(-5*np.max(Tlist)*1e9,5*np.max(Tlist)*1e9)
#cbar=fig.colorbar(surf, ax=ax) 
#cbar.set_label("Chirp [MHz]" )
##plt.savefig('PulseChirp_wholePulse.pdf',bbox_inches='tight')
#plt.show()
#
#
#bb2=np.array([11.2,5.6, 0,-0.112,-0.56,-1.11,-2.8,-5.6,-11.2])
#dBP=np.array([  4,   4,20,    20,   24,   32,  40,  50, 4])
#
#plt.figure()
#plt.title('Max amp. vs. disp')
#plt.plot(bb2,dBP,'b.',markersize=15)
#plt.xlabel('beta2 [ps^2/m]')
#plt.ylabel('Max ampl of m=0 [dB]')
#plt.grid()
#plt.show()
#
#
