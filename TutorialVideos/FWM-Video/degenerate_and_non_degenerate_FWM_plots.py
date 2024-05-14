# -*- coding: utf-8 -*-
"""
Created on Sat May 11 19:57:45 2024

@author: okrarup
"""

import numpy as np
BETA2_AT_1550_NM_TYPICAL_SMF_S2_PER_M = -23e-27
import matplotlib.pyplot as plt

def Bu_power(z,gamma,P,delta_beta):
    const = (delta_beta-gamma*P)/2
    if const == 0.0:
        return gamma**2*P**3*z**2
    else:
        return gamma**2*P**3*(np.sin(const*z)/const)**2

def gain_coeff(delta_beta,gamma,P_tot):
    NL_factor = gamma*P_tot
    return np.sqrt(NL_factor**2- (NL_factor+delta_beta)**2/4 )

def gain_coeff_wa_equals_wb(delta_beta,gamma,P_tot):
    NL_factor = gamma*P_tot
    return np.sqrt(NL_factor**2- (2*NL_factor+delta_beta)**2/4 )

if __name__ == "__main__":


    #Plot for degenerate FWM
    delta_beta_list = [-1e-6,0,1e-6] #1/m
    z=np.linspace(0,4.5e6,1000)
    P = 1e-3#W
    gamma=1e-3 #/W/m

    fig,ax=plt.subplots(dpi=1000)
    ax.set_title(f"$\gamma$P={gamma*P/1e-6}[1/$\mu$m]")
    for delta_beta in delta_beta_list:
        y=Bu_power(z,gamma,P,delta_beta)

        ax.plot(z/1e6,y,label=f'$\Delta\\beta$={delta_beta/1e-6}[1/$\mu$m]')

    ax.set_xlabel(f'z [$\mu$m]')
    ax.set_ylabel('$B_u$ [W]')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=3,
        fancybox=True,
        shadow=True,
    )
    plt.show()

    #Gain for non-degenerate FWM for wa != wb
    delta_beta_array = np.linspace(-15,10,1000)
    P_tot_list = np.array([1e3,2e3,3e3])
    fig,ax=plt.subplots(dpi=1000)
    for P_tot in P_tot_list:
        y=gain_coeff(delta_beta_array,gamma,P_tot)

        ax.plot(delta_beta_array,y,label=f'$\gamma$P={gamma*P_tot }[1/m]')

    ax.set_xlabel(f'$\Delta\\beta$ [1/m]')
    ax.set_ylabel('gain [1/m]')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=4,
        fancybox=True,
        shadow=True,
    )
    plt.show()

    #Gain for non-degenerate FWM for wa = wb
    delta_beta_array = np.linspace(-15,10,1000)
    P_tot_list = np.array([1e3,2e3,3e3])
    fig,ax=plt.subplots(dpi=1000)
    for P_tot in P_tot_list:
        y=gain_coeff_wa_equals_wb(delta_beta_array,gamma,P_tot)

        ax.plot(delta_beta_array,y,label=f'$\gamma$P={gamma*P_tot }[1/m]')

    ax.set_xlabel(f'$\Delta\\beta$ [1/m]')
    ax.set_ylabel('gain [1/m]')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=4,
        fancybox=True,
        shadow=True,
    )
    plt.show()


    #Gain versus Delta omega for non-deenerate FWM for wa=wb
    delta_omega_array = np.linspace(-100e9,100e9,1000)

    P_tot_list = np.array([10e-3,20e-3,30e-3])
    fig,ax=plt.subplots(dpi=1000)
    for P_tot in P_tot_list:
        y=gain_coeff_wa_equals_wb(delta_omega_array**2*BETA2_AT_1550_NM_TYPICAL_SMF_S2_PER_M,gamma,P_tot)

        ax.plot(delta_omega_array/1e9,y*1e3,label=f'$\gamma$P={gamma*P_tot }[1/m]')

    ax.set_xlabel(f'$\Delta\omega$ [GHz]')
    ax.set_ylabel('gain [1/km]')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=4,
        fancybox=True,
        shadow=True,
    )
    plt.show()




