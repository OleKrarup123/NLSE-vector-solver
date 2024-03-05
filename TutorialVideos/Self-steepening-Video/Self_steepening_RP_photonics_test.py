# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 13:48:49 2024

@author: okrarup
"""

from ssfm_functions import *

if __name__ == "__main__":

    os.chdir(os.path.realpath(os.path.dirname(__file__)))

    Trange=1.6e-12
    N = 2 ** 10  # Number of points
    dt = Trange/N  # Time resolution [s]

    centerFreq_test = FREQ_CENTER_C_BAND_HZ*2
    centerWavelength = freq_to_wavelength(centerFreq_test)  # laser wl in m

    timeFreq_test = TimeFreq(N, dt, centerFreq_test)

    beta_list = [-1e4*1e-30]#[]#[0, -24e-39]  # Dispersion in units of s^(entry+2)/m

    fiber_diameter = 9e-6  # m
    n2_silica = 30e-21  # m**2/W

    #
    gamma_test = get_gamma_from_fiber_params(centerWavelength,n2_silica,fiber_diameter)

    #  Initialize fibers
    alpha_test = 0  # 0 0.22/1e3


    number_of_steps=2**13
    testDuration = 100e-15
    length_test =20* np.pi/2*testDuration**2/np.abs(beta_list[0])
    spanloss = alpha_test * length_test

    fiber_test = FiberSpan(
        length_test,
        number_of_steps,
        gamma_test,
        beta_list,
        alpha_test,
        use_self_steepening=True)

    fiber_list = [fiber_test]
    fiber_link = FiberLink(fiber_list)

    # Set up signal
    test_FFT_tol=1e-2
    testTimeOffset = 0  # Time offset
    testFreqOffset = 0  # Freq offset from center frequency

    testChirp = 0
    testPulseType = "sech"
    testOrder = 1
    testNoiseAmplitude = 0

    testAmplitude = 2*np.sqrt(np.abs(beta_list[0])/gamma_test/testDuration**2) #np.sqrt(1e-9 /(testDuration))  # Amplitude in units of sqrt(W)

    testInputSignal = InputSignal(
        timeFreq_test,
        testAmplitude,
        testDuration,
        testTimeOffset,
        testFreqOffset,
        testChirp,
        testPulseType,
        testOrder,
        testNoiseAmplitude,
        FFT_tol=test_FFT_tol
    )
    expName = "Self_steepening_test"

    # Run SSFM
    ssfm_result_list = SSFM(
        fiber_link,
        testInputSignal,
        showProgressFlag=True,
        experimentName=expName,
        FFT_tol=test_FFT_tol
    )

    nrange = 200
    dB_cutoff = -60

    plot_everything_about_result(ssfm_result_list, nrange, dB_cutoff,nrange, dB_cutoff)
    #plot_first_and_last_spectrum(ssfm_result_list, nrange, dB_cutoff)
    #plot_first_and_last_pulse(ssfm_result_list, nrange, dB_cutoff)