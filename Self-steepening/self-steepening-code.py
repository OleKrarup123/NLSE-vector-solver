# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 14:26:22 2024

@author: okrarup
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 10:23:48 2022

@author: okrarup
"""


# ********Note on sign convention********
#
# This code uses the exp(-1j*omega*t) sign convention because
# exp(1j(beta*z-omega*t)) represents a plane wave propagating in the
# positive z-direction.


import os
from typing import TextIO
from datetime import datetime
import numpy as np
import numpy.typing as npt
from scipy import signal
from scipy.fftpack import fft, ifft, fftshift, ifftshift, fftfreq
from scipy.constants import pi, c, h
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.legend import LineCollection
from matplotlib.animation import FuncAnimation, PillowWriter
LIGHTSPEED_M_PER_S = c
PLANCKCONST_J_PER_HZ = h

# Characteristic frequencies of L-band
FREQ_MIN_L_BAND_HZ = 186.05 * 1e12
FREQ_MAX_L_BAND_HZ = 190.875 * 1e12
FREQ_CENTER_L_BAND_HZ = (FREQ_MIN_L_BAND_HZ + FREQ_MAX_L_BAND_HZ) / 2
FREQ_WIDTH_L_BAND_HZ = FREQ_MAX_L_BAND_HZ - FREQ_MIN_L_BAND_HZ

# Characteristic frequencies of C-band
FREQ_MIN_C_BAND_HZ = 191.275 * 1e12
FREQ_MAX_C_BAND_HZ = 196.15 * 1e12
FREQ_WIDTH_C_BAND_HZ = FREQ_MAX_C_BAND_HZ - FREQ_MIN_C_BAND_HZ
FREQ_CENTER_C_BAND_HZ = (FREQ_MIN_C_BAND_HZ + FREQ_MAX_C_BAND_HZ) / 2


def get_freq_range_from_time(time_s: npt.NDArray[float]
                             ) -> npt.NDArray[float]:
    """
    Calculate frequency range for spectrum based on time basis.

    When plotting a discretized pulse signal as a function of time,
    a time range is needed. To plot the spectrum of the pulse, one
    can compute the FFT and plot it versus the frequency range
    calculated by this function

    Parameters
    ----------
    time_s : npt.NDArray[float]
        Time range in seconds.

    Returns
    -------
    npt.NDArray[float]
        Frequency range in Hz.

    """
    return fftshift(fftfreq(len(time_s), d=time_s[1] - time_s[0]))


def get_phase(pulse: npt.NDArray[complex]) -> npt.NDArray[float]:
    """
    Gets the phase of the pulse from its complex angle

    Calcualte phase by getting the complex angle of the pulse,
    unwrapping it and centering on middle entry.

    Parameters
    ----------
    pulse : npt.NDArray[complex]
        Complex electric field envelope in time domain.

    Returns
    -------
    phi : npt.NDArray[float]
        Phase of pulse at every instance in radians.

    """

    phi = np.unwrap(np.angle(pulse))  # Get phase starting from 1st entry
    phi = phi - phi[int(len(phi) / 2)]  # Center phase on middle entry
    return phi


def get_chirp(time_s: npt.NDArray[float],
              pulse: npt.NDArray[complex]) -> npt.NDArray[float]:
    """
    Get local chirp at every instance of pulse

    Calculate local chirp as the (negative) time derivative of the local phase

    Parameters
    ----------
    time_s : npt.NDArray[float]
        Time range in seconds.
    pulse : npt.NDArray[complex]
        Complex electric field envelope in time domain.

    Returns
    -------
    chirp : npt.NDArray[float]
          Chirp in Hz at every instant.

    """

    phi = get_phase(pulse)
    # Change in phase. Prepend to ensure consistent array size
    # dphi = np.diff(phi, prepend=phi[0] - (phi[1] - phi[0]), axis=0)
    # Change in time.  Prepend to ensure consistent array size
    # dt = np.diff(time_s, prepend=time_s[0] - (time_s[1] - time_s[0]), axis=0)
    # chirp = -1.0 / (2 * pi) * dphi / dt
    chirp = -1.0/2/pi*np.gradient(phi, time_s)
    return chirp


class ChannelClass:
    """
    Class for storing info about a certain frequency channel.

    Attributes:
        self.channelCenterFreq_Hz : float
            Central frequency of channel.
        self.channelMinFreq_Hz : float
            Lower frequency of channel.
        self.channelMaxFreq_Hz : float
            Upper frequency of channel.
        self.channelWidth_Hz : float
            Full channel width.


        self.signalCenterFreq_Hz : float
            Central frequency of signal.
        self.signalBW_Hz : float
            Signal Bandwidth.
        self.signalMinFreq_Hz : float
            Lower frequency of the signal in the channel.
        self.signalMaxFreq_Hz : float
            Upper frequency of the signal in the channel.

        self.leftGap_Hz : float
            Frequency gap between lowest channel freq and lowest signal freq

        self.rightGap_Hz : float
            Frequency gap between upper signal freq and upper channel freq

    """

    def __init__(
        self,
        channelCenterFreq_Hz: float,
        channelMinFreq_Hz: float,
        channelMaxFreq_Hz: float,
        signalCenterFreq_Hz: float,
        signalBW_Hz: float,
    ):
        """


        Parameters
        ----------
        channelCenterFreq_Hz : float
            Central frequency of channel.
        channelMinFreq_Hz : float
            Lower frequency of channel.
        channelMaxFreq_Hz : float
            Upper frequency of channel.
        signalCenterFreq_Hz : float
            Central frequency of signal.
        signalBW_Hz : float
            Signal Bandwidth.

        Returns
        -------
        None.

        """

        # Quick sanity check that center frequency is between min and max
        assert (
            channelMinFreq_Hz < channelMaxFreq_Hz
        ), "Error: channelMinFreq_Hz must be smaller than"
        f" channelMaxFreq_Hz, but channelMinFreq_Hz = {channelMinFreq_Hz}"
        f" >= {channelMaxFreq_Hz} = channelMaxFreq_Hz"
        assert (
            channelCenterFreq_Hz < channelMaxFreq_Hz
        ), "Error: channelCenterFreq_Hz must be smaller than"
        " maxFreq_Hz, but channelCenterFreq_Hz = {channelCenterFreq_Hz}"
        ">= {channelMaxFreq_Hz} = channelMaxFreq_Hz"
        assert (
            channelMinFreq_Hz < channelCenterFreq_Hz
        ), "Error: channelMinFreq_Hz must be smaller than"
        " centerFreq_Hz, but channelMinFreq_Hz = {channelMinFreq_Hz}"
        " >= {channelCenterFreq_Hz} = channelCenterFreq_Hz"

        self.channelCenterFreq_Hz = channelCenterFreq_Hz
        self.channelMinFreq_Hz = channelMinFreq_Hz
        self.channelMaxFreq_Hz = channelMaxFreq_Hz
        self.channelWidth_Hz = self.channelMaxFreq_Hz - self.channelMinFreq_Hz

        self.signalCenterFreq_Hz = signalCenterFreq_Hz
        self.signalBW_Hz = signalBW_Hz
        self.signalMinFreq_Hz = (self.signalCenterFreq_Hz -
                                 0.5 * self.signalBW_Hz)
        self.signalMaxFreq_Hz = (self.signalCenterFreq_Hz +
                                 0.5 * self.signalBW_Hz)

        # Quick sanity checks to ensure that signal is fully inside channel.
        # May seem pedantic, but making mistakes when allocating channels is
        # very easy!
        assert (
            signalBW_Hz > 0
        ), f"Error: {signalBW_Hz =} but should be greater than zero! "
        assert (
            channelMinFreq_Hz <= self.signalMinFreq_Hz
        ), "Error: channelMinFreq_Hz    must be smaller than signalMinFreq_Hz"
        ", but channelMinFreq_Hz ={channelMinFreq_Hz}>{self.signalMinFreq_Hz}"
        "= signalMinFreq_Hz"
        assert (
            channelMaxFreq_Hz >= self.signalMaxFreq_Hz
        ), "Error: channelMaxFreq_Hz must be greater than signalMaxFreq_Hz,"
        " but channelMaxFreq_Hz={channelMaxFreq_Hz} < {self.signalMaxFreq_Hz}"
        " = signalMaxFreq_Hz "

        self.leftGap_Hz = self.signalMinFreq_Hz - self.channelMinFreq_Hz
        self.rightGap_Hz = self.channelMaxFreq_Hz - self.signalMaxFreq_Hz


class TimeFreq:
    """
    Class for storing info about the time axis and frequency axis.

    Attributes:
        number_of_points (int): Number of time points
        time_step (float): Duration of each time step
        t (npt.NDArray[float]): Array containing all the time points
        tmin (float): First entry in time array
        tmax (float): Last entry in time array

        centerFrequency (float): Central optical frequency
        f (npt.NDArray[float]): Frequency range (relative to centerFrequency)
                     corresponding to t when FFT is taken
        fmin (float): Lowest (most negative) frequency component
        fmax (float): Highest (most positive) frequency component
        freq_step (float): Frequency resolution
    """

    def __init__(self,
                 N: int,
                 dt: float,
                 centerFrequency: float):
        """

        Constructor for the TimeFreq class.


m        Parameters
        ----------
        N : int
            Number of time points.
        dt : float
            Duration of each time step.
        centerFrequency : float
            Carrier frequency of the spectrum.

        Returns
        -------
        None.

        """

        self.number_of_points = N
        self.time_step = dt
        t = np.linspace(0, N * dt, N)
        self.t = t - np.mean(t)
        self.tmin = self.t[0]
        self.tmax = self.t[-1]

        self.centerFrequency = centerFrequency
        self.f = get_freq_range_from_time(self.t)
        self.fmin = self.f[0]
        self.fmax = self.f[-1]
        self.freq_step = self.f[1] - self.f[0]

        assert np.min(self.centerFrequency +
                      self.f) >= 0, f"ERROR! Lowest frequency of {np.min(self.centerFrequency+self.f)/1e9:.3f}GHz is below 0. Consider increasing the center frequency!"

        self.describe_config()

    def describe_config(self,
                        destination=None):
        """
        Prints information about the initialized TimeFreq class to file
        or terminal (default)

        Parameters
        ----------
        destination, optional
            File to which the output should be printed. The default is None,
            which prints to terminal.

        Returns
        -------
        None.

        """

        print(" ### timeFreq Configuration Parameters ###", file=destination)
        print(
            f"Number of points\t\t= {self.number_of_points}", file=destination)
        print(
            f"Start time, tmin\t\t= {self.tmin*1e12:.3f}ps", file=destination)
        print(f"Stop time, tmax\t\t= {self.tmax*1e12:.3f}ps", file=destination)
        print(
            f"Time resolution\t\t= {self.time_step*1e12:.3f}ps",
            file=destination)
        print("  ", file=destination)

        print(
            f"  Center frequency\t\t= {self.centerFrequency/1e12:.3f}THz",
            file=destination,
        )
        print(
            f"Start frequency\t\t= {self.fmin/1e12:.3f}THz", file=destination)
        print(
            f"Stop frequency \t\t= {self.fmax/1e12:.3f}THz", file=destination)
        print(
            f"Frequency resolution \t\t= {self.freq_step/1e6:.3f}MHz",
            file=destination,
        )
        print("   ", file=destination)

    def saveTimeFreq(self):
        """
        Saves info needed to construct this TimeFreq instance to .csv
        file so they can be loaded later using the load_TimeFreq function.

        Parameters:
            self
        """
        timeFreq_df = pd.DataFrame(
            columns=["number_of_points", "dt_s", "centerFreq_Hz"]
        )

        timeFreq_df.loc[len(timeFreq_df.index)] = [
            self.number_of_points,
            self.time_step,
            self.centerFrequency,
        ]

        timeFreq_df.to_csv("timeFreq.csv")


def load_TimeFreq(path: str) -> TimeFreq:
    """
    Loads TimeFreq for previous run

    Takes a path to a previous run, opens the relevant .csv file and extracts
    stored info from which the timeFreq class for that run can be restored.

    Parameters:
        path (str): Path to previous run

    Returns:
        TimeFreq: TimeFreq used in previous run.

    """

    df = pd.read_csv(path + "\\timeFreq.csv")
    number_of_points = df["number_of_points"]
    dt_s = df["dt_s"]
    centerFreq_Hz = df["centerFreq_Hz"]

    return TimeFreq(int(number_of_points[0]), dt_s[0], centerFreq_Hz[0])


def get_power(amplitude_in_time_or_freq_domain: npt.NDArray[complex]
              ) -> npt.NDArray[float]:
    """
    Computes temporal power or PSD

    For a real electric field, power averaged over an optical cycle is

    P = 1/T int_0^T( E_real**2 )dt.

    For a complex electric field, this same power is calculated as

    P = 0.5*|E|**2.

    Using the complex field makes calculations easier and the factor of
    0.5 is simply absorbed into the nonlinear parameter, gamma.
    Same thing works in the frequency domain.

    Parameters
    ----------
    amplitude_in_time_or_freq_domain : npt.NDArray[complex]
        Temporal or spectral amplitude.

    Returns
    -------
    power : npt.NDArray[complex]
        Temporal power (W) or PSD (J/Hz) at any instance or frequency.

    """
    power = np.abs(amplitude_in_time_or_freq_domain) ** 2
    return power


def get_energy(
    time_or_freq: npt.NDArray[float],
    amplitude_in_time_or_freq_domain: npt.NDArray[complex],
) -> float:
    """
    Computes energy of signal or spectrum

    Gets the power or PSD of the signal from
    get_power(amplitude_in_time_or_freq_domain)
    and integrates it w.r.t. either time or
    frequency to get the energy.

    Parameters
    ----------
    time_or_freq : npt.NDArray[float]
        Time range in seconds or freq. range in Hz.
    amplitude_in_time_or_freq_domain : npt.NDArray[complex]
        Temporal amplitude in [sqrt(W)] or spectral amplitude [sqrt(J/Hz)].

    Returns
    -------
    energy: float
        Signal energy in J .

    """
    energy = np.trapz(
        get_power(amplitude_in_time_or_freq_domain), time_or_freq)
    return energy


def gaussian_pulse(
    time_s: npt.NDArray[float],
    peakAmplitude: float,
    duration_s: float,
    time_offset_s: float,
    freq_offset_Hz: float,
    chirp: float,
    order: float,
) -> npt.NDArray[complex]:
    """
    Generates a (super) gaussian pulse with the specified power, duration,
    offset, frequency shift, (order) and chirp.

    Parameters
    ----------
    time_s : npt.NDArray[float]
        Time range in seconds.
    peakAmplitude : float
        Peak amplitude in units of sqrt(W)
    duration_s : float
       RMS width of Gaussian, i.e. time at which the amplitude is reduced
       by a factor exp(-0.5) = 0.6065 .
    time_offset_s : float
        Time at which the Gaussian peaks.
    freq_offset_Hz : float
        Center frequency relative to carrier frequency specified in timeFreq.
    chirp : float
        Dimensionless parameter controlling the chirp.
    order : float
        Controls shape of pulse as exp(-x**(2*order)) will be approximately
        square for large values of 'order'.

    Returns
    -------
    gaussian_pulse: npt.NDArray[complex]
        Gaussian pulse in time domain in units of sqrt(W).

    """

    assert order > 0, f"Error: Order of gaussian is {order}. Must be >0"
    carrier_freq = np.exp(-1j * 2 * pi * freq_offset_Hz * time_s)
    pulse = (
        peakAmplitude * np.exp(-(1 + 1j * chirp) / 2
                               * ((time_s - time_offset_s) / (duration_s)) ** (2 * order)
                               )
        * carrier_freq
    )
    return pulse


def square_pulse(
    time_s: npt.NDArray[float],
    peakAmplitude: float,
    duration_s: float,
    time_offset_s: float,
    freq_offset_Hz: float,
    chirp: float,
) -> npt.NDArray[complex]:
    """
    Generates a square pulse with the specified power, duration, offset,
    frequency shift and chirp.

    Parameters
    ----------
    time_s : npt.NDArray[float]
        Time range in seconds.
    peakAmplitude : float
        Peak amplitude in units of sqrt(W).
    duration_s : float
        Total duration of the square pulse.
    time_offset_s : float
        Time at which the pulse peaks.
    freq_offset_Hz : float
        Center frequency relative to carrier frequency specified in timeFreq.
    chirp : float
        Dimensionless parameter controlling the chirp.


    Returns
    -------
    square_pulse : npt.NDArray[complex]
        Square pulse in time domain in units of sqrt(W).

    """
    pulse = gaussian_pulse(
        time_s,
        peakAmplitude,
        duration_s,
        time_offset_s,
        freq_offset_Hz,
        chirp,
        100
    )
    return pulse


def sinc_pulse(
    time_s: npt.NDArray[float],
    peakAmplitude: float,
    duration_s: float,
    time_offset_s: float,
    freq_offset_Hz: float,
    chirp: float,
) -> npt.NDArray[complex]:
    """
    Creates a sinc pulse (sin(pi*x)/(pi*x))

    Generates a sinc pulse, which is useful as
    its spectral shape will be square.

    Parameters
    ----------
    time_s : npt.NDArray[float]
        Time range in seconds.
    peakAmplitude : float
        Peak amplitude in units of sqrt(W)
    duration_s : float
       Time (relative to peak) at which the first zero of the sinc function
       is reached; 'Half width at first zero'.
       Note that the full frequency bandwidth of the spectrum (square shape) of
       a sinc pulse will be BW_Hz = 1/duration_s
    time_offset_s : float
        Time at which the pulse peaks.
    freq_offset_Hz : float
        Center frequency relative to carrier frequency specified in timeFreq.
    chirp : float
        Dimensionless parameter controlling the chirp.


    Returns
    -------
    sinc_pulse: npt.NDArray[complex]
        Sinc pulse in time domain in units of sqrt(W).

    """

    carrier_freq = np.exp(-1j * 2 * pi * freq_offset_Hz * time_s)
    chirp_factor = np.exp(
        -(1j * chirp) / 2 * ((time_s - time_offset_s) / (duration_s)) ** 2
    )
    pulse = (
        peakAmplitude
        * np.sinc((time_s - time_offset_s) / (duration_s))
        * chirp_factor
        * carrier_freq
    )
    return pulse


def sech_pulse(
    time_s: npt.NDArray[float],
    peakAmplitude: float,
    duration_s: float,
    time_offset_s: float,
    freq_offset_Hz: float,
    chirp: float,
) -> npt.NDArray[complex]:
    """
    Creates hyperbolic secant pulse

    Generates a hyperbolic secant pulse (1/cosh(t)), which is the pulse
    shape that corresponds to a fundamental soliton; a solution to the NLSE
    for anormalous dispersion where the pulse remains unchanged as it
    propagates down the fiber.

    Parameters
    ----------
    time_s : npt.NDArray[float]
        Time range in seconds.
    peakAmplitude : float
        Peak amplitude in units of sqrt(W)
    duration_s : float
        Characteristic duration of sech pulse. sech(duration_s)=0.648...
    time_offset_s : float
        Time at which the pulse peaks.
    freq_offset_Hz : float
        Center frequency relative to carrier frequency specified in timeFreq.
    chirp : float
        Dimensionless parameter controlling the chirp.

    Returns
    -------
    sech_pulse : npt.NDArray[complex]
        Sech pulse in time domain in units of sqrt(W).

    """

    carrier_freq = np.exp(-1j * 2 * pi * freq_offset_Hz * time_s)
    chirp_factor = np.exp(
        -(1j * chirp) / 2 * ((time_s - time_offset_s) / (duration_s)) ** 2
    )
    pulse = (
        peakAmplitude
        / np.cosh((time_s - time_offset_s) / duration_s)
        * chirp_factor
        * carrier_freq
    )
    return pulse


def noise_ASE(
        time_s: npt.NDArray[float],
        noiseStdev: float
) -> npt.NDArray[complex]:
    """
    Generates white noise in the time domain with the
    specified Standard Deviation

    Generates an array of complex numbers with random phase from -pi to pi and
    amplitudes distributed normally around 0 and a standard
    deviation of noiseStdev in units of sqrt(W).

    Parameters
    ----------
    time_s : npt.NDArray[float]
        Time range in seconds.
    noiseStdev : float
        Standard deviation of temporal amplitude fluctuations in sqrt(W).

    Returns
    -------
    random_noise : npt.NDArray[complex]
        White noise.

    """

    random_amplitudes = np.random.normal(loc=0.0,
                                         scale=noiseStdev,
                                         size=len(time_s)) * (1+0j)

    random_phases = np.random.uniform(-pi, pi, len(time_s))
    random_noise = random_amplitudes * np.exp(1j * random_phases)
    return random_noise


def get_pulse(
    time_s: npt.NDArray[float],
    peakAmplitude: float,
    duration_s: float,
    time_offset_s: float,
    freq_offset_Hz: float,
    chirp: float,
    pulseType: str,
    order: float = 1.0,
    noiseStdev: float = 0.0,
) -> npt.NDArray[complex]:
    """
    Helper function that creates pulse with the specified properties

    Creates a Gaussian, sech, sinc or square pulse based on the 'pulseType'
    parameter.
    If pulseType == 'custom' it is assumed that the user wants to specify
    the pulse amplitude 'manually', in which case only noise is returned.

    Parameters
    ----------
    time_s : npt.NDArray[float]
        Time range in seconds.
    peakAmplitude : float
        Peak amplitude in units of sqrt(W)
    duration_s : float
        Characteristic duration of pulse. Meaning varies based on pulse type.
    time_offset_s : float
        Time at which the pulse peaks.
    freq_offset_Hz : float
        Center frequency relative to carrier frequency specified in timeFreq.
    chirp : float
        Dimensionless parameter controlling the chirp.
    pulseType : str
        String that determines which pulse type should be generated.
    order : float, optional
        Order of the Super Gaussian, exp(-x**(2*order)). The default is 1.0.
    noiseStdev : float, optional
        Standard deviation of temporal amplitude fluctuations in sqrt(W).
        The default is 0.0.

    Returns
    -------
    output_pulse : npt.NDArray[complex]
        Specified pulse in time domain in units of sqrt(W).

    """

    noise = noise_ASE(time_s, noiseStdev)
    output_pulse = 1j*np.zeros_like(time_s)
    if pulseType.lower() in ["gaussian", "gauss"]:
        output_pulse = (
            gaussian_pulse(
                time_s,
                peakAmplitude,
                duration_s,
                time_offset_s,
                freq_offset_Hz,
                chirp,
                order,
            ) + noise
        )

    if pulseType.lower() == "sech":
        output_pulse = (
            sech_pulse(
                time_s,
                peakAmplitude,
                duration_s,
                time_offset_s,
                freq_offset_Hz,
                chirp
            ) + noise
        )

    if pulseType.lower() == "square":
        output_pulse = (
            square_pulse(
                time_s,
                peakAmplitude,
                duration_s,
                time_offset_s,
                freq_offset_Hz,
                chirp
            ) + noise
        )

    if pulseType.lower() == "sinc":
        output_pulse = (
            sinc_pulse(
                time_s,
                peakAmplitude,
                duration_s,
                time_offset_s,
                freq_offset_Hz,
                chirp
            ) + noise
        )

    if pulseType.lower() == "custom":
        output_pulse = noise

    return output_pulse


def get_spectrum_from_pulse(
    time_s: npt.NDArray[float],
    pulse_amplitude: npt.NDArray[complex],
    FFT_tol: float = 1e-7,
) -> npt.NDArray[complex]:
    """


    Parameters
    ----------
    time_s : npt.NDArray[float]
        Time range in seconds.
    pulse_amplitude: npt.NDArray[complex]
        Complex amplitude of pulse in time domain in units of sqrt(W).
    FFT_tol : float, optional
        When computing the FFT and going from temporal to spectral domain, the
        energy (which theoretically should be conserved) cannot change
        fractionally by more than FFT_tol. The default is 1e-7.

    Returns
    -------
    spectrum_amplitude : npt.NDArray[complex]
        Complex spectral amplitude in units of sqrt(J/Hz).

    """

    pulseEnergy = get_energy(time_s, pulse_amplitude)  # Get pulse energy
    f = get_freq_range_from_time(time_s)
    dt = time_s[1] - time_s[0]

    assert dt > 0, (f"ERROR: dt must be positive, "
                    f"but {dt=}. {time_s[1]=},{time_s[0]=}")
    spectrum_amplitude = fftshift(
        fft(pulse_amplitude)) * dt  # Take FFT and do shift
    spectrumEnergy = get_energy(f, spectrum_amplitude)  # Get spectrum energy

    err = np.abs((pulseEnergy / spectrumEnergy - 1))

    assert (
        err < FFT_tol
    ), (f"ERROR = {err:.3e} > {FFT_tol:.3e} = FFT_tol : Energy changed "
        "when going from Pulse to Spectrum!!!")

    return spectrum_amplitude


def get_time_from_freq_range(frequency_Hz: npt.NDArray[float]
                             ) -> npt.NDArray[float]:
    """
    Calculate time range for pulse based on frequency range.

    Essentially the inverse of the get_freq_range_from_time function.
    If we have a frequency range and take the iFFT of a spectrum amplitude
    to get the pulse amplitude in the time domain, this function provides the
    appropriate time range.

    Parameters
    ----------
    frequency_Hz : npt.NDArray[float]
        Freq. range in Hz.

    Returns
    -------
    time_s : npt.NDArray[float]
        Time range in s.

    """

    time_s = fftshift(fftfreq(len(frequency_Hz),
                              d=frequency_Hz[1] - frequency_Hz[0]))
    return time_s


def get_pulse_from_spectrum(frequency_Hz: npt.NDArray[float],
                            spectrum_amplitude: npt.NDArray[complex],
                            FFT_tol: float = 1e-7) -> npt.NDArray[complex]:
    """
    Converts the spectral amplitude of a signal in the freq. domain temporal
    amplitude in time domain

    Uses the iFFT to shift from freq. to time domain and ensures that energy
    is conserved

    Parameters
    ----------
    frequency_Hz : npt.NDArray[float]
        Frequency in Hz.
    spectrum_amplitude : npt.NDArray[complex]
        Spectral amplitude in sqrt(J/Hz).
    FFT_tol : float, optional
        Maximum fractional change in signal
        energy when doing FFT. The default is 1e-7.

    Returns
    -------
    pulse : npt.NDArray[complex]
        Temporal amplitude in sqrt(W).

    """

    spectrumEnergy = get_energy(frequency_Hz, spectrum_amplitude)

    time = get_time_from_freq_range(frequency_Hz)
    dt = time[1] - time[0]

    pulse = ifft(ifftshift(spectrum_amplitude)) / dt
    pulseEnergy = get_energy(time, pulse)

    err = np.abs((pulseEnergy / spectrumEnergy - 1))

    assert (
        err < FFT_tol
    ), (f"ERROR = {err:.3e} > {FFT_tol:.3e} = FFT_tol : Energy changed too "
        "much when going from Spectrum to Pulse!!!")

    return pulse


class FiberSpan:
    """
    Class for storing info about a single fiber span.

    Attributes:
        Length (float): Length of fiber in [m]
        numberOfSteps (int): Number of identical steps
                             the fiber is divided into
        gamma (float): Nonlinearity parameter in [1/W/m]
        beta_list (list): List of dispersion
                          coefficients [beta2,beta3,...] [s^(entry+2)/m]
        alpha_dB_per_m (float): Attenuation coeff in [dB/m]
        alpha_Np_per_m (float): Attenuation coeff in [Np/m]
        use_self_steepening (bool): Toggles self-steepening effect. Default is False
        total_loss_dB (float):  Length*alpha_dB_per_m
        out_amp_dB       (float): optional output amplification in dB
        noise_factor_dB  (float): optional noise factor of amplifier in dB
        input_atten_dB   (float): optional input attenuation in dB
        out_atten_dB     (float): optional output attenuation in dB

    """

    def __init__(
        self,
        L: float,
        numberOfSteps: int,
        gamma: float,
        beta_list: list[float],
        alpha_dB_per_m: float,
        use_self_steepening: bool = False,
        ramanModel: str = "None",
        out_amp_dB: float = 0.0,
        noise_factor_dB: float = -1e3,
        input_atten_dB: float = 0.0,
        out_atten_dB: float = 0.0
    ):
        """
        Constructor for FiberSpan


        Parameters
        ----------
        L : float
            Length of fiber in [m].
        numberOfSteps : int
            Number of identical steps the fiber is divided into
        gamma : float
            Nonlinearity parameter in [1/W/m].
        beta_list : list[float]
            List of dispersion coefficients [beta2,beta3,...] [s^(entry+2)/m].
        alpha_dB_per_m : float
            Attenuation coeff in [dB/m].
        ramanModel : str, optional
            String to select Raman model. Default, "None", indicates that Raman
            should be ignored for this fiber.
        out_amp_dB : float, optional
            Lumped amplification at theend of the
            fiber span. The default is 0.0.
        noise_factor_dB : float, optional
            Noise factor of the amplification at the end of the
            fiber span. The default is -1e3.
        input_atten_dB : float, optional
            Attenuation at the input of the fiber due to splicing or
            misalignment. The default is 0.0.
        out_atten_dB : float, optional
            Attenuation at the output of the fiber due to splicing or
            misalignment. The default is 0.0.


        Returns
        -------
        None.

        """
        self.Length = float(L)
        self.numberOfSteps = int(numberOfSteps)
        self.z_array = np.linspace(0, self.Length, self.numberOfSteps + 1)
        self.dz = self.z_array[1] - self.z_array[0]

        self.gamma = float(gamma)

        # Pad list of betas so we always have terms up to 8th order
        while len(beta_list) <= 6:
            beta_list.append(0.0)
        self.beta_list = beta_list
        self.alpha_dB_per_m = float(alpha_dB_per_m)
        self.use_self_steepening = use_self_steepening
        # Loss coeff is usually specified in dB/km,
        # but Nepers/km is more useful for calculations
        self.alpha_Np_per_m = self.alpha_dB_per_m * np.log(10) / 10.0
        self.total_loss_dB = alpha_dB_per_m * self.Length
        # TODO: Make alpha frequency dependent.

        # TODO: Implement Raman model
        # Default: No Raman effect
        self.ramanModel = ramanModel
        self.fR = 0.0
        self.tau1 = 0.0
        self.tau2 = 0.0

        self.RamanInFreqDomain_func = lambda freq: 0.0

        # Raman parameters taken from Govind P. Agrawal's book,
        # "Nonlinear Fiber Optics".
        if ramanModel.lower() == "agrawal":
            self.fR = (
                0.180  # Relative contribution of Raman effect to overall nonlinearity
            )
            # Average angular oscillation time of molecular bonds in silica lattice. Note: 1/(2*pi*12.2fs) = 13.05THz = Typical Raman frequency
            self.tau1 = 12.2 * 1e-15
            # Average exponential decay time of molecular bond oscilaltions. Note: 2*1/(2*pi*30.0fs) = 10.61 THz = Typical Raman gain spectrum FWHM
            self.tau2 = 30.0 * 1e-15

            # Frequency domain representation of Raman response taken from https://github.com/omelchert/GNLStools/blob/main/src/GNLStools.py
            self.RamanInFreqDomain_func = lambda freq: (
                self.tau1 ** 2 + self.tau2 ** 2
            ) / (
                self.tau1 ** 2 * (1 - 1j * freq * 2 * pi * self.tau2) ** 2
                + self.tau2 ** 2
            )  # Freq domain representation of Raman response

        self.out_amp_dB = out_amp_dB
        self.noise_factor_dB = noise_factor_dB
        self.input_atten_dB = input_atten_dB
        self.out_atten_dB = out_atten_dB

        self.describe_fiber()

    def describe_fiber(self, destination=None):
        """
        Prints a description of the fiber to destination

        Parameters:
            self
            destination (class '_io.TextIOWrapper') (optional): File to which
            destination should be printed. If None, print to console
        """
        print(" ### Characteristic parameters of fiber: ###", file=destination)
        print(f"Fiber Length [km] \t= {self.Length/1e3} ", file=destination)
        print(f"Number of Steps \t= {self.numberOfSteps} ", file=destination)
        print(f"dz [m] \t= {self.dz} ", file=destination)

        print(f"Fiber gamma [1/W/m] \t= {self.gamma} ", file=destination)

        for i, beta_n in enumerate(self.beta_list):
            print(
                f"Fiber beta{i+2} [s^{i+2}/m] \t= {beta_n} ",
                file=destination)
        print(
            f"Fiber alpha_dB_per_m \t= {self.alpha_dB_per_m} ",
            file=destination)
        print(
            f"Fiber alpha_Np_per_m \t= {self.alpha_Np_per_m} ",
            file=destination)
        print(
            f"Fiber total loss [dB] \t= {self.total_loss_dB} ",
            file=destination)
        print(
            f"Raman Model \t= {self.ramanModel}. (fR,tau1,tau2)="
            f"({self.fR:.3},{self.tau1/1e-15:.3},{self.tau2/1e-15:.3}) ",
            file=destination,
        )
        print(
            f"Input Loss [dB] = {self.input_atten_dB} ", file=destination
        )
        print(
            f"Output Amplification [dB] = {self.out_amp_dB} ", file=destination
        )

        print(
            f"Noise Factor [dB] = {self.noise_factor_dB} ", file=destination
        )

        print(
            f"Output Attenuation [dB] = {self.out_atten_dB} ", file=destination
        )

        print(" ", file=destination)


# Class for holding info about span of concatenated fibers.
class FiberLink:
    """
    Class for storing info about multiple concatenated fibers.

    Attributes:
        fiber_list (list[FiberSpan]): List of FiberSpan objects
        number_of_fibers_in_span (int): Number of fibers concatenated together
    """

    def __init__(self, fiber_list: list[FiberSpan]):
        """
        Constructor for the FiberLink

        Parameters:
            self
            fiber_list (list[FiberSpan]): List of FiberSpan objects
        """

        self.fiber_list = fiber_list
        self.number_of_fibers_in_span = len(fiber_list)

    def save_fiber_link(self):
        """
        Saves info about each fiber in span to .csv file so they can be
        loaded later by the load_fiber_link function

        Parameters:
            self
        """
        fiber_df = pd.DataFrame(
            columns=[
                "Length_m",
                "numberOfSteps",
                "gamma_per_W_per_m",
                "beta2_s2_per_m",
                "beta3_s3_per_m",
                "beta4_s4_per_m",
                "beta5_s5_per_m",
                "beta6_s6_per_m",
                "beta7_s7_per_m",
                "beta8_s8_per_m",
                "alpha_dB_per_m",
                "alpha_Np_per_m",
                "ramanModel",
            ]
        )

        for fiber in self.fiber_list:
            fiber_df.loc[len(fiber_df.index)] = [
                fiber.Length,
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
                fiber.ramanModel,
            ]
        fiber_df.to_csv("fiber_link.csv")


def load_fiber_link(path: str) -> FiberLink:
    """
    Loads FiberLink for previous run

    Takes a path to a previous run, opens the relevant .csv file and extracts
    stored info from which the FiberLink for that run can be restored.

    Parameters:
        path (str): Path to previous run

    Returns:
        FiberLink: A class containing a list of fibers from a previous run.

    """
    df = pd.read_csv(path + "\\fiber_link.csv")
    Length_m = df["Length_m"]
    numberOfSteps = df["numberOfSteps"]
    gamma_per_W_per_m = df["gamma_per_W_per_m"]
    beta2_s2_per_m = df["beta2_s2_per_m"]
    beta3_s3_per_m = df["beta2_s3_per_m"]
    beta4_s4_per_m = df["beta2_s4_per_m"]
    beta5_s5_per_m = df["beta2_s5_per_m"]
    beta6_s6_per_m = df["beta2_s6_per_m"]
    beta7_s7_per_m = df["beta2_s7_per_m"]
    beta8_s8_per_m = df["beta2_s8_per_m"]

    alpha_dB_per_m = df["alpha_dB_per_m"]
    ramanModel = df["ramanModel"]

    fiber_list = []

    for i in range(len(Length_m)):
        beta_list_i = [
            beta2_s2_per_m[i],
            beta3_s3_per_m[i],
            beta4_s4_per_m[i],
            beta5_s5_per_m[i],
            beta6_s6_per_m[i],
            beta7_s7_per_m[i],
            beta8_s8_per_m[i],
        ]

        current_fiber = FiberSpan(
            Length_m[i],
            numberOfSteps[i],
            gamma_per_W_per_m[i],
            beta_list_i,
            alpha_dB_per_m[i],
            ramanModel[i],
        )
        fiber_list.append(current_fiber)
    return FiberLink(fiber_list)


class InputSignal:
    """
    Class for storing info about signal launched into a fiber link.

    Attributes:
        Amax (float): Peak amplitude of signal in [sqrt(W)]
        Pmax (float): Peak power of signal in [W]
        duration (float): Temporal duration of signal [s]
        offset (float): Delay of signal relative to t=0 in [s]
        chirp (float): Chirping factor of sigal
        pulseType (str): Selects pulse type from a set of pre-defined ones.
                         Select "custom" to define the signal manually
        order (float): For n==1 and pulseType = "Gaussian" a regular Gaussian
                     pulse is returned. For n>=1 return a super-Gaussian
        noiseAmplitude (float): Amplitude of added white
                                noise in units of [sqrt(W)].
        timeFreq (TimeFreq): Contains info about discretized time and freq axes
        amplitude (npt.NDArray): Numpy array containing the signal
                                 amplitude over time in [sqrt(W)]
        spectrum (npt.NDArray): Numpy array containing spectral amplitude
                                obtained from FFT of
                                self.amplitude in [sqrt(W)/Hz]
    """

    def __init__(
        self,
        timeFreq: TimeFreq,
        peak_amplitude: float,
        duration: float,
        time_offset_s: float,
        freq_offset_Hz: float,
        chirp: float,
        pulseType: str,
        order: float = 1.0,
        noiseAmplitude: float = 0.0,
        showOutput=True,
        FFT_tol=1e-7,
    ):
        """
        Constructor for InputSignal

        Parameters:
            timeFreq (TimeFreq): Contains info about discretized
                                 time and freq axes
            peak_amplitude (float): Peak amplitude of signal in [sqrt(W)]
            duration (float): Temporal duration of signal [s]
            time_offset_s (float): Delay of signal relative to t=0 in [s]
            chirp (float): Chirping factor of sigal
            pulseType (str): Selects pulse type from a set of pre-defined ones.
                             Select "custom" to define the signal manually
            order (float): For n==1 a and pulseType = "Gaussian" a regular
                           Gaussian pulse is returned. For n>=1 return
                           a super-Gaussian
            noiseAmplitude (float): Amplitude of added white
                                    noise in units of [sqrt(W)].
            showOutput=True (bool) (optional): Flag to determine if
                                               fiber characteristics
                                               should be printed
            FFT_tol=1e-7 (float) (optional): Maximum fractional change in
                                             signal energy when doing FFT

        """

        self.duration = duration
        self.time_offset_s = time_offset_s
        self.freq_offset_Hz = freq_offset_Hz
        self.chirp = chirp
        self.pulseType = pulseType
        self.order = order
        self.noiseAmplitude = noiseAmplitude
        self.timeFreq = timeFreq

        self.FFT_tol = FFT_tol

        self.amplitude = get_pulse(
            self.timeFreq.t,
            peak_amplitude,
            duration,
            time_offset_s,
            freq_offset_Hz,
            chirp,
            pulseType,
            order,
            noiseAmplitude,
        )

        self.spectrum = 1j * np.zeros_like(self.amplitude)

        self.Pmax = 0.0
        self.update_Pmax()
        self.Amax = 0.0
        self.update_Amax()

        if get_energy(self.timeFreq.t, self.amplitude) == 0.0:
            self.spectrum = np.copy(self.amplitude)
        else:
            self.update_spectrum()
        if showOutput:
            self.describe_input_signal()

    def update_spectrum(self):
        """
        Updates the spectrum. Useful if the time domain signal is altered, for
        example when a custom signal is generated by adding multiple ones
        together.

        Returns
        -------
        None.

        """
        self.spectrum = get_spectrum_from_pulse(
            self.timeFreq.t, self.amplitude, FFT_tol=self.FFT_tol
        )
        self.update_Pmax()
        self.update_Amax()

    def update_Pmax(self):
        """
        Updates the maximum power of signal. Useful if the amplitude is
        modified when a custom signal is generated.

        Returns
        -------
        None.

        """
        self.Pmax = np.max(get_power(self.amplitude))

    def update_Amax(self):
        """
        Updates the maximum amplitude of signal. Useful if the amplitude is
        modified when a custom signal is generated.

        Returns
        -------
        None.

        """
        self.Amax = np.sqrt(self.Pmax)

    def describe_input_signal(self, destination=None):
        """
        Prints a description of the input signal to destination

        Parameters:
            self
            destination (class '_io.TextIOWrapper') (optional): File to which
                    destination should be printed. If None, print to console
        """

        print(" ### Input Signal Parameters ###", file=destination)
        print(f"  Pmax \t\t\t\t= {self.Pmax:.3f} W", file=destination)
        print(
            f"  Duration \t\t\t= {self.duration*1e12:.3f} ps",
            file=destination)
        print(
            f"  Time offset \t\t\t= {self.time_offset_s*1e12:.3f} ps",
            file=destination
        )
        print(
            f"  Freq offset \t\t\t= {self.freq_offset_Hz/1e9:.3f} GHz",
            file=destination
        )
        print(f"  Chirp \t\t\t= {self.chirp:.3f}", file=destination)
        print(f"  pulseType \t\t\t= {self.pulseType}", file=destination)
        print(f"  order \t\t\t= {self.order}", file=destination)
        print(
            f"  noiseAmplitude \t\t= {self.noiseAmplitude:.3f} sqrt(W)",
            file=destination,
        )

        print("   ", file=destination)

        scalingFactor, prefix = get_units(self.timeFreq.t[-1])
        fig, ax = plt.subplots(dpi=300)
        ax.plot(self.timeFreq.t/scalingFactor, get_power(self.amplitude), '.')
        ax.set_xlabel(f'Time [{prefix}s]')
        ax.set_ylabel('Power [W]')
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.show()

    def saveInputSignal(self):
        """
        Saves info needed to construct this InputSignal instance to .csv
        file so they can be loaded later using the load_input_signal function.

        Parameters:
            self
        """
        # Initialize dataframe
        signal_df = pd.DataFrame(
            columns=[
                "Amax_sqrt(W)",
                "Pmax_W",
                "duration_s",
                "time_offset_s",
                "freq_offset_Hz",
                "chirp",
                "pulseType",
                "order",
                "noiseAmplitude_sqrt(W)",
            ]
        )

        # Fill it with values used for generating input signal
        signal_df.loc[len(signal_df.index)] = [
            self.Amax,
            self.Pmax,
            self.duration,
            self.time_offset_s,
            self.freq_offset_Hz,
            self.chirp,
            self.pulseType,
            self.order,
            self.noiseAmplitude,
        ]
        # Export dataframe to .csv file
        signal_df.to_csv("Input_signal.csv")

        # Also export timeFreq
        self.timeFreq.saveTimeFreq()

        if self.pulseType == "custom":
            custom_input_df = pd.DataFrame(
                columns=["time_s", "amplitude_sqrt_W_real",
                         "amplitude_sqrt_W_imag"]
            )

            custom_input_df["time_s"] = self.timeFreq.t
            custom_input_df["amplitude_sqrt_W_real"] = np.real(self.amplitude)
            custom_input_df["amplitude_sqrt_W_imag"] = np.imag(self.amplitude)

            custom_input_df.to_csv("Custom_input_signal.csv")


def load_input_signal(path: str) -> InputSignal:
    """
    Loads InputSignal for previous run

    Takes a path to a previous run, opens the relevant .csv file and extracts
    stored info from which the InputSignal for that run can be restored.

    Parameters:
        path (str): Path to previous run

    Returns:
        InputSignal: A class containing the input signal and time base.

    """
    # Open dataframe with pulse parameters
    df = pd.read_csv(path + "\\Input_signal.csv")

    Amax_sqrt_W = df["Amax_sqrt(W)"][0]
    duration_s = df["duration_s"][0]
    time_offset_s = df["time_offset_s"][0]
    freq_offset_Hz = df["freq_offset_Hz"][0]
    chirp = df["chirp"][0]
    pulseType = df["pulseType"][0]
    order = int(df["order"][0])
    noiseAmplitude_sqrt_W = df["noiseAmplitude_sqrt(W)"][0]

    # Load timeFreq
    timeFreq = load_TimeFreq(path)

    # Initialize class for loaded signal
    loaded_input_signal = InputSignal(
        timeFreq,
        Amax_sqrt_W,
        duration_s,
        time_offset_s,
        freq_offset_Hz,
        chirp,
        pulseType,
        order,
        noiseAmplitude_sqrt_W,
    )

    # If signal type is "custom", load the raw amplitude values
    if pulseType == "custom":
        df_custom = pd.read_csv(path + "\\Custom_input_signal.csv")

        A_real = np.array(df_custom["amplitude_sqrt_W_real"])
        A_imag = np.array(df_custom["amplitude_sqrt_W_imag"])
        A = A_real + 1j * A_imag

        loaded_input_signal.amplitude = A
    return loaded_input_signal


class SSFMResult:
    """
    Class for storing info about results computed by SSFM.

    Attributes:
        input_signal ( InputSignal ): Signal launched into fiber
        fiber ( FiberSpan ): Fiber signal was sent through
        experimentName ( str ): Name of experiment
        dirs ( tuple ): Contains directory where current script is located and
                    the directory where output is to be saved

        pulseMatrix ( npt.NDArray[complex] ): Amplitude of pulse at every
                                z-location in fiber
        spectrumMatrix ( npt.NDArray[complex] ): Spectrum of pulse at
                                    every z-location in fiber
    """

    def __init__(
        self,
        input_signal: InputSignal,
        fiber: FiberSpan,
        experimentName: str,
        directories: str,
    ):
        """
        Constructor for SSFMResult.

       Parameters:
            input_signal ( InputSignal ): Signal launched into fiber
            fiber ( FiberSpan ): Fiber signal was sent through
            experimentName ( str ): Name of experiment
            directories ( tuple ): Contains directory where current script is
            located and the directory where output is to be saved
        """
        self.input_signal = input_signal
        self.fiber = fiber
        self.experimentName = experimentName
        self.dirs = directories

        self.pulseMatrix = np.zeros(
            (len(fiber.z_array), input_signal.timeFreq.number_of_points)
        ) * (1 + 0j)
        self.spectrumMatrix = np.copy(self.pulseMatrix)

        self.pulseMatrix[0, :] = np.copy(input_signal.amplitude)
        self.spectrumMatrix[0, :] = np.copy(input_signal.spectrum)


def get_units(value: float) -> [float, str]:
    """
    Helper function for getting SI prefix (k, M, G, T, etc.)

    SSFM simulations can be used for both fibers with lengths on the
    scale from m to km and for photonic integrated circuits, which may only
    be a few cm long. Similarly, power, frequencies and times of interrest
    may have a wide range of scales. This function automatically determines
    the SI prefix and a scaling factor to be used for plotting.

    Parameters:
        value (float): Value whose order of magnitude is to be determined

    Returns:
        scalingFactor (float): If we want to plot a frequency of
        unknown magnitude, we would do plt.plot(f/scalingFactor)
        prefix (str): In the label of the plot, we would
        write plt.plot(f/scalingFactor,label=f"Freq. [{prefix}Hz]")
    """
    logval = np.log10(value)

    scalingFactor = 1.0
    prefix = ""

    if logval < -12:
        scalingFactor = 1e-15
        prefix = "f"
        return scalingFactor, prefix
    if logval < -9:
        scalingFactor = 1e-12
        prefix = "p"
        return scalingFactor, prefix
    if logval < -6:
        scalingFactor = 1e-9
        prefix = "n"
        return scalingFactor, prefix
    if logval < -3:
        scalingFactor = 1e-6
        prefix = "u"
        return scalingFactor, prefix
    if logval < -2:
        scalingFactor = 1e-3
        prefix = "m"
        return scalingFactor, prefix
    if logval < 0:
        scalingFactor = 1e-2
        prefix = "c"
        return scalingFactor, prefix
    if logval < 3:
        scalingFactor = 1e-0
        prefix = ""
        return scalingFactor, prefix
    if logval < 6:
        scalingFactor = 1e3
        prefix = "k"
        return scalingFactor, prefix
    if logval < 9:
        scalingFactor = 1e6
        prefix = "M"
        return scalingFactor, prefix
    if logval < 12:
        scalingFactor = 1e9
        prefix = "G"
        return scalingFactor, prefix
    if logval < 15:
        scalingFactor = 1e12
        prefix = "T"
        return scalingFactor, prefix
    if logval > 15:
        scalingFactor = 1e15
        prefix = "P"
        return scalingFactor, prefix


# TODO: clean up tabs printed to console.
def describe_sim_parameters(
    fiber: FiberSpan,
    input_signal: InputSignal,
    fiber_index: int,
    destination=None
):
    """
    Computes, prints and plots characteristic distances (L_eff, L_D, L_NL etc.)

    When solving the NLSE, different effects such as attenuation, dispersion,
    SPM, soliton oscillations etc. take place on different length scales.
    This function computes these length scales, prints them and plots a
    comparison, which is saved for reference. Note that this function
    adaptively if beta2 is positive or negative.

    Parameters:
        fiber (FiberSpan): Class containing info about the current fiber
        input_signal (InputSignal): Class containing info about input signal
        fiber_index (int): Index of fiber in the span
        destination (std) (optional): If None, print to console.
        Otherwise, print to file and make plot

    Returns:


    """
    scalingfactor, prefix = get_units(fiber.Length)
    length_list = np.array([])
    # Ensure that we don't measure distances in Mm or Gm
    if scalingfactor > 1e3:
        scalingfactor = 1e3
        prefix = "k"
    if destination is not None:
        fig, ax = plt.subplots(dpi=125)
        ax.set_title(
            (f" Fiber Index = {fiber_index} \nComparison of"
             "characteristic lengths")
        )
    print(" ### Characteristic parameters of simulation: ###",
          file=destination)
    print(
        f"  Length_fiber \t= {fiber.Length/scalingfactor:.2e} {prefix}m",
        file=destination,
    )

    if fiber.alpha_Np_per_m > 0:

        if fiber.alpha_Np_per_m == 0.0:
            L_eff = fiber.Length
        else:
            L_eff = (
                1 - np.exp(-fiber.alpha_Np_per_m * fiber.Length)
            ) / fiber.alpha_Np_per_m
        print(
            f"  L_eff       \t= {L_eff/scalingfactor:.2e} {prefix}m",
            file=destination
        )

        length_list = np.append(length_list, L_eff)
    if destination is not None:
        ax.barh("Fiber Length", fiber.Length / scalingfactor, color="C0")

        if fiber.alpha_Np_per_m > 0:
            ax.barh("Effective Length", L_eff / scalingfactor, color="C1")
    Length_disp_array = np.ones_like(fiber.beta_list) * 1.0e100

    for i, beta_n in enumerate(fiber.beta_list):

        if beta_n != 0.0:
            Length_disp = input_signal.duration ** (2 + i) / np.abs(beta_n)
            print(
                f"  Length_disp_{i+2} \t= {Length_disp/scalingfactor:.2e} {prefix}m",
                file=destination,
            )
            Length_disp_array[i] = Length_disp

            length_list = np.append(length_list, Length_disp)

            if destination is not None:
                ax.barh(
                    f"Dispersion Length (n = {i+2})",
                    Length_disp / scalingfactor,
                    color="C2",
                )
        else:
            Length_disp = 1e100
        Length_disp_array[i] = Length_disp
    if fiber.gamma != 0.0:

        Length_NL = 1 / fiber.gamma / input_signal.Pmax
        N_soliton = np.sqrt(Length_disp_array[0] / Length_NL)
        length_list = np.append(length_list, Length_NL)

        if destination is not None:
            ax.barh("Nonlinear Length", Length_NL / scalingfactor, color="C3")
        print(
            f"  Length_NL \t= {Length_NL/scalingfactor:.2e} {prefix}m",
            file=destination
        )
        print(f"  N_soliton \t= {N_soliton:.2e}", file=destination)
        print(f"  N_soliton^2 \t= {N_soliton**2:.2e}", file=destination)
    if fiber.beta_list[0] < 0:

        z_soliton = pi / 2 * Length_disp
        length_list = np.append(length_list, z_soliton)
        if destination is not None:
            ax.barh("Soliton Length", z_soliton / scalingfactor, color="C4")
        print(" ", file=destination)
        print(
            (f"  sign(beta2) \t= {np.sign(fiber.beta_list[0])}, so Solitons"
             " and Modulation Instability may occur "),
            file=destination,
        )
        print(
            f"   z_soliton \t= {z_soliton/scalingfactor:.2e} {prefix}m",
            file=destination,
        )
        print(f"   N_soliton \t= {N_soliton:.2e}", file=destination)
        print(f"   N_soliton^2 \t= {N_soliton**2:.2e}", file=destination)

        print(" ", file=destination)

        # https://prefetch.eu/know/concept/modulational-instability/
        f_MI = (
            np.sqrt(2 * fiber.gamma * input_signal.Pmax /
                    np.abs(fiber.beta_list[0]))
            / 2
            / pi
        )
        gain_MI = 2 * fiber.gamma * input_signal.Pmax
        print(f"   Freq. w. max MI gain = {f_MI/1e9:.2e}GHz", file=destination)
        print(
            f"   Max MI gain \t\t= {gain_MI*scalingfactor:.2e} /{prefix}m ",
            file=destination,
        )
        print(
            (f"   Min MI gain distance = {1/(gain_MI*scalingfactor):.2e}"
             f"{prefix}m "),
            file=destination,
        )
        print(" ", file=destination)
        length_list = np.append(length_list, 1 / gain_MI)
        if destination is not None:
            ax.barh("MI gain Length", 1 / (gain_MI * scalingfactor),
                    color="C5")
    elif fiber.beta_list[0] > 0 and fiber.gamma > 0:
        # https://prefetch.eu/know/concept/optical-wave-breaking/
        # Minimum N-value of Optical Wave breaking with Gaussian pulse
        Nmin_OWB = np.exp(3 / 4) / 2

        N_ratio = N_soliton / Nmin_OWB
        if N_ratio <= 1:
            Length_wave_break = 1e100
        else:
            # Characteristic length for OWB with Gaussian pulse
            Length_wave_break = Length_disp_array[0] / \
                np.sqrt(N_ratio ** 2 - 1)
        length_list = np.append(length_list, Length_wave_break)
        print(" ", file=destination)
        print(
            (f"   sign(beta2) \t\t\t\t= {np.sign(fiber.beta_list[0])},"
             " so Optical Wave Breaking may occur "),
            file=destination,
        )
        print(
            " Nmin_OWB (cst.) \t\t\t= 0.5*exp(3/4) (assuming Gaussian pulses)",
            file=destination,
        )
        print(
            f" N_ratio = N_soliton/Nmin_OWB \t= {N_ratio:.2e}",
            file=destination)
        print(
            (f" Length_wave_break\t\t\t= {Length_wave_break/scalingfactor:.2e}"
             f"{prefix}m"),
            file=destination,
        )

        if destination is not None:
            ax.barh("OWB Length", Length_wave_break /
                    scalingfactor, color="C6")
    if destination is not None:
        ax.barh("$\Delta$z", fiber.dz / scalingfactor, color="C7")
        length_list = np.append(length_list, fiber.dz)

        ax.set_xscale("log")
        ax.set_xlabel(f"Length [{prefix}m]")

        Lmin = np.min(length_list) / scalingfactor * 1e-1
        Lmax = fiber.Length / scalingfactor * 1e2
        ax.set_xlim(Lmin, Lmax)

        plt.savefig(
            f"Length_chart_{fiber_index}.png",
            bbox_inches="tight",
            pad_inches=1,
            orientation="landscape",
        )

        plt.show()


def describe_run(
    current_time: datetime,
    current_fiber: FiberSpan,
    current_input_signal: InputSignal,
    fiber_index: int = 0,
    destination: TextIO = None,
):
    """
    Prints info about fiber, characteristic lengths etc.

    Calls the self-describe function of fiber, the describe_sim_parameters
    function and prints info to specified destination

    Parameters
    ----------
    current_time : datetime
        Current date and time at which run was initiated.
    current_fiber : FiberSpan
        Class describing the current fiber
    current_input_signal : InputSignal
        Info about input signal.
    fiber_index : str or int, optional
        String of integer indexing fiber in fiber span. The default is 0.
    destination : TYPE, optional
        If None, print to console. Else, print to specified file.
        The default is None.

    Returns
    -------
    None.

    """

    print("Info about fiber", file=destination)
    current_fiber.describe_fiber(destination=destination)
    print(" ", file=destination)

    describe_sim_parameters(
        current_fiber,
        current_input_signal,
        fiber_index,
        destination=destination
    )


# TODO: update this with EDFA and input/output loss
def describeInputConfig(
    current_time: datetime,
    fiber: FiberSpan,
    input_signal: InputSignal,
    fiber_index: int
):
    """
    Prints info about fiber, characteristic lengths and stepMode

    Calls the self-describe function of fiber, the describe_sim_parameters function and prints stepMode info to console and file

    Parameters:
        current_time            (datetime): Current date and time at which run was initiated
        fiber                   (FiberSpan): Info about current fiber
        input_signal            (InputSignal): Info about input signal
        fiber_index             (str) : Integer indexing fiber in fiber span

    Returns:

    """
    with open(f"input_config_description_{fiber_index}.txt", "w") as output_file:
        # Print info to terminal

        describe_run(current_time, fiber, input_signal,
                     fiber_index=str(fiber_index))

        # Print info to file
        describe_run(
            current_time,
            fiber,
            input_signal,
            fiber_index=str(fiber_index),
            destination=output_file,
        )


def create_output_directory(experimentName: str) -> [(str, str), datetime]:
    """
    Creates output directory for output (graphs etc.)

    Parameters
    ----------
    experimentName : str
        String with the name of the current experiment.

    Returns
    -------
    base_dir : str
        Location of the script that is being run.
    current_dir: str
        Location where the output of the experiment will be saved.
    current_time : datetime.datetime
        datetime object of time where the run was started.

    """
    os.chdir(os.path.realpath(os.path.dirname(__file__)))
    base_dir = os.getcwd() + "\\"
    os.chdir(base_dir)

    current_dir = ""
    current_time = datetime.now()

    if experimentName == "most_recent_run":
        current_dir = base_dir + "most_recent_run\\"
        overwrite_folder_flag = True
    else:

        current_dir = (
            base_dir
            + "Simulation Results\\" +
            f"{experimentName}\\"
            f"{current_time.year}_{current_time.month}_{current_time.day}_" +
            f"{current_time.hour}_{current_time.minute}_" +
            f"{current_time.second}\\"
        )
        overwrite_folder_flag = False
    os.makedirs(current_dir, exist_ok=overwrite_folder_flag)
    os.chdir(current_dir)

    print(f"Current time is {current_time}")
    print("Current dir is " + current_dir)

    return (base_dir, current_dir), current_time


def load_previous_run(basePath: str) -> [FiberLink, InputSignal]:
    """
    Loads all relevant info about previous run

    When path to previous run folder is specified, open .csv files describing
    fiber, signal and stepconfig.
    Use the stored values to reconstruct the parameters for the run.

    Parameters:
        basePath (str): Path to run folder

    Returns:
        fiber_link (FiberLink): Info about fiber span consisting of
                                1 or more fibers
        input_signal (InputSignal): Info about input signal

    """
    print(f"Loading run in {basePath}")

    fiber_link = load_fiber_link(basePath + "\\input_info\\")
    input_signal = load_input_signal(basePath + "\\input_info\\")

    print(f"Successfully loaded run in {basePath}")

    return fiber_link, input_signal


def get_NL_factor_simple(fiber: FiberSpan,
                         timeFreq: TimeFreq,
                         pulse: npt.NDArray[complex],
                         dz_m: float) -> npt.NDArray[complex]:
    """
    Calculates nonlinear phase shift in time domain for the simple, scalar case
    without Raman

    Parameters
    ----------
    fiber : FiberSpan
        Fiber that we are currently propagating through.
    timeFreq : TimeFreq
        Unused in this function but keep it so this function has the same
        arguments as get_NL_factor_full.
    pulse : np.array([complex])
        Pulse whose power determines the nonlinear phase shift.
    dz : float
        Step size in m in the z-direction.

    Returns
    -------
    np.array([complex])
        Array of complex exponentials to be applied to signal.

    """

    return np.exp(1j * fiber.gamma * get_power(pulse) * dz_m)


def get_NL_factor_self_steepening(fiber: FiberSpan,
                                  timeFreq: TimeFreq,
                                  pulse: npt.NDArray[complex],
                                  dz_m: float) -> npt.NDArray[complex]:
    """
    Calculates nonlinear phase shift in time domain for the simple, scalar case
    without Raman, but with self-steepening.

    Explanation of self-steepening:
    https://prefetch.eu/know/concept/self-steepening/

    Parameters
    ----------
    fiber : FiberSpan
        Fiber that we are currently propagating through.
    timeFreq : TimeFreq
        Unused in this function but keep it so this function has the same
        arguments as get_NL_factor_full.
    pulse : np.array([complex])
        Pulse whose power determines the nonlinear phase shift.
    dz : float
        Step size in m in the z-direction.

    Returns
    -------
    np.array([complex])
        Array of complex exponentials to be applied to signal.

    """
    pulse_power = get_power(pulse)
    output = np.exp(1j * fiber.gamma*(pulse_power+1j/2/np.pi/timeFreq.centerFrequency /
                    (pulse+np.sqrt(np.max(pulse_power))/1e6*(1+0j))*np.gradient(pulse_power*pulse, timeFreq.t)) * dz_m)

    return output


# TODO: Fully implement Raman model
def get_NL_factor_full(fiber: FiberSpan,
                       timeFreq: TimeFreq,
                       pulse: npt.NDArray[complex],
                       dz_m: float) -> npt.NDArray[complex]:
    # TODO: Implement Raman effect for both long and short-duration pulses
    fR = fiber.fR
    freq = timeFreq.f
    t = timeFreq.t

    f0 = timeFreq.centerFrequency
    RamanInFreqDomain = fiber.RamanInFreqDomain_func(freq)

    def NR_func(current_pulse):
        return (1 - fR) * get_power(
            current_pulse
        ) * current_pulse + fR * current_pulse * get_pulse_from_spectrum(
            freq, get_spectrum_from_pulse(t, get_power(
                current_pulse)) * RamanInFreqDomain
        )

    NR_in_freq_domain = (
        1j
        * dz_m
        * fiber.gamma
        * (1.0 + freq / f0)
        * get_pulse_from_spectrum(freq, NR_func(get_spectrum_from_pulse(timeFreq.t, pulse)))
    )

    return np.exp(get_pulse_from_spectrum(freq, NR_in_freq_domain))


# TODO: Make NF and gain frequency dependent?
def get_noise_PSD(NF_dB: float,
                  gain_dB: float,
                  f_Hz: npt.NDArray[float],
                  df_Hz: float) -> npt.NDArray[complex]:
    """
    Calculates PSD of noise from an amplifier behaving like an idealized EDFA:

    https://electricajournal.org/Content/files/sayilar/37/1111-1122.pdf

    Note that amplifier saturation is NOT accounted for, and that the gain
    bandwidth is assumed to be infinite (i.e. EVERY frequency experiences
                                         amplification!)

    Parameters
    ----------
    NF_dB : float
        Noise factor of the EDFA in dB.
    gain_dB : float
        Gain in dB of the EDFA.
    f_Hz : npt.NDArray[float]
        Frequencies at which we want the PSD to be evaluated.
    df : float
        Spectral resolution.

    Returns
    -------
    npt.NDArray[complex]
        PSD at a certain frequency for a given noise factor,
        gain and resolution.

    """

    NF_lin = dB_to_lin(NF_dB)
    G_lin = dB_to_lin(gain_dB)

    return 0.5 * NF_lin * G_lin * PLANCKCONST_J_PER_HZ * f_Hz / df_Hz


def SSFM(
    fiber_link: FiberLink,
    input_signal: InputSignal,
    experimentName: str = "most_recent_run",
    showProgressFlag: bool = False,
    FFT_tol: float = 1e-7,
) -> list[SSFMResult]:
    """
    Runs the Split-Step Fourier method and calculates field throughout fiber

    Runs the SSFM to solve the NLSE with the specified parameters.
    Goes through the following steps:
        1) Create folder for saving input config and results
        2) Loops over fibers in fiber_link. Gets zsteps
                for each and runs the SSFM
        3) Stores results of each fiber in a separate SSFMResult and
                uses pulse at the end as input to next one
        4) Returns list of SSFMResult objects

    Parameters:
        fiber_link (FiberLink): Class holding fibers through
                                which the signal is propagated
        input_signal (InputSignal): Class holding info about
                                    initial input signal
        experimentName ="most_recent_run" (str) (optional): Name of folder for
                                                            present simulation.
        showProgressFlag = False (bool) (optional): Print percentage
                                                    progress to terminal?
        FFT_tol=1e-7 (float) (optional): Maximum fractional change in signal
                                         energy when doing FFT


    Returns:
        list: List of SSFMResult corresponding to each fiber segment.

    """
    print("########### Initializing SSFM!!! ###########")

    t = input_signal.timeFreq.t
    # dt = input_signal.timeFreq.time_step
    f = input_signal.timeFreq.f
    df = input_signal.timeFreq.freq_step
    fc = input_signal.timeFreq.centerFrequency

    # Create output directory, switch to it and return
    # appropriate paths and current time
    dirs, current_time = create_output_directory(experimentName)

    # Make new folder to hold info about the input signal and fiber span
    base_dir = dirs[0]
    current_dir = dirs[1]

    newFolderName = "input_info\\"
    newFolderPath = newFolderName
    os.makedirs(newFolderPath, exist_ok=True)
    os.chdir(newFolderPath)

    # Save parameters of fiber span to file in directory
    fiber_link.save_fiber_link()

    # Save input signal parameters
    input_signal.saveInputSignal()

    # Return to main output directory
    os.chdir(current_dir)

    current_input_signal = input_signal

    ssfm_result_list = []

    print(f"Starting SSFM loop over {len(fiber_link.fiber_list)} fibers")

    for fiber_index, fiber in enumerate(fiber_link.fiber_list):

        print(
            (f"Propagating through fiber number {fiber_index+1} out of "
             f"{fiber_link.number_of_fibers_in_span}")
        )

        # Initialize arrays to store pulse and spectrum throughout fiber
        ssfm_result = SSFMResult(
            current_input_signal, fiber, experimentName, dirs
        )

        newFolderName = "Length_info\\"
        newFolderPath = newFolderName
        os.makedirs(newFolderPath, exist_ok=True)
        os.chdir(newFolderPath)

        # TODO: Decide if this should be re-enabled
        # Print simulation info to both terminal and .txt file in output folder
        # describeInputConfig(current_time,
        #                     fiber,
        #                     current_input_signal,
        #                     fiber_index)

        # Return to main output directory
        os.chdir(current_dir)

        # Pre-calculate dispersion term
        dispterm = np.zeros_like(input_signal.timeFreq.f) * 1.0
        for idx, beta_n in enumerate(fiber.beta_list):
            n = idx + 2  # Note: zeroth entry in beta_list is beta2
            dispterm += (beta_n / np.math.factorial(n)
                         * (-2 * pi * f) ** (n)
                         )

            if idx == 1 and beta_n > 0:
                fig, ax = plt.subplots(dpi=300)
                ax.plot(f, dispterm)
                plt.show()

        # Pre-calculate effect of dispersion and loss as it's
        # the same everywhere
        disp_and_loss = np.exp(
            fiber.dz * (1j * dispterm - fiber.alpha_Np_per_m / 2))
        disp_and_loss_half_step = disp_and_loss ** 0.5

        # Precalculate constants for nonlinearity

        # Use simple NL model by default if Raman is ignored

        # TODO: sort out logic for choosing NL function
        if fiber.use_self_steepening:
            NL_function = get_NL_factor_self_steepening
        else:
            NL_function = get_NL_factor_simple

        if fiber.ramanModel != "None":
            NL_function = get_NL_factor_full

        inputAttenuationField_lin = dB_to_lin(fiber.input_atten_dB / 2)
        outputAttenuationField_lin = 1.0  # temporarily=1 until we reach end
        # temporary value until we reach fiber end
        noise_ASE_array = 1.0 * np.zeros_like(f)
        outputAmp_factor = 1.0  # temporary value until we reach fiber end

        # Initialize arrays to store temporal profile
        # and spectrum while calculating SSFM
        spectrum = (
            get_spectrum_from_pulse(
                current_input_signal.timeFreq.t,
                current_input_signal.amplitude,
                FFT_tol=FFT_tol,
            )
            * disp_and_loss_half_step
            * inputAttenuationField_lin
        )

        pulse = get_pulse_from_spectrum(
            input_signal.timeFreq.f, spectrum, FFT_tol=FFT_tol)

        # ^Done just above
        # Apply half dispersion step
        #
        # Start loop
        #   Apply full NL step
        #   Apply full Disp step
        # End loop
        # Apply half dispersion step
        # Save outputs and proceed to next fiber

        print(f"Running SSFM with {fiber.numberOfSteps} steps")
        updates = 0
        for z_step_index in range(fiber.numberOfSteps):

            # Apply nonlinearity
            pulse *= NL_function(fiber, input_signal.timeFreq, pulse, fiber.dz)

            if z_step_index in [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]:
                fig, ax = plt.subplots(dpi=300)
                ax.set_title(f'z = {z_step_index*fiber.dz:.3}m')
                ax.plot(t/1e-12, get_power(pulse))
                ax.set_ylim(0, 1.5e3)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.set_xlabel('Time [ps]')
                ax.set_ylabel('Power [W]')

                plt.show()

            # if True in np.isnan(pulse):
            #     fig, ax = plt.subplots(dpi=300)
            #     ax.plot(t, get_power(pulse), color='C2')
            #     plt.show()
            #     print(np.isnan(NL_function(
            #         fiber, input_signal.timeFreq, pulse, fiber.dz)))

            #     fig, ax = plt.subplots(dpi=300)
            #     ax.plot(t, np.isnan(NL_function(
            #         fiber, input_signal.timeFreq, pulse, fiber.dz)), color='C2')
            #     plt.show()

            #     assert 1 == 2

            # Go to spectral domain and apply disp and loss

            # if not fiber.use_self_steepening:
            #     fig, ax = plt.subplots(dpi=300)
            #     ax.plot(f, dispterm)
            #     plt.show()

            #     fig, ax = plt.subplots(dpi=300)
            #     ax.plot(f, get_phase(get_spectrum_from_pulse(
            #         t, pulse, FFT_tol=FFT_tol)))
            #     ax.set_ylim(-150, 150)
            #     plt.show()

            spectrum = get_spectrum_from_pulse(
                t, pulse, FFT_tol=FFT_tol) * (disp_and_loss)

            # if not fiber.use_self_steepening:
            #     fig, ax = plt.subplots(dpi=300)
            #     ax.plot(f, dispterm)
            #     plt.show()

            #     fig, ax = plt.subplots(dpi=300)
            #     ax.plot(f, get_phase(spectrum))
            #     ax.set_ylim(-150, 150)

            #     plt.show()

            #     assert 1 == 2

            # If at the end of fiber span, apply output amp and noise
            if z_step_index == fiber.numberOfSteps - 1:
                randomPhases = np.random.uniform(-pi, pi, len(f))
                randomPhaseFactor = np.exp(1j * randomPhases)
                outputAttenuationField_lin = dB_to_lin(
                    fiber.out_atten_dB / 2)
                outputAmp_factor = 10 ** (fiber.out_amp_dB / 20)
                noise_ASE_array = randomPhaseFactor * np.sqrt(
                    get_noise_PSD(
                        fiber.noise_factor_dB,
                        fiber.out_amp_dB,
                        f + fc,
                        df
                    )
                )

            # Apply half dispersion step to spectrum and store results
            ssfm_result.spectrumMatrix[z_step_index + 1, :] = (
                spectrum * disp_and_loss_half_step * outputAmp_factor
                + noise_ASE_array
            ) * outputAttenuationField_lin

            print(z_step_index)

            if z_step_index == 459:
                fig, ax = plt.subplots(dpi=300)
                ax.plot(f/1e12, get_power(spectrum))
                ax.plot(
                    f/1e12, get_power(ssfm_result.spectrumMatrix[z_step_index + 1, :]))
                ax.set_xlabel('f [THz]')
                ax.set_ylabel('Energy dens. [J/Hz]')

                plt.show()

            ssfm_result.pulseMatrix[z_step_index + 1, :] = get_pulse_from_spectrum(
                f,
                ssfm_result.spectrumMatrix[z_step_index + 1, :],
                FFT_tol=FFT_tol
            )

            # Return to time domain
            pulse = get_pulse_from_spectrum(f,
                                            spectrum,
                                            FFT_tol=FFT_tol)

            finished = 100 * (z_step_index / fiber.numberOfSteps)
            if divmod(finished, 10)[0] > updates and showProgressFlag:
                updates += 1
                print(
                    (f"SSFM progress through fiber number {fiber_index+1} = "
                     f"{np.floor(finished):.2f}%")
                )
        # Append list of output results

        ssfm_result_list.append(ssfm_result)

        # Take signal at output of this fiber and feed it into the next one
        current_input_signal.amplitude = np.copy(
            ssfm_result.pulseMatrix[z_step_index + 1, :]
        )
        current_input_signal.spectrum = np.copy(
            ssfm_result.spectrumMatrix[z_step_index + 1, :]
        )
    print("Finished running SSFM!!!")

    # Exit current output directory and return to base directory.
    os.chdir(base_dir)

    return ssfm_result_list


def save_plot(basename: str):
    """
    Helper function for adding file type suffix to name of plot

    Helper function for adding file type suffix to name of plot

    Parameters:
        basename (str): Name to which a file extension is to be
        appended if not already present.

    Returns:


    """

    if basename.lower().endswith((".pdf", ".png", ".jpg")) is False:
        basename += ".png"
    plt.savefig(basename, bbox_inches="tight", pad_inches=0)


def unpack_Zvals(ssfm_result_list: list[SSFMResult]) -> npt.NDArray[float]:
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
        ssfm_result_list (list): List of ssmf_result_class
            objects corresponding to each fiber segment

    Returns:
        npt.NDArray: z_values for each fiber concatenated together.

    """
    number_of_fibers = len(ssfm_result_list)
    if number_of_fibers == 1:
        return ssfm_result_list[0].fiber.z_array
    zvals = np.array([])

    previous_length = 0
    for i, ssfm_result in enumerate(ssfm_result_list):

        if i == 0:
            zvals = np.copy(ssfm_result.fiber.z_array[0:-1])
        elif (i > 0) and (i < number_of_fibers - 1):
            zvals = np.append(
                zvals, ssfm_result.fiber.z_array[0:-1] + previous_length)
        elif i == number_of_fibers - 1:
            zvals = np.append(
                zvals, ssfm_result.fiber.z_array + previous_length)
        previous_length += ssfm_result.fiber.Length
    return zvals


def unpack_matrix(ssfm_result_list: list[SSFMResult],
                  zvals: npt.NDArray[float],
                  pulse_or_spectrum: str) -> npt.NDArray[complex]:
    """
    Unpacks pulseMatrix or spectrumMatrix for individual
    fibers in ssfm_result_list into single array

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
        zvals (npt.NDArray) : Array of unpacked z_values from unpack_Zvals. Needed for pre-allocating returned matrix
        timeFreq (TimeFreq): timeFreq for simulation. Needed for pre-allocation
        pulse_or_spectrum (str) : Indicates if we want to unpack pulseMatrix or spectrumMatrix

    Returns:
        npt.NDArray: Array of size (n_z_steps,n_time_steps) describing pulse amplitude or spectrum for whole fiber span.

    """
    timeFreq = ssfm_result_list[0].input_signal.timeFreq
    number_of_fibers = len(ssfm_result_list)

    # print(f"number_of_fibers = {number_of_fibers}")

    matrix = np.zeros((len(zvals), len(timeFreq.t))) * (1 + 0j)

    starting_row = 0

    for i, ssfm_result in enumerate(ssfm_result_list):

        if pulse_or_spectrum.lower() == "pulse":
            sourceMatrix = ssfm_result.pulseMatrix
        elif pulse_or_spectrum.lower() == "spectrum":
            sourceMatrix = ssfm_result.spectrumMatrix
        else:
            print(
                ("ERROR: Please set pulse_or_spectrum to either "
                 " 'pulse' or 'spectrum'!!!")
            )
            return
        if number_of_fibers == 1:
            return sourceMatrix
        if i == 0:
            matrix[0: len(ssfm_result.fiber.z_array) - 1, :] = sourceMatrix[
                0: len(ssfm_result.fiber.z_array) - 1, :
            ]
        elif (i > 0) and (i < number_of_fibers - 1):

            matrix[
                starting_row: starting_row + len(ssfm_result.fiber.z_array) - 1, :
            ] = sourceMatrix[0: len(ssfm_result.fiber.z_array) - 1, :]
        elif i == number_of_fibers - 1:

            matrix[
                starting_row: starting_row + len(ssfm_result.fiber.z_array), :
            ] = sourceMatrix[0: len(ssfm_result.fiber.z_array), :]
        starting_row += len(ssfm_result.fiber.z_array) - 1
    return matrix


def plot_first_and_last_pulse(ssfm_result_list: list[SSFMResult],
                              nrange: int,
                              dB_cutoff: float,
                              **kwargs):
    """
    Plots input pulse and output pulse of simulation

    Line plot of input pulse and output pulse of SSFM run centered in the
    middle of the time array and with nrange points on either side

    Parameters
    ----------
    ssfm_result_list : list[SSFMResult]
        List of ssmf_result_class objects corresponding to each fiber segment.
    nrange : int
        How many points on either side of the center do we wish to plot?
    dB_cutoff : float
        Lowest y-value in plot is this many dB smaller than the peak power.
    **kwargs : TYPE
        If firstandlastpulsescale=='log' is contained in
                  keyword args, set y-scale to log.

    Returns
    -------
    None.

    """

    timeFreq = ssfm_result_list[0].input_signal.timeFreq

    Nmin = np.max([int(timeFreq.number_of_points / 2 - nrange), 0])
    Nmax = np.min(
        [int(timeFreq.number_of_points / 2 + nrange),
         timeFreq.number_of_points - 1]
    )

    zvals = unpack_Zvals(ssfm_result_list)

    t = timeFreq.t[Nmin:Nmax] * 1e12

    P_initial = get_power(ssfm_result_list[0].pulseMatrix[0, Nmin:Nmax])
    P_final = get_power(ssfm_result_list[-1].pulseMatrix[-1, Nmin:Nmax])

    scalingFactor, prefix = get_units(np.max(zvals))

    os.chdir(ssfm_result_list[0].dirs[1])
    fig, ax = plt.subplots(dpi=125)
    ax.set_title("Initial pulse and final pulse")
    ax.plot(t, P_initial, label=f"Initial Pulse at z = 0{prefix}m")
    ax.plot(
        t,
        P_final,
        label=f"Final Pulse at z = {zvals[-1]/scalingFactor}{prefix}m")

    ax.set_xlabel("Time [ps]")
    ax.set_ylabel("Power [W]")

    for kw, value in kwargs.items():
        if kw.lower() == "firstandlastpulsescale" and value.lower() == "log":
            ax.set_yscale("log")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    save_plot("first_and_last_pulse")
    plt.show()
    os.chdir(ssfm_result_list[0].dirs[0])


def plot_pulse_matrix_2D(ssfm_result_list: list[SSFMResult],
                         nrange: int,
                         dB_cutoff: float):
    """
    Plots amplitude calculated by SSFM as colour surface

    2D colour plot of signal amplitude in time domain
    throughout entire fiber span
    normalized to the highest peak power throughout.

    Parameters
    ----------
    ssfm_result_list : list[SSFMResult]
        List of ssmf_result_class objects corresponding to each fiber segment.
    nrange : int
        How many points on either side of the center do we wish to plot?
    dB_cutoff : float
        Lowest y-value in plot is this many dB smaller than the peak power.

    Returns
    -------
    None.

    """

    timeFreq = ssfm_result_list[0].input_signal.timeFreq

    Nmin = np.max([int(timeFreq.number_of_points / 2 - nrange), 0])
    Nmax = np.min(
        [int(timeFreq.number_of_points / 2 + nrange),
         timeFreq.number_of_points - 1]
    )

    zvals = unpack_Zvals(ssfm_result_list)
    print(f"length of zvals = {len(zvals)}")
    matrix = unpack_matrix(ssfm_result_list, zvals, "pulse")

    # Plot pulse evolution throughout fiber in normalized log scale
    os.chdir(ssfm_result_list[0].dirs[1])
    fig, ax = plt.subplots(dpi=125)
    ax.set_title("Pulse Evolution (dB scale)")
    t_ps = timeFreq.t[Nmin:Nmax] * 1e12
    z = zvals
    T_ps, Z = np.meshgrid(t_ps, z)
    P = get_power(matrix[:, Nmin:Nmax]) / \
        np.max(get_power(matrix[:, Nmin:Nmax]))
    P[P < 1e-100] = 1e-100
    P = 10 * np.log10(P)
    P[P < dB_cutoff] = dB_cutoff
    surf = ax.contourf(T_ps, Z, P, levels=40, cmap="jet")
    ax.set_xlabel("Time [ps]")
    ax.set_ylabel("Distance [m]")
    cbar = fig.colorbar(surf, ax=ax)
    save_plot("pulse_evo_2D")
    plt.show()
    os.chdir(ssfm_result_list[0].dirs[0])


def plot_pulse_matrix_3D(ssfm_result_list: list[SSFMResult],
                         nrange: int,
                         dB_cutoff: float):
    """
     Plots amplitude calculated by SSFM as 3D colour surface

     3D colour plot of signal amplitude in time domain
     throughout entire fiber span normalized to the highest
     peak power throughout.

    Parameters
    ----------
    ssfm_result_list : list[SSFMResult]
        List of ssmf_result_class objects corresponding to each fiber segment.
    nrange : int
        How many points on either side of the center do we wish to plot?
    dB_cutoff : float
        Lowest y-value in plot is this many dB smaller than the peak power.

    Returns
    -------
    None.

    """

    timeFreq = ssfm_result_list[0].input_signal.timeFreq

    Nmin = np.max([int(timeFreq.number_of_points / 2 - nrange), 0])
    Nmax = np.min(
        [int(timeFreq.number_of_points / 2 + nrange),
         timeFreq.number_of_points - 1]
    )

    zvals = unpack_Zvals(ssfm_result_list)
    matrix = unpack_matrix(ssfm_result_list, zvals, "pulse")

    # Plot pulse evolution in 3D
    os.chdir(ssfm_result_list[0].dirs[1])
    fig, ax = plt.subplots(1, 1, figsize=(
        10, 7), subplot_kw={"projection": "3d"})
    plt.title("Pulse Evolution (dB scale)")

    t = timeFreq.t[Nmin:Nmax] * 1e12
    z = zvals
    T_surf, Z_surf = np.meshgrid(t, z)
    P_surf = get_power(matrix[:, Nmin:Nmax]) / \
        np.max(get_power(matrix[:, Nmin:Nmax]))
    P_surf[P_surf < 1e-100] = 1e-100
    P_surf = 10 * np.log10(P_surf)
    P_surf[P_surf < dB_cutoff] = dB_cutoff
    # Plot the surface.
    surf = ax.plot_surface(
        T_surf, Z_surf, P_surf, cmap=cm.jet, linewidth=0, antialiased=False
    )
    ax.set_xlabel("Time [ps]")
    ax.set_ylabel("Distance [m]")
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    save_plot("pulse_evo_3D")
    plt.show()
    os.chdir(ssfm_result_list[0].dirs[0])


def plot_pulse_chirp_2D(ssfm_result_list: list[SSFMResult],
                        nrange: int,
                        dB_cutoff: float,
                        **kwargs):
    """
    Plots local chirp throughout entire fiber span.

    2D colour plot of local chirp throughout entire fiber span with red indicating

    lower frequencies and blue indicating higher ones.

    Parameters
    ----------
    ssfm_result_list : list[SSFMResult]
        List of ssmf_result_class objects corresponding to each fiber segment.
    nrange : int
        How many points on either side of the center do we wish to plot?
    dB_cutoff : float
        Lowest y-value in plot is this many dB smaller than the peak power.
    **kwargs : TYPE
        If chirpPlotRange=(fmin,fmax) is contained in **kwargs, use these
        values to set color scale.

    Returns
    -------
    None.

    """

    timeFreq = ssfm_result_list[0].input_signal.timeFreq

    Nmin = np.max([int(timeFreq.number_of_points / 2 - nrange), 0])
    Nmax = np.min(
        [int(timeFreq.number_of_points / 2 + nrange),
         timeFreq.number_of_points - 1]
    )

    zvals = unpack_Zvals(ssfm_result_list)
    matrix = unpack_matrix(ssfm_result_list, zvals, "pulse")

    # Plot pulse evolution throughout fiber  in normalized log scale
    os.chdir(ssfm_result_list[0].dirs[1])
    fig, ax = plt.subplots(dpi=125)
    ax.set_title("Pulse Chirp Evolution")
    t = timeFreq.t[Nmin:Nmax] * 1e12
    z = zvals
    T, Z = np.meshgrid(t, z)

    Cmatrix = np.ones((len(z), len(t))) * 1.0

    for i in range(len(zvals)):
        Cmatrix[i, :] = get_chirp(t / 1e12, matrix[i, Nmin:Nmax]) / 1e9
    chirpplotrange_set_flag = False
    for kw, value in kwargs.items():
        if kw.lower() == "chirpplotrange" and type(value) == tuple:
            Cmatrix[Cmatrix < value[0]] = value[0]
            Cmatrix[Cmatrix > value[1]] = value[1]
            chirpplotrange_set_flag = True
    if chirpplotrange_set_flag is False:
        Cmatrix[Cmatrix < -50] = -50  # Default fmin = -50GHz
        Cmatrix[Cmatrix > 50] = 50  # Default fmax = -50GHz
    surf = ax.contourf(T, Z, Cmatrix, levels=40, cmap="RdBu")

    ax.set_xlabel("Time [ps]")
    ax.set_ylabel("Distance [m]")
    cbar = fig.colorbar(surf, ax=ax)
    cbar.set_label("Chirp [GHz]")
    save_plot("chirp_evo_2D")
    plt.show()
    os.chdir(ssfm_result_list[0].dirs[0])


def plot_everything_about_pulses(ssfm_result_list: list[SSFMResult],
                                 nrange: int,
                                 dB_cutoff: float,
                                 **kwargs):
    """


    Parameters
    ----------
    ssfm_result_list : list[SSFMResult]
        List of ssmf_result_class objects corresponding to each fiber segment.
    nrange : int
        How many points on either side of the center do we wish to plot?
    dB_cutoff : float
        Lowest y-value in plot is this many dB smaller than the peak power.
    **kwargs : TYPE
        Use keywords and values to skip auxillary plots that
        take a long time to generate.

    Returns
    -------
    None.

    """

    print("  ")
    plot_first_and_last_pulse(ssfm_result_list, nrange, dB_cutoff, **kwargs)
    plot_pulse_matrix_2D(ssfm_result_list, nrange, dB_cutoff)

    for kw, value in kwargs.items():
        if kw.lower() == "show_chirp_plot_flag" and value is True:
            plot_pulse_chirp_2D(ssfm_result_list, nrange, dB_cutoff, **kwargs)
        if kw.lower() == "show_3D_plot_flag" and value is True:
            plot_pulse_matrix_3D(ssfm_result_list, nrange, dB_cutoff)
    print("  ")


def plot_first_and_last_spectrum(ssfm_result_list: list[SSFMResult],
                                 nrange: int,
                                 dB_cutoff: float):
    """
    Plots input spectrum and output spectrum of simulation

    Line plot of input spectrum and output spectrum of SSFM run centered in
    the middle of the time array and with nrange points on either side

    Parameters
    ----------
    ssfm_result_list : list[SSFMResult]
        List of ssmf_result_class objects corresponding to each fiber segment.
    nrange : int
        Determines how many points on either side
        of the center we wish to plot.
    dB_cutoff : float
        Lowest y-value in plot is this many dB smaller than the peak power.

    Returns
    -------
    None.

    """

    timeFreq = ssfm_result_list[0].input_signal.timeFreq
    center_freq_Hz = timeFreq.centerFrequency
    Nmin = np.max([int(timeFreq.number_of_points / 2 - nrange), 0])
    Nmax = np.min(
        [int(timeFreq.number_of_points / 2 + nrange),
         timeFreq.number_of_points - 1]
    )

    zvals = unpack_Zvals(ssfm_result_list)

    P_initial = get_power(ssfm_result_list[0].spectrumMatrix[0, Nmin:Nmax])
    P_final = get_power(ssfm_result_list[-1].spectrumMatrix[-1, Nmin:Nmax])

    Pmax_initial = np.max(P_initial)
    Pmax_final = np.max(P_final)
    Pmax = np.max([Pmax_initial, Pmax_final])

    f = np.flipud((timeFreq.f[Nmin:Nmax] + center_freq_Hz)) / 1e12

    scalingFactor, prefix = get_units(np.max(zvals))
    os.chdir(ssfm_result_list[0].dirs[1])
    fig, ax = plt.subplots(dpi=125)
    ax.set_title("Initial spectrum and final spectrum")
    ax.plot(f, P_initial, label=f"Initial Spectrum at {zvals[0]}{prefix}m")
    ax.plot(
        f, P_final, label=f"Final Spectrum at {zvals[-1]/scalingFactor}{prefix}m")
    ax.axvline(x=center_freq_Hz / 1e12, color="gray", alpha=0.4)
    ax.set_xlabel("Freq. [THz]")
    ax.set_ylabel("PSD [J/Hz]")
    ax.set_yscale("log")
    ax.set_ylim(Pmax / (10 ** (-dB_cutoff / 10)), 2 * Pmax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    save_plot("first_and_last_spectrum")
    plt.show()
    os.chdir(ssfm_result_list[0].dirs[0])


def plot_spectrum_matrix_2D(ssfm_result_list: list[SSFMResult],
                            nrange: int,
                            dB_cutoff: float):
    """
    Plots spectrum calculated by SSFM as colour surface

    2D colour plot of spectrum in freq domain throughout entire fiber span
    normalized to the highest peak power throughout.

    Parameters
    ----------
    ssfm_result_list : list[SSFMResult]
        List of ssmf_result_class objects corresponding to each fiber segment.
    nrange : int
        How many points on either side of the center do we wish to plot?
    dB_cutoff : float
        Lowest y-value in plot is this many dB smaller than the peak power.

    Returns
    -------
    None.

    """

    timeFreq = ssfm_result_list[0].input_signal.timeFreq
    zvals = unpack_Zvals(ssfm_result_list)
    matrix = unpack_matrix(ssfm_result_list, zvals, "spectrum")

    Nmin = np.max([int(timeFreq.number_of_points / 2 - nrange), 0])
    Nmax = np.min(
        [int(timeFreq.number_of_points / 2 + nrange),
         timeFreq.number_of_points - 1]
    )
    center_freq_Hz = timeFreq.centerFrequency

    # Plot pulse evolution throughout fiber in normalized log scale
    os.chdir(ssfm_result_list[0].dirs[1])
    fig, ax = plt.subplots(dpi=125)
    ax.set_title("Spectrum Evolution (dB scale)")
    f = np.flipud((timeFreq.f[Nmin:Nmax] + center_freq_Hz)) / 1e12
    z = zvals
    F, Z = np.meshgrid(f, z)
    Pf = get_power(matrix[:, Nmin:Nmax]) / \
        np.max(get_power(matrix[:, Nmin:Nmax]))
    Pf[Pf < 1e-100] = 1e-100
    Pf = 10 * np.log10(Pf)
    Pf[Pf < dB_cutoff] = dB_cutoff
    surf = ax.contourf(F, Z, Pf, levels=40)
    ax.set_xlabel("Freq. [THz]")
    ax.set_ylabel("Distance [m]")
    cbar = fig.colorbar(surf, ax=ax)
    save_plot("spectrum_evo_2D")
    plt.show()
    os.chdir(ssfm_result_list[0].dirs[0])


def plot_spectrum_matrix_3D(ssfm_result_list: list[SSFMResult],
                            nrange: int,
                            dB_cutoff: float):
    """
    Plots spectrum calculated by SSFM as 3D colour surface

    3D colour plot of signal spectrum in freq domain throughout
    entire fiber span normalized to the highest peak power throughout.

    Parameters
    ----------
    ssfm_result_list : list[SSFMResult]
        List of ssmf_result_class objects corresponding to each fiber segment.
    nrange : int
        How many points on either side of the center do we wish to plot?
    dB_cutoff : float
        Lowest y-value in plot is this many dB smaller than the peak power.

    Returns
    -------
    None.

    """

    timeFreq = ssfm_result_list[0].input_signal.timeFreq
    zvals = unpack_Zvals(ssfm_result_list)
    matrix = unpack_matrix(ssfm_result_list, zvals, "spectrum")

    Nmin = np.max([int(timeFreq.number_of_points / 2 - nrange), 0])
    Nmax = np.min(
        [int(timeFreq.number_of_points / 2 + nrange),
         timeFreq.number_of_points - 1]
    )
    center_freq_Hz = timeFreq.centerFrequency

    # Plot pulse evolution in 3D
    os.chdir(ssfm_result_list[0].dirs[1])
    fig, ax = plt.subplots(1, 1, figsize=(
        10, 7), subplot_kw={"projection": "3d"})
    plt.title("Spectrum Evolution (dB scale)")

    f = np.flipud((timeFreq.f[Nmin:Nmax] + center_freq_Hz)) / 1e12
    z = zvals
    F_surf, Z_surf = np.meshgrid(f, z)
    P_surf = get_power(matrix[:, Nmin:Nmax]) / \
        np.max(get_power(matrix[:, Nmin:Nmax]))
    P_surf[P_surf < 1e-100] = 1e-100
    P_surf = 10 * np.log10(P_surf)
    P_surf[P_surf < dB_cutoff] = dB_cutoff
    # Plot the surface.
    surf = ax.plot_surface(
        F_surf, Z_surf, P_surf, cmap=cm.viridis, linewidth=0, antialiased=False
    )
    ax.set_xlabel("Freq. [GHz]")
    ax.set_ylabel("Distance [m]")
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    save_plot("spectrum_evo_3D")
    plt.show()
    os.chdir(ssfm_result_list[0].dirs[0])


def plot_everything_about_spectra(ssfm_result_list: list[SSFMResult],
                                  nrange: int,
                                  dB_cutoff: float,
                                  **kwargs):
    """
    Generates all plots of pulse amplitudes throughout fiber span

    Calls plot_first_and_last_spectrum, plot_spectrum_matrix_2D and
    plot_spectrum_matrix_3D sequentially and saves the plots in the
    appropriate directory

    Parameters
    ----------
    ssfm_result_list : list[SSFMResult]
        List of ssmf_result_class objects corresponding to each fiber segment.
    nrange : int
        How many points on either side of the center do we wish to plot?
    dB_cutoff : float
        Lowest y-value in plot is this many dB smaller than the peak power.
    **kwargs : TYPE
        If 'show_3D_plot_flag'=True is selected, make 3D plot of the spectrum.

    Returns
    -------
    None.

    """

    print("  ")
    plot_first_and_last_spectrum(ssfm_result_list, nrange, dB_cutoff)
    plot_spectrum_matrix_2D(ssfm_result_list, nrange, dB_cutoff)

    for kw, value in kwargs.items():
        if kw.lower() == "show_3D_plot_flag" and value is True:
            plot_spectrum_matrix_3D(ssfm_result_list, nrange, dB_cutoff)
    print("  ")


def make_chirp_gif(ssfm_result_list: list[SSFMResult],
                   nrange: int,
                   chirpRange_GHz: list[float] = [-20, 20],
                   framerate: int = 30):
    """
    Animate pulse evolution as .gif and show local chirp

    Animate pulse power evolution and show local chirp by changing line color.
    Saves result as .gif file.
    Note: Producing the animation can take a several minutes on a regular
    PC, so please be patient.

    Parameters
    ----------
    ssfm_result_list : list[SSFMResult]
        List of ssmf_result_class objects corresponding to each fiber segment.
    nrange : int
        How many points on either side of the center do we wish to plot?
    chirpRange_GHz : list[float], optional
        Min and Max frequency values in GHz to determine line
        color. The default is [-20, 20].
    framerate : int, optional
        Framerate of .gif animation. May want to reduce this number for
        simulations with few steps. The default is 30.

    Returns
    -------
    None.

    """

    print(
        "Making .gif anination of pulse evolution. This may "
        "take a while, so please be patient."
    )

    os.chdir(ssfm_result_list[0].dirs[1])

    print(f"The .gif animation will be saved in {os.getcwd()}")

    timeFreq = ssfm_result_list[0].input_signal.timeFreq
    zvals = unpack_Zvals(ssfm_result_list)
    matrix = unpack_matrix(ssfm_result_list, zvals, "pulse")
    scalingFactor, letter = get_units(np.max(zvals))

    Nmin = np.max([int(timeFreq.number_of_points / 2 - nrange), 0])
    Nmax = np.min(
        [int(timeFreq.number_of_points / 2 + nrange),
         timeFreq.number_of_points - 1]
    )

    Tmin = timeFreq.t[Nmin]
    Tmax = timeFreq.t[Nmax]

    points = np.array(
        [timeFreq.t * 1e12, get_power(matrix[len(zvals) - 1, Nmin:Nmax])],
        dtype=object
    ).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[0:-1], points[1:]], axis=1)

    # Make custom colormap
    colors = ["red", "gray", "blue"]
    cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)

    # Initialize color normalization function
    norm = plt.Normalize(chirpRange_GHz[0], chirpRange_GHz[1])

    # Initialize line collection to be plotted
    lc = LineCollection(segments, cmap=cmap1, norm=norm)
    lc.set_array(
        get_chirp(timeFreq.t[Nmin:Nmax],
                  matrix[len(zvals) - 1, Nmin:Nmax]) / 1e9
    )

    # Initialize figure
    fig, ax = plt.subplots(dpi=125)
    line = ax.add_collection(lc)
    fig.colorbar(line, ax=ax, label="Chirp [GHz]")

    Pmax = np.max(np.abs(matrix)) ** 2

    # Function for specifying axes

    def init():

        ax.set_xlim([Tmin * 1e12, Tmax * 1e12])
        ax.set_ylim([0, 1.05 * Pmax])

        ax.set_xlabel("Time [ps]")
        ax.set_ylabel("Power [W]")

    # Function for updating the plot in the .gif

    def update(i: int):
        ax.clear()  # Clear figure
        init()  # Reset axes  {num:{1}.{5}}  np.round(,2)
        ax.set_title(
            f"Pulse evolution, z = {zvals[i]/scalingFactor:.2f}{letter}m")

        # Make collection of points from pulse power
        points = np.array(
            [timeFreq.t[Nmin:Nmax] * 1e12, get_power(matrix[i, Nmin:Nmax])],
            dtype=object
        ).T.reshape(-1, 1, 2)

        # Make collection of lines from points
        segments = np.concatenate([points[0:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap1, norm=norm)

        # Activate norm function based on local chirp

        lc.set_array(
            get_chirp(timeFreq.t[Nmin:Nmax], matrix[i, Nmin:Nmax]) / 1e9)
        # Plot line
        line = ax.add_collection(lc)

    # Make animation
    ani = FuncAnimation(fig, update, range(len(zvals)), init_func=init)
    plt.show()

    # Save animation as .gif

    writer = PillowWriter(fps=int(framerate))
    ani.save(
        f"{ssfm_result_list[0].experimentName}_fps={int(framerate)}.gif",
        writer=writer)

    os.chdir(ssfm_result_list[0].dirs[0])


def get_average(time_or_freq: npt.NDArray[float],
                pulse_or_spectrum: npt.NDArray[complex]) -> float:
    """
    Computes central time (or frequency) of pulse (spectrum)

    Computes central time (or frequency) of pulse (spectrum) by calculating
    the 'expectation value'.

    Parameters
    ----------
    time_or_freq : npt.NDArray[float]
        Time range in seconds or freq. range in Hz.
    pulse_or_spectrum : npt.NDArray[complex]
        Temporal or spectral amplitude.

    Returns
    -------
    meanValue : float
        average time or frequency.

    """

    E = get_energy(time_or_freq, pulse_or_spectrum)
    meanValue = np.trapz(
        time_or_freq * get_power(pulse_or_spectrum), time_or_freq) / E
    return meanValue


def get_variance(time_or_freq: npt.NDArray[float],
                 pulse_or_spectrum: npt.NDArray[complex]) -> float:
    """
    Computes variance of pulse or spectrum

    Computes variance of pulse in time domain or freq domain via
    <x**2>-<x>**2

    Parameters
    ----------
    time_or_freq : npt.NDArray[float]
        Time range in seconds or freq. range in Hz.
    pulse_or_spectrum : npt.NDArray[complex]
        Temporal or spectral amplitude.

    Returns
    -------
    variance : float
        variance in time or frequency domains.

    """
    E = get_energy(time_or_freq, pulse_or_spectrum)
    variance = (
        np.trapz(time_or_freq ** 2 *
                 get_power(pulse_or_spectrum), time_or_freq) / E
        - (get_average(time_or_freq, pulse_or_spectrum)) ** 2
    )
    return variance


def get_stdev(time_or_freq: npt.NDArray[float],
              pulse_or_spectrum: npt.NDArray[complex]) -> float:
    """
    Computes standard deviation of pulse or spectrum

    Computes std of pulse in time domain or freq domain via
    sqrt(<x**2>-<x>**2)

    Parameters
    ----------
    time_or_freq : npt.NDArray[float]
        Time range in seconds or freq. range in Hz.
    pulse_or_spectrum : npt.NDArray[complex]
        Temporal or spectral amplitude.

    Returns
    -------
    stdev : float
        Stdev in time or frequency domains.

    """

    stdev = np.sqrt(get_variance(time_or_freq, pulse_or_spectrum))
    return stdev


def plot_avg_and_std_of_time_and_freq(ssfm_result_list: list[SSFMResult]):
    """
    Plots how spectral and temporal width of signal change with distance

    Uses get_average and get_stdev to create dual-axis
    line plot of temporal and spectral center and widths throughout fiber span.
    Saves plot in appropriate folder.

    Parameters
    ----------
    ssfm_result_list : list[SSFMResult]
        List of ssmf_result_class objects corresponding to each fiber segment.

    Returns
    -------
    None.

    """

    timeFreq = ssfm_result_list[0].input_signal.timeFreq
    center_freq_Hz = timeFreq.centerFrequency
    zvals = unpack_Zvals(ssfm_result_list)

    pulseMatrix = unpack_matrix(ssfm_result_list, zvals, "pulse")
    spectrumMatrix = unpack_matrix(
        ssfm_result_list, zvals, "spectrum")

    meanTimeArray = np.zeros(len(zvals)) * 1.0
    meanFreqArray = np.copy(meanTimeArray)
    stdTimeArray = np.copy(meanTimeArray)
    stdFreqArray = np.copy(meanTimeArray)
    f = np.flipud((timeFreq.f))

    i = 0
    for pulse, spectrum in zip(pulseMatrix, spectrumMatrix):

        meanTimeArray[i] = get_average(timeFreq.t, pulse)
        meanFreqArray[i] = get_average(f, spectrum)

        stdTimeArray[i] = get_stdev(timeFreq.t, pulse)
        stdFreqArray[i] = get_stdev(f, spectrum)

        i += 1
    scalingFactor_Z, prefix_Z = get_units(np.max(zvals))
    maxCenterTime = np.max(np.abs(meanTimeArray))
    maxStdTime = np.max(stdTimeArray)

    scalingFactor_pulse, prefix_pulse = get_units(
        np.max([maxCenterTime, maxStdTime])
    )
    scalingFactor_spectrum, prefix_spectrum = get_units(
        np.max([meanFreqArray, stdFreqArray])
    )

    os.chdir(ssfm_result_list[0].dirs[1])
    fig, ax = plt.subplots(dpi=125)
    plt.title("Evolution of temporal/spectral widths and centers")
    ax.plot(zvals / scalingFactor_Z, meanTimeArray /
            scalingFactor_pulse, label="Pulse")

    ax.fill_between(
        zvals / scalingFactor_Z,
        (meanTimeArray - stdTimeArray) / scalingFactor_pulse,
        (meanTimeArray + stdTimeArray) / scalingFactor_pulse,
        alpha=0.3,
        color="C0",
        label="1$\sigma$ width",
    )

    ax.set_xlabel(f"Distance [{prefix_Z}m]")
    ax.set_ylabel(f"Time [{prefix_pulse}s]", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    ax.set_ylim(
        timeFreq.tmin / scalingFactor_pulse,
        timeFreq.tmax / scalingFactor_pulse
    )

    ax2 = ax.twinx()
    ax2.plot(
        zvals / scalingFactor_Z,
        meanFreqArray / scalingFactor_spectrum,
        "C1-",
        label=f"Spectrum Center rel. to $f_c$={center_freq_Hz/1e12:.5}THz ",
    )
    ax2.fill_between(
        zvals / scalingFactor_Z,
        (meanFreqArray - stdFreqArray) / scalingFactor_spectrum,
        (meanFreqArray + stdFreqArray) / scalingFactor_spectrum,
        alpha=0.3,
        color="C1",
        label="1$\sigma$ width",
    )

    ax2.set_ylim(
        timeFreq.fmin / scalingFactor_spectrum,
        timeFreq.fmax / scalingFactor_spectrum
    )
    ax2.set_ylabel(f"Freq. [{prefix_spectrum}Hz]", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=2,
        fancybox=True,
        shadow=True,
    )

    save_plot("Width_evo")
    plt.show()
    os.chdir(ssfm_result_list[0].dirs[0])


def plot_everything_about_result(
    ssfm_result_list: list[SSFMResult],
    nrange_pulse: int,
    dB_cutoff_pulse: float,
    nrange_spectrum: int,
    dB_cutoff_spectrum: float,
    **kwargs,
):
    """
    Generates all plots of pulse amplitudes, spectra etc. throughout fiber span

    Calls   plot_avg_and_std_of_time_and_freq, plot_everything_about_pulses and
    plot_everything_about_spectra sequentially, saving plots in the appropriate
    directory

    Parameters
    ----------
    ssfm_result_list : list[SSFMResult]
        List of ssmf_result_class objects corresponding to each fiber segment.
    nrange_pulse : int
        For pulse plots, determines how many points on either
        side of the center we wish to plot.
    dB_cutoff_pulse : float
        For pulse plots, lowest y-value in plot is this many
        dB smaller than the peak power.
    nrange_spectrum : int
        For spectrum plots, determines how many points on either
        side of the center we wish to plot.
    dB_cutoff_spectrum : float
        For spectrum plots, lowest y-value in plot is this many
        dB smaller than the peak power.
    **kwargs : TYPE
        Various keyword arguments.

    Returns
    -------
    None.

    """
    plot_avg_and_std_of_time_and_freq(ssfm_result_list)
    plot_everything_about_pulses(
        ssfm_result_list, nrange_pulse, dB_cutoff_pulse, **kwargs)

    plot_everything_about_spectra(
        ssfm_result_list, nrange_spectrum, dB_cutoff_spectrum, **kwargs
    )


# TODO: implement wavelet diagram
def waveletTest(M, s):

    w = 1
    x = np.arange(0, M) - (M - 1.0) / 2
    x = x / s
    wavelet = np.exp(1j * w * x) * np.exp(-0.5 * x ** 2) * np.pi ** (-0.25)
    output = np.sqrt(1 / s) * wavelet
    return output


def waveletTransform(
    timeFreq: TimeFreq, pulse, nrange_pulse, nrange_spectrum, dB_cutoff
):

    Nmin_pulse = np.max([int(timeFreq.number_of_points / 2 - nrange_pulse), 0])
    Nmax_pulse = np.min(
        [
            int(timeFreq.number_of_points / 2 + nrange_pulse),
            timeFreq.number_of_points - 1,
        ]
    )

    Tmax = timeFreq.t[Nmax_pulse]

    t = timeFreq.t[Nmin_pulse:Nmax_pulse]

    wavelet_durations = np.linspace((t[1] - t[0]) * 10, Tmax, 1000)

    print((t[1] - t[0]) * 100, Tmax)
    print(1 / Tmax / 1e9, 1 / ((t[1] - t[0]) * 100) / 1e9)

    dt_wavelet = wavelet_durations[1] - wavelet_durations[0]

    plt.figure()
    plt.plot(t, np.real(pulse[Nmin_pulse:Nmax_pulse]))
    plt.plot(t, np.imag(pulse[Nmin_pulse:Nmax_pulse]))
    plt.show()

    plt.figure()
    plt.plot(t, get_chirp(t, pulse[Nmin_pulse:Nmax_pulse]) / 1e9)
    plt.ylabel("Chirp [GHz]")
    plt.show()

    cwtmatr = signal.cwt(
        pulse[Nmin_pulse:Nmax_pulse],
        signal.morlet2,
        wavelet_durations,
        dtype=complex
    )

    Z = np.abs(cwtmatr) ** 2
    print(np.max(Z))
    Z /= np.max(Z)

    Z[Z < 10 ** (dB_cutoff / 10)] = 10 ** (dB_cutoff / 10)

    fig, ax = plt.subplots(dpi=125)
    ax.set_title("Wavelet transform of final pulse")
    T, F = np.meshgrid(t, 1 / wavelet_durations)

    surf = ax.contourf(T / 1e-12, F / 1e9, Z, levels=40)
    ax.set_xlabel("Time. [ps]")
    ax.set_ylabel("Freq. [GHz]")
    cbar = fig.colorbar(surf, ax=ax)
    save_plot("wavelet_final")
    plt.show()


def dB_to_lin(Val_dB: float) -> float:
    """
    Converts value in dB to value in linear scale

    Parameters
    ----------
    Val_dB : float
        Value in dB.

    Returns
    -------
    float
        Value in decimal.

    """
    return 10 ** (Val_dB / 10)


def lin_to_dB(Val_lin: float) -> float:
    """
    Converts value in linear scale to value in dB

    Parameters
    ----------
    Val_lin : float
        Value in decimal.

    Returns
    -------
    float
        Value in dB.

    """
    return 10 * np.log10(Val_lin)


def wavelength_to_freq(wavelength_m: float) -> float:
    """
    Converts wavelength in m to frequency in Hz

    Converts wavelength in m to frequency in Hz using f=c/lambda

    Parameters:
        wavelength_m (float): Wavelength in m

    Returns:
        float: Frequency in Hz
    """
    return LIGHTSPEED_M_PER_S / wavelength_m


def freq_to_wavelength(freq_Hz: float) -> float:
    """
    Converts frequency in Hz to wavelength in m

    Converts frequency in Hz to wavelength in m using lambda=c/f

    Parameters:
        freq_Hz (float): Frequency in Hz

    Returns:
        float: Wavelength in m
    """
    return LIGHTSPEED_M_PER_S / freq_Hz


def wavelength_BW_to_freq_BW(wavelength_m: float,
                             wavelengthBW_m: float
                             ) -> float:
    """
    Converts bandwidth in m to bandwidth in Hz

    A signal centered at lambda_0 with bandwidth specified in terms
    of wavelength will have a frequency bandwidth of c*lambda_BW/lambda**2

    Parameters:
        wavelength_m   (float): Wavelength in m
        wavelengthBW_m (float): Wavelength bandwidth in m

    Returns:
        float: Frequency bandwidth in Hz
    """
    return LIGHTSPEED_M_PER_S * wavelengthBW_m / wavelength_m ** 2


def freq_BW_to_wavelength_BW(freq_Hz: float, freqBW_Hz: float) -> float:
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
    return LIGHTSPEED_M_PER_S * freqBW_Hz / freq_Hz ** 2


def get_gamma_from_fiber_params(wavelength_m: float,
                                n2_m2_W: float,
                                coreDiameter_m: float) -> float:
    """
    Calculates the nonlinear fiber parameter, gamma in 1/W/m.

    Parameters
    ----------
    wavelength_m : float
        Wavelength in m for which we want gamma.
    n2_m2_W : float
        Nonlinear refractive index of the material.
    coreDiameter_m : float
        Diameter of fiber core.

    Returns
    -------
    float
        gamma in 1/W/m.

    """
    return 2 * pi / wavelength_m * n2_m2_W / (pi * coreDiameter_m ** 2 / 4)


def extract_spectrum_range(freqList: npt.NDArray[float],
                           spectralAmplitude: npt.NDArray[complex],
                           freq1: float,
                           freq2: float) -> npt.NDArray[complex]:
    """
    Takes in a spectrum and extracts only values that fall in a certain range
    by setting all other ones to zero.

    Parameters
    ----------
    freqList : np.array([float])
        Absolute frequency.
    spectralAmplitude : np.array([complex])
        Spectral amplitude.
    freq1 : float
        Minimum frequency to be extracted.
    freq2 : float
        Maximum frequency to be extracted.

    Returns
    -------
    outputArray : np.array([complex])
        Same as spectralAmplitude but with all entries corresponding to
        frequencies outside the [freq1,freq2] range set to zero.

    """

    assert (
        freq1 < freq2
    ), (f"Error: freq1 must be smaller than freq2,"
        f" but freq1 = {freq1}>= {freq2} = freq2 ")

    array1 = np.abs(freqList - freq1)
    index1 = array1.argmin()

    array2 = np.abs(freqList - freq2)
    index2 = array2.argmin()

    outputArray = np.zeros_like(spectralAmplitude) * 1j

    outputArray[index1:index2] = spectralAmplitude[index1:index2]

    return outputArray


def get_value_at_freq(freqList: npt.NDArray[float],
                      freqOfInterest: float,
                      array: npt.NDArray[complex]) -> complex:
    """
    Helper function to extract value from array. For a discretized spectrum,
    array, at frequencies freqList, this function allows one to extract
    the entry in array, which is closes to the one specified by freqOfInterest.

    Parameters
    ----------
    freqList : npt.NDArray[float]
        Absolute frequency.
    freqOfInterest : float
        Absolute frequency at which we want the value of array.
    array : npt.NDArray[complex]
        Array from which we wish to extract a value.

    Returns
    -------
    extracted_value : complex
        Extracted value.

    """
    absArray = np.abs(freqList - freqOfInterest)
    index = absArray.argmin()
    extracted_value = array[index]
    return extracted_value


def get_current_SNR_dB(freqs: npt.NDArray[float],
                       spectrum: npt.NDArray[complex],
                       channel: ChannelClass,
                       freqTol: float = 0.05) -> float:
    """
    Get SNR of channel in spectrum

    Computes the SNR of the specified channel in the spectrum by integrating
    up the energy in the signal and dividing by the integrated noise energy.
    Note that interpolation is used for the noise inside the signal range.


    Parameters
    ----------
    freqs : npt.NDArray[float]
        Absolute frequencies of the spectrum.
    spectrum : npt.NDArray[complex]
        Amplitude spectrum of signal whose SNR we want.
    channel : ChannelClass
        Specific channel whose SNR value we want.
    freqTol : float, optional
        When computing the noise inside the signal BW, use the PSD evaluated at
        (1-freqTol) of the distance between the lower freq of the channel and
        the lower freq of the signal. The default is 0.05.

    Returns
    -------
    SNR_i_dB : float
        SNR value in dB of the channel for the specified spectrum.

    """

    freqTol = np.abs(freqTol)

    assert freqTol < 1, (f"ERROR: freqTol = {freqTol}, but should "
                         "be smaller than 1!")

    signal = extract_spectrum_range(
        freqs, spectrum, channel.signalMinFreq_Hz, channel.signalMaxFreq_Hz
    )
    signalEnergy = get_energy(freqs, signal)

    noiseBelow = extract_spectrum_range(
        freqs, spectrum, channel.channelMinFreq_Hz, channel.signalMinFreq_Hz
    )
    noiseAbove = extract_spectrum_range(
        freqs, spectrum, channel.signalMaxFreq_Hz, channel.channelMaxFreq_Hz
    )

    noiseEnergyBelow = get_energy(freqs, noiseBelow)
    noiseEnergyAbove = get_energy(freqs, noiseAbove)

    leftEdgeNoisePSD = (
        np.abs(
            get_value_at_freq(
                freqs,
                channel.channelMinFreq_Hz + channel.leftGap_Hz * (1 - freqTol),
                noiseBelow,
            )
        )
        ** 2
    )
    rightEdgeNoisePSD = (
        np.abs(
            get_value_at_freq(
                freqs,
                channel.channelMaxFreq_Hz -
                channel.rightGap_Hz * (1 - freqTol),
                noiseAbove,
            )
        )
        ** 2
    )

    signalBW = channel.signalBW_Hz

    slope = (rightEdgeNoisePSD - leftEdgeNoisePSD) / signalBW
    offset = leftEdgeNoisePSD

    noiseEnergyInside = 0.5 * slope * signalBW ** 2 + offset * signalBW

    totalNoiseEnergy = noiseEnergyBelow + noiseEnergyInside + noiseEnergyAbove

    SNR_i_dB = 10 * np.log10(signalEnergy / totalNoiseEnergy)

    return SNR_i_dB


def get_channel_SNR_dB(ssfm_result_list: list[SSFMResult],
                       channel: ChannelClass,
                       freqTol: float = 0.05
                       ) -> [npt.NDArray[float], npt.NDArray[float]]:
    """
    Calculates SNR throughout fiber span for a given channel

    Parameters
    ----------
    ssfm_result_list : list[SSFMResult]
        List of SSFMResult objects containing signal info for each fiber.
    channel_list : list[ChannelClass]
        List of ChannelClass objects describing the min, center and max
        freqs of each channel.
    freqTol : float
        When computing the noise inside the signal BW, use the PSD evaluated at
        (1-freqTol) of the distance between the lower freq of the channel and
        the lower freq of the signal. The default is 0.05.

    Returns
    -------
    zvals : np.array([float])
        z-positions of each signal.
    outputArray : np.array([float])
        SNR value at each z-position.

    """

    zvals = unpack_Zvals(ssfm_result_list)
    timeFreq = ssfm_result_list[0].input_signal.timeFreq
    spectrumMatrix = unpack_matrix(
        ssfm_result_list,
        zvals,
        "spectrum")
    freqs = timeFreq.f + timeFreq.centerFrequency

    outputArray = np.zeros_like(zvals) * 1.0

    for i, spectrum in enumerate(spectrumMatrix):
        outputArray[i] = get_current_SNR_dB(
            freqs, spectrum, channel, freqTol=freqTol)
    return zvals, outputArray


def get_final_SNR_dB(ssfm_result_list: list[SSFMResult],
                     channel_list: list[ChannelClass],
                     freqTol: float = 0.05) -> npt.NDArray[float]:
    """
    Calculates SNR for all channels at output of fiber span

    Parameters
    ----------
    ssfm_result_list : list[SSFMResult]
        List of SSFMResult objects containing signal info for each fiber.
    channel_list : list[ChannelClass]
        List of ChannelClass objects describing the min, center and max freqs
        of each channel.
    freqTol : float
        When computing the noise inside the signal BW, use the PSD evaluated at
        (1-freqTol) of the distance between the lower freq of the channel and
        the lower freq of the signal. The default is 0.05.

    Returns
    -------
    outputArray : np.array([float])
        Array containing the SNR value at link end for each channel

    """

    freqs = (
        ssfm_result_list[0].input_signal.timeFreq.f
        + ssfm_result_list[0].input_signal.timeFreq.centerFrequency
    )
    finalSpectrum = ssfm_result_list[-1].spectrumMatrix[-1, :]

    outputArray = np.zeros(len(channel_list))
    for i, currentChannel in enumerate(channel_list):
        outputArray[i] = get_current_SNR_dB(
            freqs, finalSpectrum, currentChannel, freqTol=freqTol
        )
    return outputArray


def plot_final_SNR_dB(ssfm_result_list: list[SSFMResult],
                      channel_list: list,
                      freqTol: float = 0.05):
    """
    Plots the SNR at the output of a fiber span

    Parameters
    ----------
    ssfm_result_list : list[SSFMResult]
        List of SSFMResult objects containing signal info for each fiber.
    channel_list : list[ChannelClass]
        List of ChannelClass objects describing the min, center and max
        freqs of each channel.
    freqTol : float
        When computing the noise inside the signal BW, use the PSD evaluated at
        (1-freqTol) of the distance between the lower freq of the channel and
        the lower freq of the signal. The default is 0.05.

    Returns
    -------
    None.

    """
    os.chdir(ssfm_result_list[0].dirs[1])
    signalCenterFreq_list = np.zeros(len(channel_list))

    fig, ax = plt.subplots(dpi=125)
    for i, channel in enumerate(channel_list):
        signalCenterFreq_list[i] = channel.signalCenterFreq_Hz / 1e12
        ax.axvline(x=channel.channelMinFreq_Hz /
                   1e12, color="gray", alpha=0.35)
        ax.axvline(x=channel.channelMaxFreq_Hz /
                   1e12, color="gray", alpha=0.35)
    finalSNR_dB = get_final_SNR_dB(ssfm_result_list, channel_list, freqTol)

    ax.set_title("Final Nonlinear SNR")
    ax.plot(signalCenterFreq_list, finalSNR_dB, ".")
    ax.set_xlabel("Freq [THz]")
    ax.set_ylabel("$SNR_{NL}$ [dB]")
    ax.grid()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_plot("SNR_final")
    plt.show()
    os.chdir(ssfm_result_list[0].dirs[0])


def plot_SNR_for_channels(
    ssfm_result_list: list[SSFMResult],
    channel_list: list[ChannelClass],
    channelNumber_list: list[int],
    **kwargs
):
    """
    Plots the SNR for the specified channels throughout the fiber span.

    Parameters
    ----------
    ssfm_result_list : list[SSFMResult]
        List of SSFMResult objects containing signal info for each fiber.
    channel_list : list[ChannelClass]
        List of ChannelClass objects describing the min, center and max
        freqs of each channel.
    channelNumber_list : list[int]
        List of channel numbers to be plotted.
    **kwargs : Optional
        Optional keyword arguments to set ylims of plot.

    Returns
    -------
    None.

    """

    os.chdir(ssfm_result_list[0].dirs[1])
    fig, ax = plt.subplots(dpi=125)
    ax.set_title("Evolution of SNR")

    distance_so_far = 0.0
    for result in ssfm_result_list:
        distance_so_far += result.fiber.Length
        ax.axvline(x=distance_so_far / 1e3, color="black",
                   linestyle="--", alpha=1.0)
    for idx, channelNumber in enumerate(channelNumber_list):

        channel = channel_list[channelNumber]

        z, SNR = get_channel_SNR_dB(ssfm_result_list, channel)

        ax.plot(z / 1e3,
                SNR,
                ".",
                color=f"C{idx}",
                label=f"Ch# {channelNumber} ")

        ax.axhline(
            y=SNR[-1],
            color=f"C{idx}",
            alpha=0.4,
            label=f"Final SNR = {SNR[-1]:.2f}"
        )
    ax.set_xlabel("Distance [km]")
    ax.set_ylabel("SNR [dB]")

    for kw, value in kwargs.items():
        if kw.lower() == "ylims":
            ax.set_ylim(value[0], value[1])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid()

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.3),
        ncol=len(channelNumber_list),
        fancybox=True,
        shadow=True,
    )

    save_plot("SNR_plot")
    plt.show()
    os.chdir(ssfm_result_list[0].dirs[0])


if __name__ == "__main__":

    os.chdir(os.path.realpath(os.path.dirname(__file__)))

    Trange = 2*1.6e-12
    N = 2 ** 12  # Number of points
    dt = Trange/N  # Time resolution [s]

    centerFreq_test = FREQ_CENTER_C_BAND_HZ*4
    centerWavelength = freq_to_wavelength(centerFreq_test)  # laser wl in m

    timeFreq_test = TimeFreq(N, dt, centerFreq_test)

    # []#[0, -24e-39]  # Dispersion in units of s^(entry+2)/m
    beta_list = [0*1e4*1e-30]

    fiber_diameter = 9e-6  # m
    n2_silica = 30e-21  # m**2/W

    #
    gamma_test = get_gamma_from_fiber_params(
        centerWavelength, n2_silica, fiber_diameter)

    #  Initialize fibers
    alpha_test = 0

    number_of_steps = 2**13
    testDuration = 100e-15
    length_test = 13
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
    test_FFT_tol = 1e-3
    testTimeOffset = 0  # Time offset
    testFreqOffset = 0  # Freq offset from center frequency

    testChirp = 0
    testPulseType = "gauss"
    testOrder = 1
    testNoiseAmplitude = 0

    # 2*np.sqrt(np.abs(beta_list[0])/gamma_test/testDuration**2) #np.sqrt(1e-9 /(testDuration))  # Amplitude in units of sqrt(W)
    testAmplitude = 32.320897717179356

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

    expName = "ThirdOrderDisp_test"

    # Run SSFM
    ssfm_result_list = SSFM(
        fiber_link,
        testInputSignal,
        showProgressFlag=True,
        experimentName=expName,
        FFT_tol=test_FFT_tol
    )

    expName = "SelfSteepening_test"

    fiber_beta3 = FiberSpan(
        length_test,
        number_of_steps,
        0,
        [0, 24e-41],
        alpha_test,
        use_self_steepening=False)

    fiber_list_beta3 = [fiber_beta3]
    fiber_link_beta3 = FiberLink(fiber_list_beta3)

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

    fig, ax = plt.subplots(dpi=300)
    t = timeFreq_test.t
    pul = testInputSignal.amplitude
    ax.plot(t*1e12, get_phase(pul))
    ax.set_xlabel('Time [ps]')
    ax.set_ylabel('Phase [rad]')
    plt.show()

    fig, ax = plt.subplots(dpi=300)
    f = timeFreq_test.f
    spec = testInputSignal.spectrum
    ax.plot(f/1e12, get_phase(spec),'.')
    ax.set_xlim(-50,50)
    ax.set_xlabel('Freq. [THz]')
    ax.set_ylabel('Phase [rad]')
    plt.show()

    fig, ax = plt.subplots(dpi=300)
    f = timeFreq_test.f
    spec = testInputSignal.spectrum
    ax.plot(f/1e12, get_power(spec),'.')
    ax.set_xlim(-50,50)
    ax.set_xlabel('Freq. [THz]')
    ax.set_ylabel('Energy dens. [J/Hz]')
    plt.show()



    # ssfm_result_list_beta3 = SSFM(
    #     fiber_link_beta3,
    #     testInputSignal,
    #     showProgressFlag=True,
    #     experimentName=expName,
    #     FFT_tol=test_FFT_tol
    # )

    nrange = 200*4
    dB_cutoff = -60

    plot_everything_about_result(
        ssfm_result_list, nrange, dB_cutoff, nrange, dB_cutoff)
    plot_everything_about_result(
        ssfm_result_list_beta3, nrange, dB_cutoff, nrange, dB_cutoff)

    fig, ax = plt.subplots(dpi=300)
    f = ssfm_result_list_beta3[0].input_signal.timeFreq.f
    spec = ssfm_result_list_beta3[0].spectrumMatrix[-1, :]
    ax.plot(f/1e12, get_phase(spec))
    ax.set_xlabel('Freq. [THz]')
    ax.set_ylabel('Phase [rad]')

    # ax.plot(f/1e12,np.real(spec))
    # ax.plot(f/1e12,np.imag(spec))
    ax.set_xlim(-20, 20)
    plt.show()

    fig, ax = plt.subplots(dpi=300)
    f = ssfm_result_list_beta3[0].input_signal.timeFreq.f
    spec = ssfm_result_list_beta3[0].spectrumMatrix[-1, :]
    # ax.plot(f/1e12,get_phase(spec))

    ax.plot(f/1e12, np.real(spec), '.')
    ax.plot(f/1e12, np.imag(spec), '.')
    ax.set_xlim(-20, 20)
    plt.show()

    # plot_pulse_chirp_2D(ssfm_result_list, nrange, dB_cutoff)
    # plot_first_and_last_spectrum(ssfm_result_list, nrange, dB_cutoff)
    plot_first_and_last_pulse(ssfm_result_list, nrange, dB_cutoff)
    plot_first_and_last_pulse(ssfm_result_list_beta3, nrange, dB_cutoff)
