# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 14:26:22 2024

@author: okrarup
"""


# ********Note on sign convention********
#
# This code uses the exp(-1j*omega*t) sign convention because
# exp(1j(beta*z-omega*t)) represents a plane wave propagating in the
# positive z-direction. A disadvantage of this convention is that
# "more negative" frequencies correspond to frequencies that are "higher"
# and "more blue". In turn, this occasionally necessitates changing the
# sign of the frequency axis, such as when 3rd order dispersion is applied.

from copy import deepcopy
import os
from dataclasses import dataclass, field
from typing import TextIO, Callable
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

FREQ_1550_NM_Hz = 193414489032258.06
FREQ_1310_NM_HZ = 228849204580152.7

BETA2_AT_1550_NM_TYPICAL_SMF_S2_PER_M = -23e-27
BETA2_AT_1625_NM_TYPICAL_SMF_S2_PER_M = -28.1e-27

ALPHA_AT_1310_NM_TYPICAL_SMF_DB_PER_M = -0.3/1e3
ALPHA_AT_1550_NM_TYPICAL_SMF_DB_PER_M = -0.22/1e3

PULSE_TYPE_LIST = ["random",
                   "gaussian",
                   "general_gaussian",
                   "sech",
                   "square",
                   "sqrt_triangle",
                   "sqrt_parabola",
                   "sinc",
                   "raised_cosine",
                   "CW",
                   "custom"]



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

@dataclass
class Channel:
    """
    Class for storing info about a certain frequency channel.

    Attributes:
        self.channel_center_freq_Hz : float
            Central frequency of channel.
        self.channel_min_freq_Hz : float
            Lower frequency of channel.
        self.channel_max_freq_Hz : float
            Upper frequency of channel.
        self.channel_width_Hz : float
            Full channel width.


        self.signal_center_freq_Hz : float
            Central frequency of signal.
        self.signal_bw_Hz : float
            Signal Bandwidth.
        self.signal_min_freq_Hz : float
            Lower frequency of the signal in the channel.
        self.signal_max_freq_Hz : float
            Upper frequency of the signal in the channel.

        self.left_gap_Hz : float
            Frequency gap between lowest channel freq and lowest signal freq

        self.right_gap_Hz : float
            Frequency gap between upper signal freq and upper channel freq

    """

    channel_center_freq_Hz: float
    channel_min_freq_Hz: float
    channel_max_freq_Hz: float
    signal_center_freq_Hz: float
    signal_bw_Hz: float

    channel_width_Hz: float = field(init=False)
    signal_min_freq_Hz: float = field(init=False)
    signal_max_freq_Hz: float = field(init=False)
    left_gap_Hz: float = field(init=False)
    right_gap_Hz: float = field(init=False)

    def __post_init__(self):
        """

        Illustration of Channel

                <---sig_BW--->

                    sig_c
                       .
                       .
                 ******.******
                 *     .     *
                 *     .     *
         |       *  |  .     * |
         |       *  |  .     * |
         |       *  |  .     * |
         |       *  |  .     * |
         |       *  |  .     * |
         |       *  |  .     * |
         |       *  |  .     * |
         |       *  |  .     * |
        _|_______*__|__._____*_|_______ ->f
    ch_min        ch_cen      ch_max

         <--lg-->           <rg>

        Parameters
        ----------
        channel_center_freq_Hz : float
            Central frequency of channel.
        channel_min_freq_Hz : float
            Lower frequency of channel.
        channel_max_freq_Hz : float
            Upper frequency of channel.
        signal_center_freq_Hz : float
            Central frequency of signal.
        signal_bw_Hz : float
            Signal Bandwidth.

        Returns
        -------
        None.

        """

        # Quick sanity check that center frequency is between min and max
        assert (
            self.channel_min_freq_Hz < self.channel_max_freq_Hz
        ), "Error: channel_min_freq_Hz must be smaller than"
        f" channel_max_freq_Hz, but {self.channel_min_freq_Hz =}"
        f" >= {self.channel_max_freq_Hz =}"
        assert (
            self.channel_center_freq_Hz < self.channel_max_freq_Hz
        ), "Error: channel_center_freq_Hz must be smaller than"
        " channel_max_freq_Hz, but {self.channel_center_freq_Hz = }"
        ">= {self.channel_max_freq_Hz = }"
        assert (
            self.channel_min_freq_Hz < self.channel_center_freq_Hz
        ), "Error: channel_min_freq_Hz must be smaller than"
        " center_frequency_Hz, but {self.channel_min_freq_Hz = }"
        " >= {self.channel_center_freq_Hz = }"

        self.channel_width_Hz = self.channel_max_freq_Hz - self.channel_min_freq_Hz

        self.signal_min_freq_Hz = (self.signal_center_freq_Hz -
                                 0.5 * self.signal_bw_Hz)
        self.signal_max_freq_Hz = (self.signal_center_freq_Hz +
                                 0.5 * self.signal_bw_Hz)

        # Quick sanity checks to ensure that signal is fully inside channel.
        # May seem pedantic, but making mistakes when allocating channels is
        # very easy!
        assert (
            self.signal_bw_Hz > 0
        ), f"Error: {self.signal_bw_Hz =} but should be greater than zero! "
        assert (
            self.channel_min_freq_Hz <= self.signal_min_freq_Hz
        ), "Error: channel_min_freq_Hz    must be smaller than signal_min_freq_Hz"
        ", but {self.channel_min_freq_Hz = }>{self.self.signal_min_freq_Hz = }."
        assert (
            self.channel_max_freq_Hz >= self.signal_max_freq_Hz
        ), "Error: channel_max_freq_Hz must be greater than signal_max_freq_Hz,"
        " but {self.channel_max_freq_Hz = } < {self.signal_max_freq_Hz = }."

        self.left_gap_Hz = self.signal_min_freq_Hz - self.channel_min_freq_Hz
        self.right_gap_Hz = self.channel_max_freq_Hz - self.signal_max_freq_Hz


@dataclass
class TimeFreq:
    #TODO: Redo docstring
    """
    Class for storing info about the time axis and frequency axis.

    Attributes:
        number_of_points (int): Number of time points
        time_step (float): Duration of each time step
        t (npt.NDArray[float]): Array containing all the time points
        t_min_s (float): First entry in time array
        t_max_s (float): Last entry in time array

        center_frequency_Hz (float): Central optical frequency
        f (npt.NDArray[float]): Frequency range (relative to center_frequency_Hz)
                     corresponding to t when FFT is taken
        f_min_Hz (float): Lowest (most negative) frequency component
        f_max_Hz (float): Highest (most positive) frequency component
        freq_step_Hz (float): Frequency resolution
    """

    #init
    number_of_points: int
    time_step_s: float
    center_frequency_Hz: float
    #post init
    #t_s: npt.NDArray[float] = field(init=False)
    t_min_s: float = field(init=False)
    t_max_s: float = field(init=False)

    #f_Hz: npt.NDArray[float] = field(init=False)
    f_min_Hz: float = field(init=False)
    f_max_Hz: float = field(init=False)

    #default
    describe_time_freq_flag: bool = True

    def __post_init__(self):
        """

        Constructor for the TimeFreq class.


        Parameters
        ----------
        number_of_points : int
            Number of time points.
        time_step_s : float
            Duration of each time step.
        center_frequency_Hz : float
            Carrier frequency of the spectrum.
        describe_time_freq_flag : bool, optional
            Flag to toggle if description of TimeFreq instance should
            be printed to file/terminal. The default is True

        Returns
        -------
        None.

        """


        self.t_min_s = self.t_s()[0]
        self.t_max_s = self.t_s()[-1]

        self.f_min_Hz = self.f_Hz()[0]
        self.f_max_Hz = self.f_Hz()[-1]

        self.freq_step_Hz = self.f_Hz()[1] - self.f_Hz()[0]

        assert np.min(self.center_frequency_Hz +
                      self.f_Hz()) >= 0, f"""ERROR! Lowest frequency of
        {np.min(self.center_frequency_Hz+self.f_Hz())/1e9:.3f}GHz is below 0.
        Consider increasing the center frequency!"""

        if self.describe_time_freq_flag:
            self.describe_config()


    def t_s(self):
        time_array = np.linspace(0,
                          self.number_of_points * self.time_step_s,
                          self.number_of_points)
        return time_array - np.mean(time_array)

    def f_Hz(self):
        return get_freq_range_from_time(self.t_s())


    def f_rel_Hz(self):
        """
        Generates array of frequencies centered at zero. Crucually,
        "more negative" frequencies in this array correspond to "greater"
        frequencies. For example, if f_rel_Hz ranges from -10GHz to +10GHz,
        and the carrier freq of interest is 200THz, the entry with a value of
        -10GHz will correspond to the spectral component at 200.01THz. This
        rather awkward behavior is a result of the sign convention.

        Returns
        -------
        f_rel_Hz : npt.NDArray[float].

        """
        return get_freq_range_from_time(self.t_s())


    def f_rel_flipped_Hz(self):
        """
        Generates array of frequencies centered at zero, where positive
        entries correspond to higher frequencies.

        Returns
        -------
        f_rel_flipped_Hz : npt.NDArray[float].

        """
        return -1.0*get_freq_range_from_time(self.t_s())

    def f_abs_Hz(self):
        """
        Generates array of frequencies centered at the carrier frequency.


        Returns
        -------
        f_rel_Hz : npt.NDArray[float].

        """
        return -self.f_rel_Hz()+self.center_frequency_Hz

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
        d = destination
        print("  ", file=d)
        print(" ### timeFreq Configuration Parameters ###", file=d)
        print(
            f"Number of points\t = {self.number_of_points:>10}", file=d)
        print(
            f"Start time, t_min_s\t = {self.t_min_s*1e12:>10.3f}ps", file=d)
        print(f"Stop time, t_max_s\t\t = {self.t_max_s*1e12:>10.3f}ps", file=d)
        print(
            f"Time resolution\t\t = {self.time_step_s*1e12:>10.3f}ps",
            file=d)
        print("  ", file=d)

        print(
            f"Center frequency\t = {self.center_frequency_Hz/1e12:>10.3f}THz",
            file=d,
        )
        print(
            f"Start frequency\t\t = {self.f_min_Hz/1e12:>10.3f}THz", file=d)
        print(
            f"Stop frequency\t\t = {self.f_max_Hz/1e12:>10.3f}THz", file=d)
        print(
            f"Frequency resolution = {self.freq_step_Hz/1e6:>10.3f}MHz",
            file=d,
        )
        print("   ", file=d)

    def save_TimeFreq(self):
        """
        Saves info needed to construct this TimeFreq instance to .csv
        file so they can be loaded later using the load_TimeFreq function.

        Parameters:
            self
        """
        timeFreq_df = pd.DataFrame(
            columns=["number_of_points", "time_step_s", "center_frequency_Hz"]
        )

        timeFreq_df.loc[len(timeFreq_df.index)] = [
            self.number_of_points,
            self.time_step_s,
            self.center_frequency_Hz,
        ]

        timeFreq_df.to_csv("timeFreq.csv", index=False)


def load_TimeFreq(path: str) -> TimeFreq:
    """
    Loads TimeFreq for previous run

    Takes a path to a previous run, opens the relevant .csv file and extracts
    stored info from which the timeFreq class for that run can be restored.

    Parameters:
        path (str): Path to previous run

    Returns:
        time_freq: TimeFreq used in previous run.

    """
    path_to_saved_timeFreq = os.path.join(path)
    df = pd.read_csv(path_to_saved_timeFreq + "\\input_info\\timeFreq.csv")
    number_of_points = df["number_of_points"]
    time_step_s = df["time_step_s"]
    center_frequency_Hz = df["center_frequency_Hz"]

    return TimeFreq(int(number_of_points[0]), time_step_s[0], center_frequency_Hz[0])


def get_power(field_in_time_or_freq_domain: npt.NDArray[complex]
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
    field_in_time_or_freq_domain : npt.NDArray[complex]
        Temporal or spectral field.

    Returns
    -------
    power : npt.NDArray[complex]
        Temporal power (W) or PSD (J/Hz) at any instance or frequency.

    """
    power = np.abs(field_in_time_or_freq_domain) ** 2
    return power


def get_photon_number(freq_Hz: npt.NDArray[float],
                      field_in_freq_domain: npt.NDArray[float])-> float:
    """


    Parameters
    ----------
    freq_Hz : npt.NDArray[float]
        Absolute frequency axis in Hz.
    field_in_freq_domain : npt.NDArray[float]
        Field in frequency domain in units of sqrt(J/Hz).

    Returns
    -------
    Number of photons in the spectrum as a float.
    """

    delta_freq_Hz = freq_Hz[0]-freq_Hz[1]

    field_sum = np.sum(get_power(field_in_freq_domain)/freq_Hz,axis=1)



    return field_sum/(PLANCKCONST_J_PER_HZ*delta_freq_Hz)




def get_energy(
    time_or_freq: npt.NDArray[float],
    field_in_time_or_freq_domain: npt.NDArray[complex],
) -> float:
    """
    Computes energy of signal or spectrum

    Gets the power or PSD of the signal from
    get_power(field_in_time_or_freq_domain)
    and integrates it w.r.t. either time or
    frequency to get the energy.

    Parameters
    ----------
    time_or_freq : npt.NDArray[float]
        Time range in seconds or freq. range in Hz.
    field_in_time_or_freq_domain : npt.NDArray[complex]
        Temporal field in [sqrt(W)] or spectral field [sqrt(J/Hz)].

    Returns
    -------
    energy: float
        Signal energy in J .

    """
    energy = np.trapz(
        get_power(field_in_time_or_freq_domain), time_or_freq)
    return energy


def general_gaussian_pulse(normalized_time: npt.NDArray[float], order: float
                           ) -> npt.NDArray[complex]:
    """
    Generates a general gaussian pulse with the specified power, duration,
    offset, frequency shift, order and chirp.

    Parameters
    ----------
    normalized_time : npt.NDArray[float]
        Time range offset and normalized to time at which the pulse has
        a value of exp(-0.5)=0.6065... .

    order : float
        Controls shape of pulse as exp(-x**(order)) will be approximately
        square for large values of 'order'.

    Returns
    -------
    gaussian_pulse: npt.NDArray[complex]
        Gaussian pulse in time domain in units of sqrt(W).

    """

    assert order > 0, f"Error: Order of gaussian is {order}. Must be > 0"

    pulse = np.exp(-0.5 * np.abs(normalized_time) ** (order))

    return pulse


def gaussian_pulse(normalized_time: npt.NDArray[float]) -> npt.NDArray[complex]:
    """
    Generates a regular gaussian pulse

    Parameters
    ----------
    normalized_time : npt.NDArray[float]
        Time range offset and normalized to time at which the pulse has
        a value of exp(-0.5)=0.6065... .

    Returns
    -------
    gaussian_pulse: npt.NDArray[complex]
        Gaussian pulse in time domain with peak normalized to 1.0.

    """
    order = 2
    pulse = general_gaussian_pulse(normalized_time, order)

    return pulse


def square_pulse(normalized_time: npt.NDArray[float]) -> npt.NDArray[complex]:
    """
    Generates a square pulse with the specified power, duration, offset,
    frequency shift and chirp.

    Parameters
    ----------
    normalized_time : npt.NDArray[float]
        Time range offset and normalized to half the duration of
        the square pulse.

    Returns
    -------
    square_pulse : npt.NDArray[complex]
        Square pulse in time domain with peak normalized to 1.0.

    """
    order = 200
    pulse = general_gaussian_pulse(normalized_time, order)
    return pulse


def raised_cosine_pulse(normalized_time: npt.NDArray[float],
                        roll_off_factor: float
                        ) -> npt.NDArray[complex]:
    """
    Creates a raised cosine pulse

    Generates a raised cosine pulse, which is useful as
    it will have no inter-symbol interference if multiple pulses
    are spaced "duration_s" apart.

    Parameters
    ----------
    normalized_time : npt.NDArray[float]
        Time range offset and normalized to time at which the first zero
        of the sinc function occurs.

    roll_off_factor: float
        Controls "how similar" the pulse will be to an ideal sinc pulse, and
        thus the "steepness" of its spectrum. Must be between 0 and 1, where
        0 yields a sinc pulse with an "infinite" duration and square spectrum.
        Choosing 1 yields a shorter pulse with a gradual spectrum roll-off.


    Returns
    -------
    rc_pulse: npt.NDArray[complex]
        RC pulse in time domain with peak normalized to 1.0.

    """

    t = normalized_time
    beta = roll_off_factor

    pulse = np.sinc(t) * np.cos(np.pi*beta*t)/(1-(2*beta*t)**2)

    return pulse


def sinc_pulse(normalized_time: npt.NDArray[float]) -> npt.NDArray[complex]:
    """
    Creates a sinc pulse (sin(pi*x)/(pi*x))

    Generates a sinc pulse, which is useful as
    its spectral shape will be square.

    Parameters
    ----------
    normalized_time : npt.NDArray[float]
        Time range offset and normalized to time at which the first zero
        of the sinc function occurs.

    Returns
    -------
    sinc_pulse: npt.NDArray[complex]
        Sinc pulse in time domain with peak normalized to 1.0.

    """

    beta = 0
    pulse = raised_cosine_pulse(normalized_time, beta)

    return pulse


def sech_pulse(normalized_time: npt.NDArray[float]) -> npt.NDArray[complex]:
    """
    Creates hyperbolic secant pulse

    Generates a hyperbolic secant pulse (1/cosh(t)), which is the pulse
    shape that corresponds to a fundamental soliton; a solution to the NLSE
    for anormalous dispersion where the pulse remains unchanged as it
    propagates down the fiber.

    Parameters
    ----------
    normalized_time : npt.NDArray[float]
        Time range offset and normalized to characteristic duration of
        sech pulse. sech(time_s/duration_s)=0.648...

    Returns
    -------
    sech_pulse : npt.NDArray[complex]
        Sech pulse in time domain with peak normalized to 1.0.

    """

    pulse = 1/np.cosh(normalized_time)

    return pulse


def sqrt_triangle_pulse(normalized_time: npt.NDArray[float]) -> npt.NDArray[complex]:
    """
    Creates sqrt(triangle(t)) pulse

    Creates sqrt(triangle(t)) pulse, whose absolute square is a triangle
    function. Rarely used in practical experiments, but useful for
    illustrating certain effects.


    Parameters
    ----------
    normalized_time : npt.NDArray[float]
        Time range offset and normalized to total duration of the
        sqrt(triangle(t)) pulse; "corner to corner".


    Returns
    -------
    sqrt_triag_pulse : npt.NDArray[complex]
        Sqrt(triangle(t)) pulse in time domain with peak normalized to 1.0.
    """

    zero_array = np.zeros_like(normalized_time)
    left_array = 1+2*normalized_time
    right_array = 1-2*normalized_time

    triangle = np.maximum(zero_array, np.minimum(left_array, right_array))

    pulse = np.sqrt(triangle)

    return pulse


def sqrt_parabola_pulse(
        normalized_time: npt.NDArray[float],) -> npt.NDArray[complex]:
    """
    Creates sqrt(parabola(t)) pulse

    Creates sqrt(parabola(t)) pulse, whose absolute square is a parabola
    function. Such pulses can arise in amplifiers (positive gain) with normal
    dispersion.


    Parameters
    ----------
    normalized_time : npt.NDArray[float]
        Time range offset and normalized to half of the total duration of the
        sqrt(paraboa(t)) pulse; "roots = +/- duration_s ".


    Returns
    -------
    sqrt_parabola_pulse_pulse : npt.NDArray[complex]
        Sqrt(parabola(t)) pulse in time domain with normalized peak at 1.0.

    """

    parabola = 1 - normalized_time**2

    parabola[parabola <= 0] = 0

    sqrt_parabola_pulse_pulse = np.sqrt(parabola)

    return sqrt_parabola_pulse_pulse


def random_pulse(
        normalized_time: npt.NDArray[float],) -> npt.NDArray[complex]:



    polynomial_array = np.zeros_like(normalized_time)*1.0
    cos_array = np.ones_like(polynomial_array)*1.0

    random_poly_roots = np.random.uniform(-2, 2,8)
    random_freqs = np.random.uniform(-0.5,0.5,8)
    random_phases = np.random.uniform(-pi,pi,8)


    random_chirp = np.random.uniform(-3,3,1)
    chrip_factor = np.exp(1j*random_chirp*normalized_time**2)



    for poly_root, freq, phase in zip(random_poly_roots,random_freqs, random_phases):
        polynomial_array += (normalized_time-poly_root)
        cos_array *= np.cos(2*pi*freq*normalized_time+phase)




    envelope_list=["gaussian","sech"]
    envelope_func_str = np.random.choice(envelope_list)

    envelope = eval(f"{envelope_func_str}_pulse(normalized_time)")


    random_phase = np.random.uniform(0,2*pi)


    pulse = polynomial_array*envelope*cos_array*chrip_factor
    pulse/=np.max(np.abs(pulse))



    return pulse


# def random_pulse(
#         normalized_time: npt.NDArray[float],) -> npt.NDArray[complex]:

#     random_poly_coeffs = np.random.uniform(-10, 10,8)

#     random_poly = np.polyval(random_poly_coeffs, normalized_time)

#     envelope_list=["gaussian","sech"]
#     envelope_func_str = np.random.choice(envelope_list)

#     print(envelope_func_str)
#     envelope = eval(f"{envelope_func_str}_pulse(normalized_time)")


#     modulation_freq = np.random.uniform(0,0.5)
#     random_phase = np.random.uniform(0,2*pi)

#     modulation = np.cos(2*np.pi*modulation_freq*normalized_time+random_phase)

#     print(random_poly_coeffs)
#     print(envelope_func_str)
#     print(modulation_freq)

#     pulse = random_poly*envelope*modulation
#     pulse/=np.max(np.abs(pulse))

#     return pulse

def noise_ASE(
        time_s: npt.NDArray[float],
        noise_stdev_sqrt_W: float
) -> npt.NDArray[complex]:
    """
    Generates white noise in the time domain with the
    specified Standard Deviation

    Generates an array of complex numbers with random phase from -pi to pi and
    field strengths distributed normally around 0 and a standard
    deviation of noise_stdev_sqrt_W in units of sqrt(W).

    Parameters
    ----------
    time_s : npt.NDArray[float]
        Time range in seconds.
    noise_stdev_sqrt_W : float
        Standard deviation of temporal field fluctuations in sqrt(W).

    Returns
    -------
    random_noise : npt.NDArray[complex]
        White noise.

    """

    random_fields = np.random.normal(loc=0.0,
                                     scale=noise_stdev_sqrt_W,
                                     size=len(time_s)) * (1+0j)

    random_phases = np.random.uniform(-pi, pi, len(time_s))
    random_noise = random_fields * np.exp(1j * random_phases)
    return random_noise


def get_pulse(
    time_s: npt.NDArray[float],
    duration_s: float,
    time_offset_s: float,
    amplitude_sqrt_W: float,
    pulse_type: str,
    freq_offset_Hz: float = 0.0,
    chirp: float = 0.0,
    order: float = 2.0,
    roll_off_factor: float = 0.0,
    noise_stdev_sqrt_W: float = 0.0,
    phase_rad: float = 0.0
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
    duration_s : float
        Characteristic duration of pulse. Meaning varies based on pulse type.
    time_offset_s : float
        Time at which the pulse peaks.
    pulse_type : str
        String that determines which pulse type should be generated.
    amplitude_sqrt_W : float
        Amplitude in units of sqrt(W)
    freq_offset_Hz : float, optional
        Center frequency relative to carrier frequency specified in timeFreq.
        The default is 0.0.
    chirp : float, optional
        Dimensionless parameter controlling the chirp.
        The default is 0.0.
    order : float, optional
        Order of the Super Gaussian, exp(-x**(2*order)).
        The default is 2.0.
    noise_stdev_sqrt_W : float, optional
        Standard deviation of temporal field fluctuations in sqrt(W).
        The default is 0.0.
    phase_rad : float, optional
        Additional phase factor multiplied onto signal.
        The default is 0.0.

    Returns
    -------
    output_pulse : npt.NDArray[complex]
        Specified pulse in time domain in units of sqrt(W).

    """

    delta_t_s = time_s-time_offset_s

    normalized_time = delta_t_s/duration_s
    phase_factor = np.exp(1j*phase_rad)
    carrier_freq = np.exp(-1j * 2 * pi * (freq_offset_Hz) * delta_t_s)
    chirp_factor = np.exp(
        -(1j * chirp) / 2 * normalized_time ** 2
    )

    noise = noise_ASE(time_s, noise_stdev_sqrt_W)
    output_pulse = 1j*np.zeros_like(time_s)

    if pulse_type.lower() in ["gaussian", "gauss"]:
        output_pulse = gaussian_pulse(normalized_time)

    if pulse_type.lower() in ["general_gaussian", "general_gauss"]:
        output_pulse = general_gaussian_pulse(normalized_time, order)

    elif pulse_type.lower() == "sech":
        output_pulse = sech_pulse(normalized_time)

    elif pulse_type.lower() == "square":
        output_pulse = square_pulse(normalized_time)

    elif pulse_type.lower() == "sqrt_triangle":
        output_pulse = sqrt_triangle_pulse(normalized_time)

    elif pulse_type.lower() == "sqrt_parabola":
        output_pulse = sqrt_parabola_pulse(normalized_time)

    elif pulse_type.lower() == "sinc":
        output_pulse = sinc_pulse(normalized_time)

    elif pulse_type.lower() == "raised_cosine":
        output_pulse = raised_cosine_pulse(normalized_time, roll_off_factor)

    elif pulse_type.lower() == "random":
        output_pulse = random_pulse(normalized_time)

    elif pulse_type.lower() == "cw":
        output_pulse = np.ones_like(normalized_time)

    elif pulse_type.lower() == "custom":
        output_pulse = output_pulse

    return (1+0j)*amplitude_sqrt_W*output_pulse*carrier_freq*chirp_factor*phase_factor+noise


def get_spectrum_from_pulse(
    time_s: npt.NDArray[float],
    pulse_field: npt.NDArray[complex],
    FFT_tol: float = 1e-7,
) -> npt.NDArray[complex]:
    """


    Parameters
    ----------
    time_s : npt.NDArray[float]
        Time range in seconds.
    pulse_field: npt.NDArray[complex]
        Complex field of pulse in time domain in units of sqrt(W).
    FFT_tol : float, optional
        When computing the FFT and going from temporal to spectral domain, the
        energy (which theoretically should be conserved) cannot change
        fractionally by more than FFT_tol. The default is 1e-7.

    Returns
    -------
    spectrum_field : npt.NDArray[complex]
        Complex spectral field in units of sqrt(J/Hz).

    """

    pulseEnergy = get_energy(time_s, pulse_field)  # Get pulse energy
    f = get_freq_range_from_time(time_s)
    dt = time_s[1] - time_s[0]

    assert dt > 0, (f"ERROR: dt must be positive, "
                    f"but {dt=}. {time_s[1]=},{time_s[0]=}")
    spectrum_field = fftshift(
        fft(pulse_field)) * dt  # Take FFT and do shift
    spectrumEnergy = get_energy(f, spectrum_field)  # Get spectrum energy

    err = np.abs((pulseEnergy / spectrumEnergy - 1))

    assert (
        err < FFT_tol
    ), (f"ERROR = {err:.3e} > {FFT_tol:.3e} = FFT_tol : Energy changed "
        "when going from Pulse to Spectrum!!!")

    return spectrum_field


def get_time_from_freq_range(frequency_Hz: npt.NDArray[float]
                             ) -> npt.NDArray[float]:
    """
    Calculate time range for pulse based on frequency range.

    Essentially the inverse of the get_freq_range_from_time function.
    If we have a frequency range and take the iFFT of a spectrum field
    to get the pulse field in the time domain, this function provides the
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
                            spectrum_field: npt.NDArray[complex],
                            FFT_tol: float = 1e-7) -> npt.NDArray[complex]:
    """
    Converts the spectral field of a signal in the freq. domain temporal
    field in time domain

    Uses the iFFT to shift from freq. to time domain and ensures that energy
    is conserved

    Parameters
    ----------
    frequency_Hz : npt.NDArray[float]
        Frequency in Hz.
    spectrum_field : npt.NDArray[complex]
        Spectral field in sqrt(J/Hz).
    FFT_tol : float, optional
        Maximum fractional change in signal
        energy when doing FFT. The default is 1e-7.

    Returns
    -------
    pulse : npt.NDArray[complex]
        Temporal field in sqrt(W).

    """

    spectrumEnergy = get_energy(frequency_Hz, spectrum_field)

    time = get_time_from_freq_range(frequency_Hz)
    dt = time[1] - time[0]

    pulse = ifft(ifftshift(spectrum_field)) / dt
    pulseEnergy = get_energy(time, pulse)

    err = np.abs((pulseEnergy / spectrumEnergy - 1))

    assert (
        err < FFT_tol
    ), (f"ERROR = {err:.3e} > {FFT_tol:.3e} = FFT_tol : Energy changed too "
        "much when going from Spectrum to Pulse!!!")

    return pulse


def gaussian_filter_power(freq, center_freq, width):
    return (1.0+0j)*np.exp(-0.5*((freq-center_freq)/width)**2)


def square_filter_power(freq, center_freq, width):
    return (1.0+0j)*np.exp(-0.5*((freq-center_freq)/width)**80)


@dataclass
class F:
    a: float
    c: float = field(init=False)
    b: float = 1.2


    def __post_init__(self):
        self.c = self.a+self.b


#TODO: Implement separate dataclass for input/output amp and attentuation of
#      fiber span.
# @dataclass
# class FiberInputOutput:
#     input_atten_dB: float = 0.0
#     input_filter_power_function_pre_amp=None
#     input_amp_dB: float = 0.0
#     input_noise_factor_dB: float = -1e3
#     input_filter_power_function_post_amp=None

#     output_filter_power_function_pre_amp=None
#     output_amp_dB: float = 0.0
#     output_noise_factor_dB: float = -1e3
#     output_filter_power_function_post_amp=None
#     output_atten_dB: float = 0.0

#     def __post_init__(self):
#         pass

def no_filter(freq):
    return (1+0j)*np.ones_like(freq)


def zero_func(freq):
    return (1+0j)*np.zeros_like(freq)

@dataclass
class FiberSpan:
    #TODO: Redo docstring

    #init
    length_m: float
    number_of_steps: int
    gamma_per_W_per_m: float
    beta_list: list[float]
    alpha_dB_per_m: float
    #post init
    dz_m: float = field(init=False)
    alpha_Np_per_m: float = field(init=False)
    raman_in_freq_domain_func: Callable[npt.NDArray[float],npt.NDArray[complex]] = field(init=False)
    total_gainloss_dB: float = field(init=False)
    #Defaults
    use_self_steepening: bool = False
    raman_model: str = "None"
    fR: float = 0.0
    tau1: float = 0.0
    tau2: float = 0.0
    input_atten_dB: float = 0.0
    input_amp_dB: float = 0.0
    input_noise_factor_dB: float = -1e3
    input_filter_power_function: Callable[ [npt.NDArray[float]],npt.NDArray[complex]] = no_filter
    output_filter_power_function: Callable[ [npt.NDArray[float]],npt.NDArray[complex]] = no_filter
    output_amp_dB: float = 0.0
    output_noise_factor_dB: float = -1e3
    output_atten_dB: float = 0.0
    describe_fiber_flag: bool = True



    def __post_init__(self):

        self.number_of_steps = int(self.number_of_steps)
        #self.z_m() = np.linspace(0, self.length_m, self.number_of_steps + 1)
        self.dz_m = self.z_m()[1] - self.z_m()[0]

        # Pad list of betas so we always have terms up to 8th order
        while len(self.beta_list) <= 6:
            self.beta_list.append(0.0)

        # Loss coeff is usually specified in dB/km,
        # but Nepers/km is more useful for calculations
        self.alpha_Np_per_m = self.alpha_dB_per_m * np.log(10) / 10.0


        # TODO: Make alpha frequency dependent.

        # TODO: Implement Raman model
        if str(self.raman_model).lower() == "none":
            self.raman_model = None

        self.fR = 0.0
        self.tau1 = 0.0
        self.tau2 = 0.0

        self.raman_in_freq_domain_func = zero_func

        if self.raman_model is None:
            self.raman_in_freq_domain_func = zero_func

        # Raman parameters taken from Govind P. Agrawal's book,
        # "Nonlinear Fiber Optics".

        elif str(self.raman_model).lower() == "agrawal":
            self.fR = (
                0.180  # Relative contribution of Raman effect to overall nonlinearity
            )
            # Average angular oscillation time of molecular bonds in silica lattice. Note: 1/(2*pi*12.2fs) = 13.05THz = Typical Raman frequency
            self.tau1 = 12.2 * 1e-15
            # Average exponential decay time of molecular bond oscilaltions. Note: 2*1/(2*pi*30.0fs) = 10.61 THz = Typical Raman gain spectrum FWHM
            self.tau2 = 30.0 * 1e-15

            # Frequency domain representation of Raman response taken from https://github.com/omelchert/GNLStools/blob/main/src/GNLStools.py
            self.raman_in_freq_domain_func = lambda freq: (
                self.tau1 ** 2 + self.tau2 ** 2
            ) / (
                self.tau1 ** 2 * (1 - 1j * freq * 2 * pi * self.tau2) ** 2
                + self.tau2 ** 2
            )  # Freq domain representation of Raman response



        # if self.input_filter_power_function is None:
        #     self.input_filter_power_function = no_filter


        # if self.output_filter_power_function is None:
        #     self.output_filter_power_function = no_filter


        self.total_gainloss_dB = (self.input_atten_dB+self.input_amp_dB
                              +self.alpha_dB_per_m * self.length_m
                              +self.output_amp_dB+self.output_atten_dB)

        if self.describe_fiber_flag:
            self.describe_fiber()

    def z_m(self):
        return np.linspace(0, self.length_m, self.number_of_steps + 1)

    def describe_fiber(self, destination=None):
        """
        Prints a description of the fiber to destination

        Parameters:
            self
            destination (class '_io.TextIOWrapper') (optional): File to which
            destination should be printed. If None, print to console
        """
        d=destination
        print(" ### Characteristic parameters of fiber: ###", file=d)
        print(' ', file=d)
        print("\t*** Fiber size info: ***", file=d)
        print(f"\t\tFiber Length [km] \t\t= {self.length_m/1e3} ", file=d)
        print(f"\t\tNumber of Steps \t\t= {self.number_of_steps} ", file=d)
        print(f"\t\tdz [m] \t\t\t\t\t= {self.dz_m} ", file=d)
        print(' ', file=d)


        print("\t*** Fiber dispersion info: ***", file=d)
        for i, beta_n in enumerate(self.beta_list):
            print(f"\t\tFiber beta{i+2} [s^{i+2}/m] \t= {beta_n}",file=d)
        print(' ', file=d)

        print("\t*** Fiber gain/loss info: ***", file=d)
        print(f"\t\tFiber input loss [dB] \t= {self.input_atten_dB}")
        print(f"\t\tFiber input amp  [dB] \t= {self.input_amp_dB}")
        print(f"\t\tInput noise factor [dB] = {self.input_noise_factor_dB}",file=d)
        print(' ', file=d)
        print(f"\t\tFiber alpha_dB_per_m \t= {self.alpha_dB_per_m} ",file=d)
        print(f"\t\tFiber alpha_Np_per_m \t= {self.alpha_Np_per_m} ",file=d)
        print(' ', file=d)

        print(f"\t\tFiber output amp [dB] \t= {self.output_amp_dB} ", file=d)
        print(f"\t\tOutput noise factor [dB]= {self.output_noise_factor_dB}",file=d)
        print(f"\t\tFiber output loss [dB] \t= {self.output_atten_dB} ", file=d)
        print(' ', file=d)
        print(f"\t\tFiber total gain/loss [dB] \t= {self.total_gainloss_dB} "
              ,file=d)
        print(' ', file=d)



        print("\t*** Fiber nonlinear info: ***", file=d)
        print(f"\t\tFiber gamma [1/W/m] \t= {self.gamma_per_W_per_m} ", file=d)
        print(f"\t\tFiber self steepening \t= {self.use_self_steepening} ", file=d)
        print(f"\t\tRaman Model \t\t\t= {self.raman_model}.")
        print(f"\t\t(fR,tau1 [fs],tau2[fs])\t= ({self.fR:.3},{self.tau1/1e-15:.3},{self.tau2/1e-15:.3})",file=d)

        print(" ", file=d)


# Class for holding info about span of concatenated fibers.
@dataclass
class FiberLink:
    """
    Class for storing info about multiple concatenated fibers.

    Attributes:
        fiber_list (list[FiberSpan]): List of FiberSpan objects
        number_of_fibers_in_span (int): Number of fibers concatenated together
    """
    #init
    fiber_list: list[FiberSpan]
    #post init
    number_of_fibers_in_span: int = field(init=False)
    #defaults

    def __post_init__(self):
        """
        Post init for the FiberLink

        Parameters:
            self
        """

        self.number_of_fibers_in_span = len(self.fiber_list)

    def get_total_loss_dB(self):

        loss_so_far_dB = 0.0

        for fiber in self.fiber_list:
            loss_so_far_dB += (fiber.total_loss_dB
                               + fiber.input_atten_dB
                               + fiber.output_atten_dB)

        return loss_so_far_dB

    def get_total_gain_dB(self):
        gain_so_far_dB = 0.0

        for fiber in self.fiber_list:
            gain_so_far_dB += fiber.input_amp_dB+fiber.output_amp_dB

        return gain_so_far_dB

    def get_total_gainloss_dB(self):
        return self.get_total_gain_dB()-self.get_total_loss_dB()

    def get_total_loss_lin(self):
        return dB_to_lin(self.get_total_loss_dB())

    def get_total_gain_lin(self):
        return dB_to_lin(self.get_total_gain_dB())

    def get_total_gainloss_lin(self):
        return dB_to_lin(self.get_total_gainloss_dB())

    def get_total_length(self):
        length_so_far = 0.0

        for fiber in self.fiber_list:
            length_so_far += fiber.length_m

        return length_so_far

    def get_total_dispersion(self):
        disp_so_far = np.zeros_like(self.fiber_list[0].beta_list)

        for fiber in self.fiber_list:
            disp_so_far += fiber.length_m*np.array(fiber.beta_list)

        return disp_so_far

    #TODO: Find a way to save filter functions
    def save_fiber_link(self):
        """
        Saves info about each fiber in span to .csv file so they can be
        loaded later by the load_fiber_link function

        Parameters:
            self
        """



        fiber_df = pd.DataFrame(
            columns=[
                "length_m",
                "number_of_steps",
                "gamma_per_W_per_m",
                "beta2_s2_per_m",
                "beta3_s3_per_m",
                "beta4_s4_per_m",
                "beta5_s5_per_m",
                "beta6_s6_per_m",
                "beta7_s7_per_m",
                "beta8_s8_per_m",
                "alpha_dB_per_m",
                "use_self_steepening",
                "raman_model",
                "input_atten_dB",
                "input_amp_dB",
                "input_noise_factor_dB",
                "output_amp_dB",
                "output_noise_factor_dB",
                "output_atten_dB"


            ]
        )

        for fiber in self.fiber_list:
            fiber_df.loc[len(fiber_df.index)] = [
                fiber.length_m,
                fiber.number_of_steps,
                fiber.gamma_per_W_per_m,
                fiber.beta_list[0],
                fiber.beta_list[1],
                fiber.beta_list[2],
                fiber.beta_list[3],
                fiber.beta_list[4],
                fiber.beta_list[5],
                fiber.beta_list[6],
                fiber.alpha_dB_per_m,
                fiber.use_self_steepening,
                str(fiber.raman_model),
                fiber.input_atten_dB,
                fiber.input_amp_dB,
                fiber.input_noise_factor_dB,
                fiber.output_amp_dB,
                fiber.output_noise_factor_dB,
                fiber.output_atten_dB
            ]
        fiber_df.to_csv("fiber_link.csv", index=False)


#TODO: Find a way to load filter functions
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




    df = pd.read_csv(path + "input_info\\fiber_link.csv")
    length_m = df["length_m"]
    number_of_steps = df["number_of_steps"]
    gamma_per_W_per_m = df["gamma_per_W_per_m"]
    beta2_s2_per_m = df["beta2_s2_per_m"]
    beta3_s3_per_m = df["beta3_s3_per_m"]
    beta4_s4_per_m = df["beta4_s4_per_m"]
    beta5_s5_per_m = df["beta5_s5_per_m"]
    beta6_s6_per_m = df["beta6_s6_per_m"]
    beta7_s7_per_m = df["beta7_s7_per_m"]
    beta8_s8_per_m = df["beta8_s8_per_m"]
    alpha_dB_per_m = df["alpha_dB_per_m"]
    use_self_steepening = df["use_self_steepening"]
    raman_model = df["raman_model"]
    input_atten_dB=df["input_atten_dB"],
    input_amp_dB=df["input_amp_dB"],
    input_noise_factor_dB=df["input_noise_factor_dB"],
    output_amp_dB=df["output_amp_dB"],
    output_noise_factor_dB=df["output_noise_factor_dB"],
    output_atten_dB=df["output_atten_dB"]

    fiber_list = []

    for i in range(len(length_m)):
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
            length_m[i],
            number_of_steps[i],
            gamma_per_W_per_m[i],
            beta_list_i,
            alpha_dB_per_m[i],
            use_self_steepening = use_self_steepening[i],
            raman_model=raman_model[i],
            input_atten_dB=input_atten_dB[0][i],
            input_amp_dB=input_amp_dB[0][i],
            input_noise_factor_dB=input_noise_factor_dB[0][i],
            output_amp_dB=output_amp_dB[0][i],
            output_noise_factor_dB=output_noise_factor_dB[0][i],
            output_atten_dB=output_atten_dB[i]
        )
        fiber_list.append(current_fiber)
    return FiberLink(fiber_list)

#TODO: Redo docstring
@dataclass
class InputSignal:


    #Init
    time_freq: TimeFreq
    duration_s: float
    amplitude_sqrt_W: float
    pulse_type: str
    #post init
    pulse_field: npt.NDArray[complex] = field(init=False)
    spectrum_field: npt.NDArray[complex] = field(init=False)
    #defaults
    time_offset_s: float = 0.0
    freq_offset_Hz: float = 0.0
    chirp: float = 0.0
    order: float = 2.0
    roll_off_factor: float = 0.0
    noise_stdev_sqrt_W: float = 0.0
    phase_rad: float = 0.0
    FFT_tol: float = 1e-7
    describe_input_signal_flag: bool = True

    def __post_init__(self):



        self.pulse_field = get_pulse(
            self.time_freq.t_s(),
            self.duration_s,
            self.time_offset_s,
            self.amplitude_sqrt_W,
            self.pulse_type,
            self.freq_offset_Hz,
            self.chirp,
            self.order,
            self.roll_off_factor,
            self.noise_stdev_sqrt_W,
            self.phase_rad
        )
        self.spectrum_field = 1j * np.zeros_like(self.pulse_field)


        if get_energy(self.time_freq.t_s(), self.pulse_field) == 0.0:
            self.spectrum_field = np.copy(self.pulse_field)
        else:
            self.update_spectrum()

        if self.describe_input_signal_flag:
            self.describe_input_signal()


    def get_peak_pulse_power(self) -> float:
        """
        Computes peak power of time domain signal.

        Returns
        -------
        float
            Power in time domain in units of W.

        """
        return np.max(get_power(self.pulse_field))


    def get_peak_pulse_field(self) -> float:
        """
        Computes peak field strength of time domain signal.

        Returns
        -------
        float
            Field strength in time domain in units of sqrt(W).

        """
        return np.sqrt(np.max(get_power(self.pulse_field)))

    def update_spectrum(self):
        """
        Updates the spectrum. Useful if the time domain signal is altered, for
        example when a custom signal is generated by adding multiple ones
        together.

        Returns
        -------
        None.

        """
        self.spectrum_field = get_spectrum_from_pulse(
            self.time_freq.t_s(),
            self.pulse_field,
            FFT_tol=self.FFT_tol
        )




    def describe_input_signal(self, destination=None):
        """
        Prints a description of the input signal to destination

        Parameters:
            self
            destination (class '_io.TextIOWrapper') (optional): File to which
                    destination should be printed. If None, print to console
        """

        print(" ### Input Signal Parameters ###", file=destination)
        print(f"  Pmax   = {self.get_peak_pulse_power():.3f} W", file=destination)
        print(
            f"  Duration  \t= {self.duration_s*1e12:.3f} ps",
            file=destination)
        print(
            f"  Time offset  \t= {self.time_offset_s*1e12:.3f} ps",
            file=destination
        )
        print(
            f"  Freq offset  \t= {self.freq_offset_Hz/1e9:.3f} GHz",
            file=destination
        )
        print(f"  Chirp  \t= {self.chirp:.3f}", file=destination)
        print(f"  pulseType  \t= {self.pulse_type}", file=destination)
        print(f"  order  \t= {self.order}", file=destination)
        print(
            f"  noiseAmplitude  = {self.noise_stdev_sqrt_W:.3f} sqrt(W)",
            file=destination,
        )

        print("   ", file=destination)

        scalingFactor, prefix = get_units(self.time_freq.t_s()[-1])
        fig, ax = plt.subplots(dpi=300)
        ax.set_title(f'Input signal for {self.pulse_type} in time domain')
        ax.plot(self.time_freq.t_s()/scalingFactor,
                get_power(self.pulse_field), '-')
        ax.set_xlabel(f'Time [{prefix}s]')
        ax.set_ylabel('Power [W]')
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(-1*self.duration_s/scalingFactor,
                    1*self.duration_s/scalingFactor)
        plt.show()




        scalingFactor, prefix = get_units(self.time_freq.f_Hz()[-1])
        fig, ax = plt.subplots(dpi=300)
        ax.set_title(f'Input signal for {self.pulse_type} in freq domain')
        ax.plot(-self.time_freq.f_Hz()/scalingFactor,
                get_power(self.spectrum_field), '.')
        ax.set_xlabel(f'Freq [{prefix}Hz]')
        ax.set_ylabel('Energy dens. [J/Hz]')
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_yscale('log')
        ax.set_xlim(-6/self.duration_s/scalingFactor,
                    6/self.duration_s/scalingFactor)
        max_power = np.max(get_power(self.spectrum_field))
        ax.set_ylim(1e-12*max_power, 2*max_power)

        plt.show()

    def save_input_signal(self):
        """
        Saves info needed to construct this InputSignal instance to .csv
        file so they can be loaded later using the load_input_signal function.

        Parameters:
            self
        """

        self.time_freq.save_TimeFreq()

        if self.pulse_type.lower() in ["custom", "random"] :
            custom_input_df = pd.DataFrame(
                columns=["time_s", "field_sqrt_W_real",
                         "field_sqrt_W_imag"]
            )

            custom_input_df["time_s"] = self.time_freq.t_s()
            custom_input_df["field_sqrt_W_real"] = np.real(self.pulse_field)
            custom_input_df["field_sqrt_W_imag"] = np.imag(self.pulse_field)

            custom_input_df.to_csv("Custom_or_random_input_signal.csv", index=False)

        else:
            # Initialize dataframe
            signal_df = pd.DataFrame(
                columns=[
                    "duration_s",
                    "time_offset_s",
                    "pulse_type",
                    "amplitude_sqrt_W",
                    "freq_offset_Hz",
                    "chirp",
                    "order",
                    "roll_off_factor",
                    "noise_stdev_sqrt_W",
                    "phase_rad",
                    "FFT_tol"
                ]
            )

            pulse_data = [
                self.duration_s,
                self.time_offset_s,
                self.pulse_type,
                self.amplitude_sqrt_W,
                self.freq_offset_Hz,
                self.chirp,
                self.order,
                self.roll_off_factor,
                self.noise_stdev_sqrt_W,
                self.phase_rad,
                self.FFT_tol
            ]

            # Fill it with values used for generating input signal
            signal_df.loc[-1] = pulse_data
            # Export dataframe to .csv file
            signal_df.to_csv("Input_signal.csv", index=False)


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
    path_to_saved_input_signal = os.path.join(path)
    try:
        try_path = path_to_saved_input_signal + "input_info\Input_signal.csv"
        df = pd.read_csv(try_path)

        duration_s = df["duration_s"][0]
        time_offset_s = df["time_offset_s"][0]
        pulse_type = df["pulse_type"][0]
        amplitude_sqrt_W = df["amplitude_sqrt_W"][0]
        freq_offset_Hz = df["freq_offset_Hz"][0]
        chirp = df["chirp"][0]
        order = df["order"][0]
        roll_off_factor = df["roll_off_factor"][0]
        noise_stdev_sqrt_W = df["noise_stdev_sqrt_W"][0]
        phase_rad = df["phase_rad"][0]
        FFT_tol = df["FFT_tol"][0]


        # Load timeFreq
        old_time_freq = load_TimeFreq(path)


        # Initialize class for loaded signal
        loaded_input_signal = InputSignal(
            old_time_freq,
            duration_s,
            amplitude_sqrt_W,
            pulse_type,
            time_offset_s,
            freq_offset_Hz,
            chirp,
            order,
            roll_off_factor,
            noise_stdev_sqrt_W,
            phase_rad,
            describe_input_signal_flag=False,
            FFT_tol = FFT_tol
        )



    except FileNotFoundError:
        try_path = path_to_saved_input_signal + "input_info\Custom_or_random_input_signal.csv"
        df = pd.read_csv(try_path)

        A_real = np.array(df["field_sqrt_W_real"])
        A_imag = np.array(df["field_sqrt_W_imag"])
        A = A_real + 1j * A_imag

        old_time_freq = load_TimeFreq(path)

        A_spectrum = get_spectrum_from_pulse(old_time_freq.t_s(), A)

        loaded_input_signal = InputSignal(
            old_time_freq,
            get_stdev(old_time_freq.t_s(), A),
            np.sqrt(np.max(get_power(A))),
            "custom",
            describe_input_signal_flag=False,
            FFT_tol = 1e-3
        )

        loaded_input_signal.pulse_field = A
        loaded_input_signal.spectrum_field = A_spectrum




    return loaded_input_signal


class SSFMResult:
    """
    Class for storing info about results computed by SSFM.

    Attributes:
        input_signal ( InputSignal ): Signal launched into fiber
        fiber ( FiberSpan ): Fiber signal was sent through
        experiment_name ( str ): Name of experiment
        dirs ( tuple ): Contains directory where current script is located and
                    the directory where output is to be saved

        pulse_matrix ( npt.NDArray[complex] ): Amplitude of pulse at every
                                z-location in fiber
        spectrum_matrix ( npt.NDArray[complex] ): Spectrum of pulse at
                                    every z-location in fiber
    """

    def __init__(
        self,
        input_signal: InputSignal,
        fiber: FiberSpan,
        experiment_name: str,
        directories: str,
    ):
        """
        Constructor for SSFMResult.

       Parameters:
            input_signal ( InputSignal ): Signal launched into fiber
            fiber ( FiberSpan ): Fiber signal was sent through
            experiment_name ( str ): Name of experiment
            directories ( tuple ): Contains directory where current script is
            located and the directory where output is to be saved
        """
        self.input_signal = input_signal
        self.fiber = fiber
        self.experiment_name = experiment_name
        self.dirs = directories

        self.pulse_matrix = np.zeros(
            (len(fiber.z_m()), input_signal.time_freq.number_of_points)
        ) * (1 + 0j)
        self.spectrum_field_matrix = np.copy(self.pulse_matrix)

        self.pulse_matrix[0, :] = np.copy(input_signal.pulse_field)
        self.spectrum_field_matrix[0, :] = np.copy(input_signal.spectrum_field)


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
    scalingfactor, prefix = get_units(fiber.length_m)
    length_list = np.array([])
    # Ensure that we don't measure distances in Mm or Gm
    if scalingfactor > 1e3:
        scalingfactor = 1e3
        prefix = "k"
    if destination is not None:
        fig, ax = plt.subplots(dpi=300)
        ax.set_title(
            (f" Fiber Index = {fiber_index} \nComparison of"
             "characteristic lengths")
        )
    print(" ### Characteristic parameters of simulation: ###",
          file=destination)
    print(
        f"  Length_fiber \t= {fiber.length_m/scalingfactor:.2e} {prefix}m",
        file=destination,
    )

    if fiber.alpha_Np_per_m == 0.0:
        L_eff = fiber.length_m

    else:

        L_eff = (
            np.exp(fiber.alpha_Np_per_m * fiber.length_m)-1
        ) / fiber.alpha_Np_per_m

    print(
        f"  L_eff       \t= {L_eff/scalingfactor:.2e} {prefix}m",
        file=destination
    )

    length_list = np.append(length_list, L_eff)
    if destination is not None:
        ax.barh("Fiber Length", fiber.length_m / scalingfactor, color="C0")

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
    if fiber.gamma_per_W_per_m != 0.0:

        Length_NL = 1 / fiber.gamma_per_W_per_m / input_signal.get_peak_pulse_power()
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
            np.sqrt(2 * fiber.gamma_per_W_per_m * input_signal.get_peak_pulse_power() /
                    np.abs(fiber.beta_list[0]))
            / 2
            / pi
        )
        gain_MI = 2 * fiber.gamma_per_W_per_m * input_signal.get_peak_pulse_power()
        print(f"   Freq. w. max MI gain = {f_MI/1e9:.2e}GHz", file=destination)
        print(
            f"   Max MI gain  = {gain_MI*scalingfactor:.2e} /{prefix}m ",
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
    elif fiber.beta_list[0] > 0 and fiber.gamma_per_W_per_m > 0:
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
            (f"   sign(beta2)   = {np.sign(fiber.beta_list[0])},"
             " so Optical Wave Breaking may occur "),
            file=destination,
        )
        print(
            " Nmin_OWB (cst.)  \t= 0.5*exp(3/4) (assuming Gaussian pulses)",
            file=destination,
        )
        print(
            f" N_ratio = N_soliton/Nmin_OWB \t= {N_ratio:.2e}",
            file=destination)
        print(
            (f" Length_wave_break \t= {Length_wave_break/scalingfactor:.2e}"
             f"{prefix}m"),
            file=destination,
        )

        if destination is not None:
            ax.barh("OWB Length", Length_wave_break /
                    scalingfactor, color="C6")
    if destination is not None:
        ax.barh("$\Delta$z", fiber.dz_m / scalingfactor, color="C7")
        length_list = np.append(length_list, fiber.dz_m)

        ax.set_xscale("log")
        ax.set_xlabel(f"Length [{prefix}m]")

        Lmin = np.min(length_list) / scalingfactor * 1e-1
        Lmax = fiber.length_m / scalingfactor * 1e2
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

    #TODO: Decide if this should be re-enabled
    # describe_sim_parameters(
    #     current_fiber,
    #     current_input_signal,
    #     fiber_index,
    #     destination=destination
    # )


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




def create_output_directory(experiment_name: str) -> [(str, str), datetime]:
    """
    Creates output directory for output (graphs etc.)

    Parameters
    ----------
    experiment_name : str
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

    if experiment_name == "most_recent_run":
        current_dir = base_dir + "most_recent_run\\"
        overwrite_folder_flag = True
    else:

        current_dir = (
            base_dir
            + "Simulation Results\\" +
            f"{experiment_name}\\"
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
                         time_freq: TimeFreq,
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

    return np.exp(1j * fiber.gamma_per_W_per_m * get_power(pulse) * dz_m)


def get_NL_factor_self_steepening(fiber: FiberSpan,
                                  time_freq: TimeFreq,
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
    output = np.exp(1j * fiber.gamma_per_W_per_m*(pulse_power+1j/2/np.pi/time_freq.center_frequency_Hz /
                    (pulse+np.sqrt(np.max(pulse_power))/1e6*(1+0j))*np.gradient(pulse_power*pulse, time_freq.t_s())) * dz_m)

    return output


# TODO: Fully implement Raman model
def get_NL_factor_full(fiber: FiberSpan,
                       time_freq: TimeFreq,
                       pulse: npt.NDArray[complex],
                       dz_m: float) -> npt.NDArray[complex]:
    # TODO: Implement Raman effect for both long and short-duration pulses
    fR = fiber.fR
    freq = time_freq.f_Hz()
    t = time_freq.t_s()


    f0 = time_freq.center_frequency_Hz
    RamanInFreqDomain = fiber.raman_in_freq_domain_func(freq)

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
        * fiber.gamma_per_W_per_m
        * (1.0 + freq / f0)
        * get_pulse_from_spectrum(freq, NR_func(get_spectrum_from_pulse(time_freq.t_s(), pulse)))
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
    f_Hz() : npt.NDArray[float]
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
    experiment_name: str = "most_recent_run",
    show_progress_flag: bool = False,
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
        experiment_name ="most_recent_run" (str) (optional): Name of folder for
                                                            present simulation.
        show_progress_flag = False (bool) (optional): Print percentage
                                                    progress to terminal?
        FFT_tol=1e-7 (float) (optional): Maximum fractional change in signal
                                         energy when doing FFT


    Returns:
        list: List of SSFMResult corresponding to each fiber segment.

    """
    print("########### Initializing SSFM!!! ###########")

    t = input_signal.time_freq.t_s()
    # dt = input_signal.time_freq.t_s()ime_step_s
    f = input_signal.time_freq.f_Hz()
    df = input_signal.time_freq.freq_step_Hz
    fc = input_signal.time_freq.center_frequency_Hz

    # Create output directory, switch to it and return
    # appropriate paths and current time
    dirs, current_time = create_output_directory(experiment_name)

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
    input_signal.save_input_signal()

    # Return to main output directory
    os.chdir(current_dir)

    current_input_signal = deepcopy(input_signal)

    ssfm_result_list = []

    print(f"Starting SSFM loop over {len(fiber_link.fiber_list)} fibers")

    for fiber_index, fiber in enumerate(fiber_link.fiber_list):

        print(
            (f"Propagating through fiber number {fiber_index+1} out of "
             f"{fiber_link.number_of_fibers_in_span}")
        )

        # Initialize arrays to store pulse and spectrum throughout fiber
        ssfm_result = SSFMResult(
            current_input_signal, fiber, experiment_name, dirs
        )

        newFolderName = "Length_info\\"
        newFolderPath = newFolderName
        os.makedirs(newFolderPath, exist_ok=True)
        os.chdir(newFolderPath)

        # TODO: Decide if this should be re-enabled
        #Print simulation info to both terminal and .txt file in output folder
        if show_progress_flag:
            describeInputConfig(current_time,
                                fiber,
                                current_input_signal,
                                fiber_index)

        # Return to main output directory
        os.chdir(current_dir)

        # Pre-calculate dispersion term
        dispterm = np.zeros_like(input_signal.time_freq.f_Hz()) * 1.0
        for idx, beta_n in enumerate(fiber.beta_list):
            n = idx + 2  # Note: zeroth entry in beta_list is beta2
            # Minus must be included for f due to -i*omega*t sign convention
            dispterm += (beta_n / np.math.factorial(n)
                         * (-2 * pi * f) ** (n)
                         )

        # Pre-calculate effect of dispersion and loss as it's
        # the same everywhere
        disp_and_loss = np.exp(
            fiber.dz_m * (1j * dispterm + fiber.alpha_Np_per_m / 2))
        disp_and_loss_half_step = disp_and_loss ** 0.5

        # Precalculate constants for nonlinearity

        # Use simple NL model by default if Raman is ignored

        # TODO: sort out logic for choosing NL function
        if fiber.use_self_steepening:
            NL_function = get_NL_factor_self_steepening
        else:
            NL_function = get_NL_factor_simple

        if fiber.raman_model is not None:
            NL_function = get_NL_factor_full

        inputAttenuationField_lin = np.sqrt(dB_to_lin(fiber.input_atten_dB))

        random_phases_input = np.random.uniform(-pi, pi, len(f))
        random_phase_factor_input = np.exp(1j * random_phases_input)
        input_amp_field_factor = 10 ** (fiber.input_amp_dB / 20)
        input_noise_ASE_array = random_phase_factor_input * np.sqrt(
            get_noise_PSD(
                fiber.input_noise_factor_dB,
                fiber.input_amp_dB,
                -f + fc,
                df
            )
        )

        outputAttenuationField_lin = 1.0  # temporarily=1 until we reach end

        # temporary values until we reach fiber end
        noise_ASE_array = 1.0 * np.zeros_like(f)
        output_amp_field_factor = 1.0
        output_filter_field_array = (1.0+0j)*np.ones_like(f)

        # Initialize arrays to store temporal profile
        initial_pulse = np.copy(current_input_signal.pulse_field)
        initial_spectrum = get_spectrum_from_pulse(
            current_input_signal.time_freq.t_s(),
            current_input_signal.pulse_field,
            FFT_tol=FFT_tol,
        )
        # Initialize spectrum and apply attenuation, input amplification, noise
        # as well as dispersion half-step
        spectrum = (
            get_spectrum_from_pulse(
                current_input_signal.time_freq.t_s(),
                current_input_signal.pulse_field,
                FFT_tol=FFT_tol,
            ) * inputAttenuationField_lin*input_amp_field_factor+input_noise_ASE_array
        ) * disp_and_loss_half_step

        # apply input filter function
        spectrum *= np.sqrt(fiber.input_filter_power_function(-f+fc))

        pulse = get_pulse_from_spectrum(
            input_signal.time_freq.f_Hz(), spectrum, FFT_tol=FFT_tol)

        #
        # Start loop
        #   Apply full NL step
        #   Apply full Disp step
        # End loop
        # Apply half dispersion step
        # Save outputs and proceed to next fiber

        print(f"Running SSFM with {fiber.number_of_steps} steps")
        updates = 0
        for z_step_index in range(fiber.number_of_steps):

            # Apply nonlinearity
            pulse *= NL_function(fiber,
                                 input_signal.time_freq, pulse, fiber.dz_m)

            # Go to spectral domain and apply disp and loss

            spectrum = get_spectrum_from_pulse(
                t, pulse, FFT_tol=FFT_tol) * (disp_and_loss)

            # If at the end of fiber span, apply output amp and noise
            if z_step_index == fiber.number_of_steps - 1:
                randomPhases = np.random.uniform(-pi, pi, len(f))
                randomPhaseFactor = np.exp(1j * randomPhases)
                outputAttenuationField_lin = np.sqrt(dB_to_lin(
                    fiber.output_atten_dB))
                output_filter_field_array = np.sqrt(
                    fiber.output_filter_power_function(-f+fc))
                output_amp_field_factor = 10 ** (fiber.output_amp_dB / 20)
                noise_ASE_array = randomPhaseFactor * np.sqrt(
                    get_noise_PSD(
                        fiber.output_noise_factor_dB,
                        fiber.output_amp_dB,
                        f + fc,
                        df
                    )
                )

            # Apply half dispersion step to spectrum and store results
            ssfm_result.spectrum_field_matrix[z_step_index + 1, :] = (
                spectrum * disp_and_loss_half_step * output_amp_field_factor
                + noise_ASE_array
            ) * outputAttenuationField_lin*output_filter_field_array

            ssfm_result.pulse_matrix[z_step_index + 1, :] = get_pulse_from_spectrum(
                f,
                ssfm_result.spectrum_field_matrix[z_step_index + 1, :],
                FFT_tol=FFT_tol
            )

            # Return to time domain
            pulse = get_pulse_from_spectrum(f,
                                            spectrum,
                                            FFT_tol=FFT_tol)

            finished = 100 * (z_step_index / fiber.number_of_steps)
            if divmod(finished, 10)[0] > updates and show_progress_flag:
                updates += 1
                print(
                    (f"SSFM progress through fiber number {fiber_index+1} = "
                     f"{np.floor(finished):.2f}%")
                )
        # Append list of output results

        ssfm_result_list.append(ssfm_result)

        # Take signal at output of this fiber and feed it into the next one
        current_input_signal.pulse_field = np.copy(
            ssfm_result.pulse_matrix[z_step_index + 1, :]
        )
        current_input_signal.spectrum_field = np.copy(
            ssfm_result.spectrum_field_matrix[z_step_index + 1, :]
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
        return ssfm_result_list[0].fiber.z_m()
    zvals = np.array([])

    previous_length = 0
    for i, ssfm_result in enumerate(ssfm_result_list):

        if i == 0:
            zvals = np.copy(ssfm_result.fiber.z_m()[0:-1])
        elif (i > 0) and (i < number_of_fibers - 1):
            zvals = np.append(
                zvals, ssfm_result.fiber.z_m()[0:-1] + previous_length)
        elif i == number_of_fibers - 1:
            zvals = np.append(
                zvals, ssfm_result.fiber.z_m() + previous_length)
        previous_length += ssfm_result.fiber.length_m
    return zvals


def unpack_matrix(ssfm_result_list: list[SSFMResult],
                  zvals: npt.NDArray[float],
                  pulse_or_spectrum: str) -> npt.NDArray[complex]:
    """
    Unpacks pulse_matrix or spectrum_matrix for individual
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
        pulse_or_spectrum (str) : Indicates if we want to unpack pulse_matrix or spectrum_matrix

    Returns:
        npt.NDArray: Array of size (n_z_steps,n_time_steps) describing pulse field or spectrum field for whole fiber span.

    """
    time_freq = ssfm_result_list[0].input_signal.time_freq
    number_of_fibers = len(ssfm_result_list)

    # print(f"number_of_fibers = {number_of_fibers}")

    matrix = np.zeros((len(zvals), len(time_freq.t_s()))) * (1 + 0j)

    starting_row = 0

    for i, ssfm_result in enumerate(ssfm_result_list):

        if pulse_or_spectrum.lower() == "pulse":
            sourceMatrix = ssfm_result.pulse_matrix
        elif pulse_or_spectrum.lower() == "spectrum":
            sourceMatrix = ssfm_result.spectrum_field_matrix
        else:
            print(
                ("ERROR: Please set pulse_or_spectrum to either "
                 " 'pulse' or 'spectrum'!!!")
            )
            return
        if number_of_fibers == 1:
            return sourceMatrix
        if i == 0:
            matrix[0: len(ssfm_result.fiber.z_m()) - 1, :] = sourceMatrix[
                0: len(ssfm_result.fiber.z_m()) - 1, :
            ]
        elif (i > 0) and (i < number_of_fibers - 1):

            matrix[
                starting_row: starting_row + len(ssfm_result.fiber.z_m()) - 1, :
            ] = sourceMatrix[0: len(ssfm_result.fiber.z_m()) - 1, :]
        elif i == number_of_fibers - 1:

            matrix[
                starting_row: starting_row + len(ssfm_result.fiber.z_m()), :
            ] = sourceMatrix[0: len(ssfm_result.fiber.z_m()), :]
        starting_row += len(ssfm_result.fiber.z_m()) - 1
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

    time_freq = ssfm_result_list[0].input_signal.time_freq

    Nmin = np.max([int(time_freq.number_of_points / 2 - nrange), 0])
    Nmax = np.min(
        [int(time_freq.number_of_points / 2 + nrange),
         time_freq.number_of_points - 1]
    )

    zvals = unpack_Zvals(ssfm_result_list)

    t = time_freq.t_s()[Nmin:Nmax] * 1e12

    P_initial = get_power(ssfm_result_list[0].pulse_matrix[0, Nmin:Nmax])
    P_final = get_power(ssfm_result_list[-1].pulse_matrix[-1, Nmin:Nmax])

    scalingFactor, prefix = get_units(np.max(zvals))

    os.chdir(ssfm_result_list[0].dirs[1])
    fig, ax = plt.subplots(dpi=300)
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
    Plots pulse field power calculated by SSFM as colour surface

    2D colour plot of pulse field signal power in time domain
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

    time_freq = ssfm_result_list[0].input_signal.time_freq

    Nmin = np.max([int(time_freq.number_of_points / 2 - nrange), 0])
    Nmax = np.min(
        [int(time_freq.number_of_points / 2 + nrange),
         time_freq.number_of_points - 1]
    )

    zvals = unpack_Zvals(ssfm_result_list)
    matrix = unpack_matrix(ssfm_result_list, zvals, "pulse")

    # Plot pulse evolution throughout fiber in normalized log scale
    os.chdir(ssfm_result_list[0].dirs[1])
    fig, ax = plt.subplots(dpi=300)
    ax.set_title("Pulse Evolution (dB scale)")
    t_ps = time_freq.t_s()[Nmin:Nmax] * 1e12
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
     Plots pulse field power calculated by SSFM as 3D colour surface

     3D colour plot of pulse field signal power in time domain
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

    time_freq = ssfm_result_list[0].input_signal.time_freq

    Nmin = np.max([int(time_freq.number_of_points / 2 - nrange), 0])
    Nmax = np.min(
        [int(time_freq.number_of_points / 2 + nrange),
         time_freq.number_of_points - 1]
    )

    zvals = unpack_Zvals(ssfm_result_list)
    matrix = unpack_matrix(ssfm_result_list, zvals, "pulse")

    # Plot pulse evolution in 3D
    os.chdir(ssfm_result_list[0].dirs[1])
    fig, ax = plt.subplots(1, 1, figsize=(
        10, 7), subplot_kw={"projection": "3d"}, dpi=300)
    fig.patch.set_facecolor('white')
    plt.title("Pulse Evolution (dB scale)")

    t = time_freq.t_s()[Nmin:Nmax] * 1e12
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
        If chirpPlotRange=(f_min_Hz,f_max_Hz) is contained in **kwargs, use these
        values to set color scale.

    Returns
    -------
    None.

    """

    time_freq = ssfm_result_list[0].input_signal.time_freq

    Nmin = np.max([int(time_freq.number_of_points / 2 - nrange), 0])
    Nmax = np.min(
        [int(time_freq.number_of_points / 2 + nrange),
         time_freq.number_of_points - 1]
    )

    zvals = unpack_Zvals(ssfm_result_list)
    matrix = unpack_matrix(ssfm_result_list, zvals, "pulse")

    # Plot pulse evolution throughout fiber  in normalized log scale
    os.chdir(ssfm_result_list[0].dirs[1])
    fig, ax = plt.subplots(dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_title("Pulse Chirp Evolution")
    t = time_freq.t_s()[Nmin:Nmax] * 1e12
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
        Cmatrix[Cmatrix < -50] = -50  # Default f_min_Hz = -50GHz
        Cmatrix[Cmatrix > 50] = 50  # Default f_max_Hz = -50GHz
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
        print(kw, value)
        if kw.lower() == "show_chirp_plot_flag" and value is True:
            plot_pulse_chirp_2D(ssfm_result_list, nrange, dB_cutoff, **kwargs)
        if kw.lower() == "show_3d_plot_flag" and value is True:
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

    time_freq = ssfm_result_list[0].input_signal.time_freq
    center_freq_Hz = time_freq.center_frequency_Hz
    Nmin = np.max([int(time_freq.number_of_points / 2 - nrange), 0])
    Nmax = np.min(
        [int(time_freq.number_of_points / 2 + nrange),
         time_freq.number_of_points - 1]
    )

    zvals = unpack_Zvals(ssfm_result_list)

    P_initial = get_power(
        ssfm_result_list[0].spectrum_field_matrix[0, Nmin:Nmax])
    P_final = get_power(
        ssfm_result_list[-1].spectrum_field_matrix[-1, Nmin:Nmax])

    Pmax_initial = np.max(P_initial)
    Pmax_final = np.max(P_final)
    Pmax = np.max([Pmax_initial, Pmax_final])

    # Minus must be included here due to -i*omega*t sign convention
    f = time_freq.f_abs_Hz()[Nmin:Nmax] / 1e12

    scalingFactor, prefix = get_units(np.max(zvals))
    os.chdir(ssfm_result_list[0].dirs[1])
    fig, ax = plt.subplots(dpi=300)
    fig.patch.set_facecolor('white')
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

    time_freq = ssfm_result_list[0].input_signal.time_freq
    zvals = unpack_Zvals(ssfm_result_list)
    matrix = unpack_matrix(ssfm_result_list, zvals, "spectrum")

    Nmin = np.max([int(time_freq.number_of_points / 2 - nrange), 0])
    Nmax = np.min(
        [int(time_freq.number_of_points / 2 + nrange),
         time_freq.number_of_points - 1]
    )
    center_freq_Hz = time_freq.center_frequency_Hz

    # Plot pulse evolution throughout fiber in normalized log scale
    os.chdir(ssfm_result_list[0].dirs[1])
    fig, ax = plt.subplots(dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_title("Spectrum Evolution (dB scale)")
    # Minus must be included here due to -i*omega*t sign convention
    f = time_freq.f_abs_Hz()[Nmin:Nmax] / 1e12
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

    time_freq = ssfm_result_list[0].input_signal.time_freq
    zvals = unpack_Zvals(ssfm_result_list)
    matrix = unpack_matrix(ssfm_result_list, zvals, "spectrum")

    Nmin = np.max([int(time_freq.number_of_points / 2 - nrange), 0])
    Nmax = np.min(
        [int(time_freq.number_of_points / 2 + nrange),
         time_freq.number_of_points - 1]
    )
    center_freq_Hz = time_freq.center_frequency_Hz

    # Plot pulse evolution in 3D
    os.chdir(ssfm_result_list[0].dirs[1])
    fig, ax = plt.subplots(1, 1, figsize=(
        10, 7), subplot_kw={"projection": "3d"}, dpi=300)
    fig.patch.set_facecolor('white')
    plt.title("Spectrum Evolution (dB scale)")

    # Minus must be included here due to -i*omega*t sign convention
    f = time_freq.f_abs_Hz()[Nmin:Nmax] / 1e12
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
    Generates all plots of pulse field throughout the FiberLink

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
        if kw.lower() == "show_3d_plot_flag" and value is True:
            plot_spectrum_matrix_3D(ssfm_result_list, nrange, dB_cutoff)
    print("  ")


def make_chirp_gif(ssfm_result_list: list[SSFMResult],
                   nrange: int,
                   chirp_range_GHz: list[float] = [-20, 20],
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
    chirp_range_GHz : list[float], optional
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

    time_freq = ssfm_result_list[0].input_signal.time_freq
    zvals = unpack_Zvals(ssfm_result_list)
    matrix = unpack_matrix(ssfm_result_list, zvals, "pulse")
    scalingFactor, letter = get_units(np.max(zvals))

    Nmin = np.max([int(time_freq.number_of_points / 2 - nrange), 0])
    Nmax = np.min(
        [int(time_freq.number_of_points / 2 + nrange),
         time_freq.number_of_points - 1]
    )

    t_min_s = time_freq.t_s()[Nmin]
    t_max_s = time_freq.t_s()[Nmax]

    points = np.array(
        [time_freq.t_s() * 1e12, get_power(matrix[len(zvals) - 1, Nmin:Nmax])],
        dtype=object
    ).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[0:-1], points[1:]], axis=1)

    # Make custom colormap
    colors = ["red", "gray", "blue"]
    cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)

    # Initialize color normalization function
    norm = plt.Normalize(chirp_range_GHz[0], chirp_range_GHz[1])

    # Initialize line collection to be plotted
    lc = LineCollection(segments, cmap=cmap1, norm=norm)
    lc.set_array(
        get_chirp(time_freq.t_s()[Nmin:Nmax],
                  matrix[len(zvals) - 1, Nmin:Nmax]) / 1e9
    )

    # Initialize figure
    fig, ax = plt.subplots(dpi=300)
    fig.patch.set_facecolor('white')
    line = ax.add_collection(lc)
    fig.colorbar(line, ax=ax, label="Chirp [GHz]")

    Pmax = np.max(np.abs(matrix)) ** 2

    # Function for specifying axes

    def init():

        ax.set_xlim([t_min_s * 1e12, t_max_s * 1e12])
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
            [time_freq.t_s()[Nmin:Nmax] * 1e12, get_power(matrix[i, Nmin:Nmax])],
            dtype=object
        ).T.reshape(-1, 1, 2)

        # Make collection of lines from points
        segments = np.concatenate([points[0:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap1, norm=norm)

        # Activate norm function based on local chirp

        lc.set_array(
            get_chirp(time_freq.t_s()[Nmin:Nmax], matrix[i, Nmin:Nmax]) / 1e9)
        # Plot line
        line = ax.add_collection(lc)

    # Make animation
    ani = FuncAnimation(fig, update, range(len(zvals)), init_func=init)
    plt.show()

    # Save animation as .gif

    writer = PillowWriter(fps=int(framerate))
    ani.save(
        f"{ssfm_result_list[0].experiment_name}_fps={int(framerate)}.gif",
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
        Temporal or spectral field.

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
        Temporal or spectral field.

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
        Temporal or spectral field.

    Returns
    -------
    stdev : float
        Stdev in time or frequency domains.

    """

    stdev = np.sqrt(get_variance(time_or_freq, pulse_or_spectrum))
    return stdev



def plot_photon_number(ssfm_result_list: list[SSFMResult]):
    """
    Plots how photon number changes with distance

    Uses get_photon_number on spectrum at each z to see how photon number
    chages. Should be constant when alpha_dB/m = 0.0 even in the presence of
    dispersion an nonlinearity.

    Parameters
    ----------
    ssfm_result_list : list[SSFMResult]
        List of ssmf_result_class objects corresponding to each fiber segment.

    Returns
    -------
    None.

    """
    time_freq = ssfm_result_list[0].input_signal.time_freq
    center_freq_Hz = time_freq.center_frequency_Hz
    zvals = unpack_Zvals(ssfm_result_list)

    spectrum_matrix = unpack_matrix(
        ssfm_result_list, zvals, "spectrum")

    photon_number_array = np.zeros(len(zvals)) * 1.0

    f = -time_freq.f_Hz()+center_freq_Hz  # Minus must be included here due to -i*omega*t sign convention


    photon_number_array = get_photon_number(f, spectrum_matrix)


    scalingFactor_Z, prefix_Z = get_units(np.max(zvals))


    os.chdir(ssfm_result_list[0].dirs[1])
    fig, ax = plt.subplots(dpi=300)
    fig.patch.set_facecolor('white')
    plt.title("Evolution of photon number")
    N0 = photon_number_array[0]
    change_in_photon_number_percent = (photon_number_array-N0)/N0*100
    ax.plot(zvals / scalingFactor_Z, change_in_photon_number_percent)


    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(f"Distance [{prefix_Z}m]")
    ax.set_ylabel(f"Change in Photon number [%]", color="C0")



    save_plot("photon_number_evo")
    plt.show()
    os.chdir(ssfm_result_list[0].dirs[0])


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

    time_freq = ssfm_result_list[0].input_signal.time_freq
    center_freq_Hz = time_freq.center_frequency_Hz
    zvals = unpack_Zvals(ssfm_result_list)

    pulse_matrix = unpack_matrix(ssfm_result_list, zvals, "pulse")
    spectrum_matrix = unpack_matrix(
        ssfm_result_list, zvals, "spectrum")

    meanTimeArray = np.zeros(len(zvals)) * 1.0
    meanFreqArray = np.copy(meanTimeArray)
    stdTimeArray = np.copy(meanTimeArray)
    stdFreqArray = np.copy(meanTimeArray)
    f = time_freq.f_abs_Hz()  # Minus must be included here due to -i*omega*t sign convention

    i = 0
    for pulse, spectrum in zip(pulse_matrix, spectrum_matrix):

        meanTimeArray[i] = get_average(time_freq.t_s(), pulse)
        meanFreqArray[i] = get_average(f, spectrum)

        stdTimeArray[i] = get_stdev(time_freq.t_s(), pulse)
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
    fig, ax = plt.subplots(dpi=300)
    fig.patch.set_facecolor('white')
    plt.title("Evolution of temporal/spectral widths and centers")
    ax.plot(zvals / scalingFactor_Z, meanTimeArray /
            scalingFactor_pulse, label="Pulse Center")

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
        time_freq.t_min_s / scalingFactor_pulse,
        time_freq.t_max_s / scalingFactor_pulse
    )

    ax2 = ax.twinx()
    ax2.plot(
        zvals / scalingFactor_Z,
        meanFreqArray / scalingFactor_spectrum,
        "C1-",
        label=f"Spectrum Center",
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
        (time_freq.f_min_Hz+center_freq_Hz) / scalingFactor_spectrum,
        (time_freq.f_max_Hz+center_freq_Hz) / scalingFactor_spectrum
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

# TODO: Make figure dpi a variable argument
def plot_everything_about_result(
    ssfm_result_list: list[SSFMResult],
    nrange_pulse: int,
    dB_cutoff_pulse: float,
    nrange_spectrum: int,
    dB_cutoff_spectrum: float,
    **kwargs,
):
    """
    Generates all plots of pulse & spectrum fields etc. throughout FiberLink

    Calls plot_avg_and_std_of_time_and_freq, plot_everything_about_pulses and
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
    #plot_avg_and_std_of_time_and_freq(ssfm_result_list)
    plot_photon_number(ssfm_result_list)

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
    time_freq: TimeFreq, pulse, nrange_pulse, nrange_spectrum, dB_cutoff
):

    Nmin_pulse = np.max([int(time_freq.number_of_points / 2 - nrange_pulse), 0])
    Nmax_pulse = np.min(
        [
            int(time_freq.number_of_points / 2 + nrange_pulse),
            time_freq.number_of_points - 1,
        ]
    )

    t_max_s = time_freq.t_s()[Nmax_pulse]

    t = time_freq.t_s()[Nmin_pulse:Nmax_pulse]

    wavelet_durations = np.linspace((t[1] - t[0]) * 10, t_max_s, 1000)

    print((t[1] - t[0]) * 100, t_max_s)
    print(1 / t_max_s / 1e9, 1 / ((t[1] - t[0]) * 100) / 1e9)

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

    fig, ax = plt.subplots(dpi=300)
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



def compare_field_powers(field_1: npt.NDArray[complex],
                   field_2: npt.NDArray[complex])-> npt.NDArray[float]:
    """
    Computes difference between two fields normalized to total local power.

    Parameters
    ----------
    field_1 : npt.NDArray[complex]
        First field in time or frequency domain.
    field_2 : npt.NDArray[complex]
        Second field in time or frequency domain.

    Returns
    -------
    power_ratio : np.array[float]
        power_ratio=power(field_1-field_2)/(power(field_1)+power(field_2)).

        A local value of -1 means that field_2>>field_1
        A local value of 0 means that field_2==field_1
        A local value of 1 means that field_2<<field_1
    """



    assert len(field_1)==len(field_2), f"ERROR: {len(field_1) =} but "
    f"{len(field_2)}"

    power_1 = get_power(field_1)
    power_2 = get_power(field_2)
    total_power = power_1+power_2

    field_diff = (field_1-field_2)
    field_diff_power = get_power(field_diff)

    power_ratio = field_diff_power/total_power

    return power_ratio


def compare_field_energies(field_1: npt.NDArray[complex],
                   field_2: npt.NDArray[complex])-> float:
    """


    Parameters
    ----------
    field_1 : npt.NDArray[complex]
        First field in time or frequency domain.
    field_2 : npt.NDArray[complex]
        Second field in time or frequency domain.

    Returns
    -------
    float
        E(field_1-field_2)/(E(field_1)+E(field_2)).

        A value of 0 means that field_1==field_2
        A value of 1 means that field_1 and field_2 are very distinct
    """

    assert len(field_1)==len(field_2), f"ERROR: {len(field_1) =} but "
    f"{len(field_2)}"

    energy_1 = np.sum(get_power(field_1))
    energy_2 = np.sum(get_power(field_2))

    field_diff = (field_1-field_2)
    field_diff_energy = np.sum(get_power(field_diff))

    energy_ratio = (field_diff_energy)/(energy_1+energy_2)

    return energy_ratio









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


def extract_spectrum_range(freq_list_Hz: npt.NDArray[float],
                           spectral_field: npt.NDArray[complex],
                           freq_min_Hz: float,
                           freq_max_Hz: float) -> npt.NDArray[complex]:
    """
    Takes in a spectrum and extracts only values that fall in a certain range
    by setting all other ones to zero.

    Parameters
    ----------
    freqList : np.array([float])
        Absolute frequency.
    spectral_field : np.array([complex])
        Spectral field.
    freq1 : float
        Minimum frequency to be extracted.
    freq2 : float
        Maximum frequency to be extracted.

    Returns
    -------
    outputArray : np.array([complex])
        Same as spectral_field but with all entries corresponding to
        frequencies outside the [freq1,freq2] range set to zero.

    """

    assert (
        freq_min_Hz < freq_max_Hz
    ), (f"Error: freq_min_Hz must be smaller than freq_max_Hz,"
        f" but {freq_min_Hz =}>= {freq_max_Hz =}  ")

    array1 = np.abs(freq_list_Hz - freq_min_Hz)
    index1 = array1.argmin()

    array2 = np.abs(freq_list_Hz - freq_max_Hz)
    index2 = array2.argmin()

    outputArray = np.zeros_like(spectral_field) * 1j

    outputArray[index1:index2] = spectral_field[index1:index2]

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
                       channel: Channel,
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
        Spectrum field of signal whose SNR we want.
    channel : Channel
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
        freqs, spectrum, channel.signal_min_freq_Hz, channel.signal_max_freq_Hz
    )
    signalEnergy = get_energy(freqs, signal)

    noiseBelow = extract_spectrum_range(
        freqs, spectrum, channel.channel_min_freq_Hz, channel.signal_min_freq_Hz
    )
    noiseAbove = extract_spectrum_range(
        freqs, spectrum, channel.signal_max_freq_Hz, channel.channel_max_freq_Hz
    )

    noiseEnergyBelow = get_energy(freqs, noiseBelow)
    noiseEnergyAbove = get_energy(freqs, noiseAbove)

    leftEdgeNoisePSD = (
        np.abs(
            get_value_at_freq(
                freqs,
                channel.channel_min_freq_Hz +
                channel.left_gap_Hz * (1 - freqTol),
                noiseBelow,
            )
        )
        ** 2
    )
    rightEdgeNoisePSD = (
        np.abs(
            get_value_at_freq(
                freqs,
                channel.channel_max_freq_Hz -
                channel.right_gap_Hz * (1 - freqTol),
                noiseAbove,
            )
        )
        ** 2
    )

    signalBW = channel.signal_bw_Hz

    slope = (rightEdgeNoisePSD - leftEdgeNoisePSD) / signalBW
    offset = leftEdgeNoisePSD

    noiseEnergyInside = 0.5 * slope * signalBW ** 2 + offset * signalBW

    totalNoiseEnergy = noiseEnergyBelow + noiseEnergyInside + noiseEnergyAbove

    SNR_i_dB = 10 * np.log10(signalEnergy / totalNoiseEnergy)

    return SNR_i_dB


def get_channel_SNR_dB(ssfm_result_list: list[SSFMResult],
                       channel: Channel,
                       freqTol: float = 0.05
                       ) -> [npt.NDArray[float], npt.NDArray[float]]:
    """
    Calculates SNR throughout fiber span for a given channel

    Parameters
    ----------
    ssfm_result_list : list[SSFMResult]
        List of SSFMResult objects containing signal info for each fiber.
    channel_list : list[Channel]
        List of Channel objects describing the min, center and max
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
    time_freq = ssfm_result_list[0].input_signal.time_freq
    spectrum_matrix = unpack_matrix(
        ssfm_result_list,
        zvals,
        "spectrum")
    freqs = time_freq.f_Hz() + time_freq.center_frequency_Hz

    outputArray = np.zeros_like(zvals) * 1.0

    for i, spectrum in enumerate(spectrum_matrix):
        outputArray[i] = get_current_SNR_dB(
            freqs, spectrum, channel, freqTol=freqTol)
    return zvals, outputArray


def get_final_SNR_dB(ssfm_result_list: list[SSFMResult],
                     channel_list: list[Channel],
                     freqTol: float = 0.05) -> npt.NDArray[float]:
    """
    Calculates SNR for all channels at output of fiber span

    Parameters
    ----------
    ssfm_result_list : list[SSFMResult]
        List of SSFMResult objects containing signal info for each fiber.
    channel_list : list[Channel]
        List of Channel objects describing the min, center and max freqs
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
        ssfm_result_list[0].input_signal.time_freq.f_Hz()
        + ssfm_result_list[0].input_signal.time_freq.center_frequency_Hz
    )
    finalSpectrum = ssfm_result_list[-1].spectrum_field_matrix[-1, :]

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
    channel_list : list[Channel]
        List of Channel objects describing the min, center and max
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

    fig, ax = plt.subplots(dpi=300)
    fig.patch.set_facecolor('white')
    for i, channel in enumerate(channel_list):
        signalCenterFreq_list[i] = channel.signal_center_freq_Hz / 1e12
        ax.axvline(x=channel.channel_min_freq_Hz /
                   1e12, color="gray", alpha=0.35)
        ax.axvline(x=channel.channel_max_freq_Hz /
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
    channel_list: list[Channel],
    channelNumber_list: list[int],
    **kwargs
):
    """
    Plots the SNR for the specified channels throughout the fiber span.

    Parameters
    ----------
    ssfm_result_list : list[SSFMResult]
        List of SSFMResult objects containing signal info for each fiber.
    channel_list : list[Channel]
        List of Channel objects describing the min, center and max
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
    fig, ax = plt.subplots(dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_title("Evolution of SNR")

    distance_so_far = 0.0
    for result in ssfm_result_list:
        distance_so_far += result.fiber.length_m
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

    np.random.seed(123)


    os.chdir(os.path.realpath(os.path.dirname(__file__)))

    N = 2 ** 14  # Number of points
    dt = 1e-12  # Time resolution [s]

    center_freq_test = FREQ_1550_NM_Hz  # FREQ_CENTER_C_BAND_HZ
    time_freq_test = TimeFreq(number_of_points=N,
                              time_step_s=dt,
                              center_frequency_Hz=center_freq_test)







    # Set up signal
    test_FFT_tol = 1e-3
    test_amplitude = 0.8
    test_pulse_type = "custom"
    test_duration_s = 1e-9
    test_freq_offset_Hz = 10e9


    test_input_signal = InputSignal(time_freq_test,
                                    test_duration_s,
                                    test_amplitude,
                                    test_pulse_type,
                                    freq_offset_Hz=test_freq_offset_Hz,
                                    FFT_tol=test_FFT_tol)

    test_input_signal.pulse_field+= get_pulse(time_freq_test.t_s(),
                                              test_duration_s,
                                              0,
                                              test_amplitude,
                                              pulse_type='cw',
                                              freq_offset_Hz=test_freq_offset_Hz)

    test_input_signal.pulse_field+= get_pulse(time_freq_test.t_s(),
                                          test_duration_s,
                                          0,
                                          test_amplitude,
                                          pulse_type='cw',
                                          freq_offset_Hz=-test_freq_offset_Hz)

    test_input_signal.update_spectrum()
    test_input_signal.describe_input_signal()





    alpha_test = 0#-0.22/1e3  # dB/m
    beta_list = [0]  # [s^2/m,s^3/m,...]  s^(entry+2)/m
    gamma_test = 1e-3  # 1/W/m
    length_test = 1e3  # m
    number_of_steps = 2**9

    fiber_test = FiberSpan(
        length_test,
        number_of_steps,
        gamma_test,
        beta_list,
        alpha_test,
        use_self_steepening=False)

    fiber_list = [fiber_test]
    fiber_link = FiberLink(fiber_list)



    # Run SSFM
    exp_name = f"FWM_test"


    ssfm_result_list = SSFM(
        fiber_link,
        test_input_signal,
        show_progress_flag=True,
        experiment_name=exp_name,
        FFT_tol=test_FFT_tol
    )


    nrange = 2000
    dB_cutoff = -60


    plot_everything_about_result(
        ssfm_result_list,
        nrange,
        dB_cutoff,
        nrange,
        dB_cutoff,
        show_3D_plot_flag=False)



    exp_name = f"Modulation_Instability_test"


    # Set up signal
    test_FFT_tol = 1e-3
    test_amplitude = 0.1
    test_pulse_type = "CW"
    test_duration_s = 1e-9
    test_freq_offset_Hz = 0.0


    test_input_signal = InputSignal(time_freq_test,
                                    test_duration_s,
                                    test_amplitude,
                                    test_pulse_type,
                                    freq_offset_Hz=test_freq_offset_Hz,
                                    FFT_tol=test_FFT_tol,
                                    noise_stdev_sqrt_W=test_amplitude/1e1)

    alpha_test = 0#-0.22/1e3  # dB/m
    beta_list = [BETA2_AT_1550_NM_TYPICAL_SMF_S2_PER_M]  # [s^2/m,s^3/m,...]  s^(entry+2)/m
    gamma_test = 2e-3  # 1/W/m
    length_test = 100e3  # m
    number_of_steps = 2**9
    print(f"{gamma_test*test_amplitude**2 = }")
    fiber_test = FiberSpan(
        length_test,
        number_of_steps,
        gamma_test,
        beta_list,
        alpha_test,
        use_self_steepening=False)

    fiber_list = [fiber_test]
    fiber_link = FiberLink(fiber_list)

    ssfm_result_list = SSFM(
        fiber_link,
        test_input_signal,
        show_progress_flag=True,
        experiment_name=exp_name,
        FFT_tol=test_FFT_tol
    )


    nrange = 400
    dB_cutoff = -60


    plot_everything_about_result(
        ssfm_result_list,
        nrange,
        dB_cutoff,
        nrange,
        dB_cutoff,
        show_3D_plot_flag=False)



