from hcipy import *
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from pathlib import Path, PosixPath
import math
from scipy.interpolate import interp1d
from speclite import filters
import scipy.integrate as integrate

def transmitted_power(wavelength, in_power, transmission_file, n_surfaces):

    # input wavelength should be in angstrom, power in erg/s/m^2

    if type(transmission_file) == PosixPath:

        refl_data = np.loadtxt(transmission_file, delimiter=',', skiprows=1)
        wavelength_trans, frac_transmitted = refl_data[:, 0], refl_data[:, 1] # unit TBD, percent

        if wavelength_trans[0] < 1e-8: # wavelength likely given in m
            wavelength_trans *= 1e10
        elif (wavelength_trans[0] < 10) and (wavelength_trans[0] > 1e-3): # wavelength likely given in microns, convert to angstrom
            wavelength_trans *= 1e4
        elif (wavelength_trans[0] > 10) and (wavelength_trans[0] < 600): # given in nanometers
            wavelength_trans *= 10
        # else: wavelength in angstrom, hopefully
            

    elif type(transmission_file) == filters.FilterResponse:

        wfs_filter = transmission_file
        wavelength_trans, frac_transmitted = wfs_filter.wavelength, wfs_filter.response*100 # angstr., percent
        expanded_transmittance = np.interp(
            wavelength,
            wavelength_trans,
            frac_transmitted,
            left=0.0,
            right=0.0
        )
        wavelength_trans = wavelength
        frac_transmitted = expanded_transmittance

    wl_min = wavelength_trans.min()
    wl_max = wavelength_trans.max()

    mask = ((wavelength >= wl_min) &
            (wavelength <= wl_max))

    indices = np.where(mask)[0]

    wavelength_cropped = wavelength[mask]
    in_power_cropped = in_power[indices[0]:indices[-1]+1]

    interp_refl = interp1d(
        wavelength_trans,
        frac_transmitted,
        kind='linear', 
        fill_value='extrapolate'
    )
    
    frac_transmitted_on_power_grid = interp_refl(wavelength_cropped)
    transmitted_power = in_power_cropped*(frac_transmitted_on_power_grid*0.01)**(n_surfaces)
    
    return wavelength_cropped, transmitted_power