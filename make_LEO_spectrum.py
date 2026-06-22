import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from astropy.modeling.models import BlackBody
import astropy.units as u
from pathlib import Path

solar_spectrum_file = Path('solar_spectrum.csv') 
solar_data = np.loadtxt(solar_spectrum_file, delimiter=',', skiprows=1)
wavelength, solar_flux = solar_data[:, 0], solar_data[:, 1] #angstrom, erg/s/cm2/angstrom

mag_sun = -26.74
mag_leo = 3 # change to mag 6
satellite_spectrum = 10**(-(mag_leo-mag_sun)/2.5)*solar_flux

col1 = fits.Column(
    name='WAVELENGTH',
    array=wavelength,
    format='D',    
    unit='Angstrom'
)

col2 = fits.Column(
    name='FLUX',
    array=satellite_spectrum,
    format='D',
    unit='erg/s/cm2/Angstrom'
)

hdu = fits.BinTableHDU.from_columns([col1, col2])
hdu.writeto(f'LEO_spectrum_mag{mag_leo}.fits', overwrite=True)