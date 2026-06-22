from hcipy import *
import numpy as np

def make_multiple_sources(telescope_pupil, 
                          off_axis_angle, # (x,y) in arcsec
                          wavelengths): # in meters
    
    wavefronts = []

    for i, theta in enumerate(off_axis_angle):
        electric_field = telescope_pupil*np.cos(theta/3600) #get_electric_field_from_offset(theta, telescope_pupil)
        for wlen in wavelengths[i]:
            wf = Wavefront(electric_field, wlen)
            wavefronts.append(wf)
            
    return wavefronts

def get_electric_field_from_offset(theta, telescope_pupil):
    pass