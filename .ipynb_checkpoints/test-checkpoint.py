from hcipy import *
import numpy as np
import matplotlib.pyplot as plt

# some parameters
pupil_diameter = 0.8 #m
focal_length = 5.6 #m
wavelength = 500e-9 #m

# make a grid of 256x256 for the pupil on which to evaluate later stuff = coordinate system
# pupil_grid = make_pupil_grid(256) # this version works with no units
pupil_grid = make_pupil_grid(256, diameter=pupil_diameter)
print(len(pupil_grid.points))

# there is a function that, for a given grid, says what the telescope pupil shape is

# 1. Initialize function (Later we'll maybe want to create one of these for ZimMAIN)
telescope_pupil_generator = make_magellan_aperture(normalized = True)

# 2. Evaluate function on grid
telescope_pupil = telescope_pupil_generator(pupil_grid)

# create a grid for the focal plane. 
# q = number of pixels between rings, num_airy = number of rings that should be visible
# focal_grid = make_focal_grid(q=8, num_airy=20) # this version works with no units
focal_grid = make_focal_grid(q=8, num_airy=20, 
                             pupil_diameter=pupil_diameter, 
                             focal_length = focal_length, 
                             reference_wavelength = wavelength)

# create a wavefront bouncing off the telescope pupil
# wavefront = Wavefront(telescope_pupil) # this version works with no units
wavefront = Wavefront(telescope_pupil, wavelength)

# initialize a propagator to transport a wavefront from pupil to focal plane
# we are in the far-field diffraction limit, so we can use Fraunhofer approximation
# prop = FraunhoferPropagator(pupil_grid, focal_grid)
prop = FraunhoferPropagator(pupil_grid, focal_grid, focal_length=focal_length)

# evaluate the propagator for the wavefront. Returned focal_image is also a wavefront
focal_image = prop.forward(wavefront)

# PSF is the intensity of this wavefront
psf = focal_image.intensity

# display psf
scaled_psf = np.log10(psf/psf.max())

#supersample telescope pupil at factor 4 for higher precision:
aperture = evaluate_supersampled(telescope_pupil, pupil_grid, 4)
imshow_field(aperture)
#imshow_field(scaled_psf, vmin = -5, grid_units=1e-6)
plt.xlabel('Focal plane distance [µm]')
plt.ylabel('Focal plane distance [µm]')
plt.colorbar()
plt.show()