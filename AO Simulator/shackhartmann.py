from hcipy import *
import numpy as np

# class ShackHartmannWavefrontSensorOptics(WavefrontSensorOptics):
#     def __init__(self, input_grid, micro_lens_array):
#         # Make propagator
#         sh_prop = FresnelPropagator(input_grid, micro_lens_array.focal_length)

#         # Make optical system
#         OpticalSystem.__init__(self, (micro_lens_array, sh_prop))
#         self.mla_index = micro_lens_array.mla_index
#         self.mla_grid = micro_lens_array.mla_grid
#         self.micro_lens_array = micro_lens_array

class MySquareShackHartmannWavefrontSensorOptics(ShackHartmannWavefrontSensorOptics):
    ## Helper class to create a Shack-Hartmann WFS with square microlens array
    def __init__(self, input_grid, f_number, num_lenslets, pupil_diameter):
        lenslet_diameter = float(pupil_diameter) / num_lenslets
        #x = np.linspace(-pupil_diameter/2+lenslet_diameter/2, pupil_diameter/2-lenslet_diameter/2, num_lenslets)
        x = np.linspace(-pupil_diameter/2, pupil_diameter/2-lenslet_diameter, num_lenslets)
        self.mla_grid = CartesianGrid(SeparatedCoords((x, x))).shifted((lenslet_diameter/2, lenslet_diameter/2))
        focal_length = f_number * lenslet_diameter
        self.micro_lens_array = MicroLensArray(input_grid, self.mla_grid, focal_length)

        ShackHartmannWavefrontSensorOptics.__init__(self, input_grid, self.micro_lens_array)
