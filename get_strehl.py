    from hcipy import *
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.ndimage as ndimage
    from matplotlib.animation import FFMpegWriter
    from tqdm.notebook import tqdm
    import astropy.io.fits as fits
    from pathlib import Path, PosixPath
    import math
    from scipy.interpolate import interp1d
    from speclite import filters
    import scipy.integrate as integrate
    from shackhartmann import MySquareShackHartmannWavefrontSensorOptics
    from transmitted_power import transmitted_power
    from csv import DictWriter

    # some telescope parameters

    def get_strehl(seeing=1, 
                   velocity=10, 
                   num_actuators=None, 
                   num_lenslets=None, 
                   SH_FOV=10, 
                   WFS_exp_time=1e-3, 
                   WFS_lambda_min=5000, 
                   WFS_lambda_max=6000, 
                   SC_exp_time=1e-2, 
                   SC_lambda_min=7000, 
                   SC_lambda_max=7500, 
                   beamsplit=0.6):

        telescope_diameter = 0.8 #m
        focal_length = 5.6 #m
        platescale = 206265 / focal_length #arcsec/m
        fov_sci = 2 # arcsec

        spider_width = 0.02 # m this is a guess
        num_spiders = 4
        central_obscuration = 0.1558 # meter
        central_obscuration_ratio = central_obscuration / telescope_diameter
        oversizing_factor = 16 / 16
        area_aperture = math.pi*(telescope_diameter/2)**2 - math.pi*(central_obscuration/2)**2 - (telescope_diameter-central_obscuration)*spider_width*num_spiders # m^2

        num_pupil_pixels = 64 * oversizing_factor
        pupil_diameter = telescope_diameter * oversizing_factor

        # transmission loss parameters

        n_telescope_mirrors = 3
        n_ao_mirrors = 7

        target_spectrum_file = Path('LEO_spectrum.fits') # can be .fits or .csv
        telescope_coating_file = Path('protected_aluminium_reflectance.csv') #microns
        ao_coating_file = Path('protected_aluminium_reflectance.csv')
        qe_file = Path('qe_curve.csv') # angstr; asked ChatGPT to make arrays from an image of QE curve for Oxford Instr. OCAM2K camera 

        wfs_spectral_band = 'r'
        wfs_filter = filters.load_filter(f'sdss2010-{wfs_spectral_band}')

        # make a telescope pupil

        pupil_grid = make_pupil_grid(num_pupil_pixels, diameter=pupil_diameter)
        pupil_aperture_generator = make_obstructed_circular_aperture(telescope_diameter, 
                                                                    central_obscuration_ratio, 
                                                                    num_spiders=num_spiders, 
                                                                    spider_width=spider_width)
        telescope_pupil = evaluate_supersampled(pupil_aperture_generator, pupil_grid, 4)


        if target_spectrum_file.suffix == '.fits':
            hdu = fits.open(target_spectrum_file)
            target_wavelength, target_flux = hdu[1].data['WAVELENGTH'], hdu[1].data['FLUX'] #angstroms, FLAM (erg * s^-1 * cm^-2 * angstr^-1)
        elif target_spectrum_file.suffix == '.csv':
            target_data = np.loadtxt(target_spectrum_file, delimiter=',', skiprows=1)
            target_wavelength, target_flux = target_data[:, 0], target_data[:, 1] 
        target_power = target_flux*area_aperture*1e4

        # reduce target light transmission due to surfaces in telescope and AO

        wavelength_crop1, transmitted_power_telescope = transmitted_power(wavelength = target_wavelength, 
                                                                        in_power = target_power,
                                                                        transmission_file = telescope_coating_file,
                                                                        n_surfaces = n_telescope_mirrors)

        wavelength_crop2, transmitted_power_ao = transmitted_power(wavelength = wavelength_crop1, 
                                                                in_power = transmitted_power_telescope,
                                                                transmission_file = ao_coating_file,
                                                                n_surfaces = n_ao_mirrors)

        # make wavefronts for after propagation through telescope & AO surfaces

        wavefronts_target = []
        for i, wlen in enumerate(wavelength_crop2):
            wf = Wavefront(telescope_pupil, wlen*1e-10) # HCIPy wavefronts take wavelengths in meters only!
            wf.total_power = transmitted_power_ao[i]
            wavefronts_target.append(wf)

        # split wavefronts for WFS and Science Camera

        # WFS beam
        wavelength_wfs = 6000e-10 # m. - reference, central wavelength
        wfs_bandwidth = 1000e-10
        wfs_beamsplit = 0.4 # between 0 (no light) and 1 (all of the light)

        # science camera beam
        wavelength_sci = 4000e-10 # m - central wavelength
        sci_bandwidth = 500e-10

        # make wavefront arrays
        wfs_min, wfs_max = wavelength_wfs - wfs_bandwidth/2, wavelength_wfs + wfs_bandwidth/2
        wfs_wavefronts = np.array(wavefronts_target)[(target_wavelength*1e-10 > wfs_min) & (target_wavelength*1e-10 < wfs_max)]
        for wf in wfs_wavefronts:
            wf.total_power *= wfs_beamsplit
        wfs_wls = [wf.wavelength*1e10 for wf in wfs_wavefronts] # convert wavelengths back to angstrom for plotting
        wfs_power = [wf.total_power for wf in wfs_wavefronts]

        sci_min, sci_max = wavelength_sci - sci_bandwidth/2, wavelength_sci + sci_bandwidth/2
        sci_wavefronts = np.array(wavefronts_target)[(target_wavelength*1e-10 > sci_min) & (target_wavelength*1e-10 < sci_max)]
        for wf in sci_wavefronts:
            if (wf.wavelength < wfs_max) and (wf.wavelength > wfs_min):
                wf.total_power *= (1 - wfs_beamsplit)
        sci_wls = np.array([wf.wavelength*1e10 for wf in sci_wavefronts]) # convert wavelengths back to angstrom for plotting
        sci_power = np.array([wf.total_power for wf in sci_wavefronts])

        # make focal plane

        spatial_res = sci_max * focal_length / telescope_diameter # all variables use meters
        num_airy = fov_sci/platescale / spatial_res
        sci_cam_px_size = 2.9e-6 # meters / px, assuming same pixel size as current LEO camera
        q = math.ceil(spatial_res / sci_cam_px_size)
        focal_grid = make_focal_grid(q=q, num_airy=num_airy, spatial_resolution=spatial_res)
        # q = number of pixels between rings, num_airy = number of rings that should be visible

        seeing = 1 # arcsec @ 500nm (convention) -> r0 = 10.1cm
        outer_scale = 40 # meter
        tau0 = 0.01 # seconds

        fried_parameter = seeing_to_fried_parameter(seeing)
        Cn_squared = Cn_squared_from_fried_parameter(fried_parameter, 500e-9)
        velocity = 0.314 * fried_parameter / tau0

        layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared, outer_scale, velocity)

        prop = FraunhoferPropagator(pupil_grid, focal_grid, focal_length=focal_length)

        # put a camera in the Science beam

        sci_camera = NoiselessDetector(focal_grid)
        sci_camera2 = NoiselessDetector(focal_grid)

        # set SH-WFS parameters

        # choose: a WFS FoV (and previously: WFS wavelength (+bandwidth+beamsplit), seeing at which to operate AO)
        # infer: number of lenslets, WFS diameter (->magnification), 
        #        min. number of req. pixels per subaperture to resolve diffraction limit, and F/#

        num_lenslets = math.ceil(telescope_diameter / fried_parameter)
        if num_lenslets%2 == 1:
            num_lenslets += 1 # make sure it's an even number so the spiders don't obscure so much light
        px_size = 24e-6 #m uniform across all 3 cameras listed in Armasuisse report (should this be coarser than the science camera?)
        fov_wfs = 10 # arcsec - size of structure of LEO

        diff_limit_telescope = 1.22 * wavelength_wfs / telescope_diameter * 206265
        diff_limit_wfs = diff_limit_telescope * num_lenslets
        spot_size_wfs = diff_limit_wfs / platescale
        px_per_subap = math.ceil(fov_wfs/diff_limit_wfs)
        sh_diameter = px_size * px_per_subap * num_lenslets # detector size
        magnification = sh_diameter / telescope_diameter 
        f_coll = focal_length * magnification
        f_sh = f_coll * px_size/spot_size_wfs
        f_number = px_size/wavelength_wfs # alternatively: f_sh / sh_diameter = px_size/wavelength_wfs /1.22 / num_lenslets
        # not sure why we should not divide by num_lenslets still but if we do, we don't get a nice WFS image.

        print(f'{fried_parameter=:f}m \n{num_lenslets=:f} \n{diff_limit_wfs=:f}" \n{f_number=:f} \n{px_per_subap=:f} \n{sh_diameter=:f}m')

        # %%
        # make a SH-WFS using above parameters

        magnifier = Magnifier(magnification)

        # I wrote my own SHWFS class so that the MLA does not center the central lenslet row right onto the spiders which obscure them
        # This new class also fixes a bug in their code where the MLA diameter is double the pupil diameter (this is a bug, there's a forum
        # discussion on it.)
        shwfs = MySquareShackHartmannWavefrontSensorOptics(pupil_grid.scaled(magnification), 
                                                        f_number, 
                                                        num_lenslets, 
                                                        sh_diameter)
        grid = shwfs.mla_grid # this will be useful for plotting


        pupil_aperture_generator_wfs = make_obstructed_circular_aperture(telescope_diameter*magnification, 
                                                                    central_obscuration_ratio, 
                                                                    num_spiders=num_spiders, 
                                                                    spider_width=spider_width*magnification)

        grid2 = pupil_grid.scaled(magnification)

        # put a camera behind the WFS, choose its exposure time

        camera = NoiselessDetector(focal_grid) # includes photon noise

        # choose number of actuators for DM <-> select number of correctable AO modes

        num_actuators = (telescope_diameter/fried_parameter)**2
        num_modes = int(round(num_actuators,0))
        print(num_modes)

        dm_modes = make_zernike_basis(num_modes=num_modes, 
                                    D=telescope_diameter,
                                    grid=pupil_grid, 
                                    use_cache=False) # btw: harmonic disk basis uses fourier bessel basis functions
        dm_modes = ModeBasis([mode / np.ptp(mode) for mode in dm_modes], pupil_grid)

        deformable_mirror = DeformableMirror(dm_modes)

        # calibrate interaction matrix 
        # for this, excite each mode individually and log what happens to the centroids on the detector

        probe_amp = 0.03 * wavelength_wfs
        response_matrix = []

        wf = Wavefront(telescope_pupil, wavelength_wfs)
        wf.total_power = 1

        # Set up animation
        playback_speed = 0.005
        fig = plt.figure(figsize=(10, 6))
        writer = FFMpegWriter(
            fps=5,
            metadata={"title": "DM Response Matrix"},
            bitrate=1800
        )
        output_file = "response_matrix.mp4"

        deformable_mirror.flatten()

        with writer.saving(fig, output_file, dpi=100):

            for i in tqdm(range(num_modes)):
                slope = 0
            
                # Probe the phase response
                amps = [-probe_amp, probe_amp]
                for amp in amps:
                    deformable_mirror.flatten()
                    deformable_mirror.actuators[i] = amp
            
                    dm_wf = deformable_mirror.forward(wf)
                    wfs_wf = shwfs(magnifier(dm_wf))
            
                    camera.integrate(wfs_wf, 1)
                    image = camera.read_out()
            
                    slopes = shwfse.estimate([image])
            
                    slope += amp * slopes / np.var(amps)
            
                response_matrix.append(slope.ravel())
            
                # Only show all modes for the first 40 modes
                if i > 40 and (i + 1) % 20 != 0:
                    continue
            
                # Plot mode response
                plt.clf()
                plt.suptitle('Mode %d / %d' % (i + 1, num_modes), y=0.87)
            
                plt.subplot(1,2,1)
                plt.title('DM surface')
                im1 = imshow_field(deformable_mirror.surface, cmap='RdBu', mask=telescope_pupil)
            
                plt.subplot(1,2,2)
                plt.title('SH spots (slopes not to scale)')
                im2 = imshow_field(image)
                plt.quiver(shwfs.mla_grid.subset(shwfse.estimation_subapertures).x,
                    shwfs.mla_grid.subset(shwfse.estimation_subapertures).y,
                    slope[0,:], slope[1,:],
                    color='white')
            
                writer.grab_frame()

        response_matrix = ModeBasis(response_matrix)

        plt.close(fig)
        print(f"Animation saved to: {output_file}")

        # invert the response matrix to get a function that takes slope as input and gives actuator movement as output

        rcond = 1e-3

        reconstruction_matrix = inverse_tikhonov(response_matrix.transformation_matrix, rcond=rcond)

        # most importantly. Close the loop, with atmosphere

        layer.reset()
        deformable_mirror.flatten()
        # Put actuators at random values, putting a little more power in low-order modes
        deformable_mirror.actuators = np.random.randn(num_modes) / (np.arange(num_modes) + 10)
        # Normalize the DM surface
        deformable_mirror.actuators *= 0.3 * wavelength_sci / np.std(deformable_mirror.surface)

        wfs_exp_time = 1/ operation_freq
        t_atmos = 0.001
        sci_exp_time = 0.1

        gain = 0.5
        leakage = 0.01
        num_iterations = 1000
        playback_speed = 0.1

        fig = plt.figure(figsize=(12, 8))
        writer = FFMpegWriter(
            fps=int(playback_speed / t_atmos),
            metadata={"title": "Closing the Loop with turbulent atmosphere"},
            bitrate=1800
        )
        output_file = f"AO_{int(operation_freq)}Hz_WFS{int(wfs_min*1e10)}A_{int(wfs_max*1e10)}A_Sci{int(sci_min*1e10)}A_{int(sci_max*1e10)}A_BS{int(wfs_beamsplit*100)}pct_SC{sci_exp_time}s_exptime_r0{int(fried_parameter*1e2)}cm_gain{gain}_{num_iterations}iter.mp4"

        strehls = []
        strehls_uncorr = []

        with writer.saving(fig, output_file, dpi=100):
            
            for timestep in tqdm(range(num_iterations)):
                
                layer.t = timestep * t_atmos
                
                phase_screen_phase = layer.phase_for(wavelength_wfs) # in radian
                phase_screen_opd = phase_screen_phase * (wavelength_wfs / (2 * np.pi)) * 1e6 # in um
            
                # WFS: Propagate through atmosphere, deformable mirror, and SHWFS
                wfs_img, wfs_photon_img = 0, 0
                
                for wf in wfs_wavefronts:
                    wf_on_sh = shwfs(magnifier(deformable_mirror(layer(wf))))
                    camera.integrate(wf_on_sh, wfs_exp_time) 
                    #wfs_img += camera.read_out()
                    wfs_photon_img += camera.read_out()*wf.wavelength*1e10*5.03e7

                psf_ref = 0
                for wf in sci_wavefronts:
                    sci_camera.integrate(prop(layer(wf)), t_atmos) # uncorrected
                    sci_camera2.integrate(prop(deformable_mirror(layer(wf))), t_atmos) # corrected
                    psf_ref += prop.forward(wf).power*sci_exp_time

                if int(layer.t*1000) % int(sci_exp_time*1000) == 0: # modulo behaves weird with decimals

                    psf_uncorr = sci_camera.read_out()
                    psf_corr = sci_camera2.read_out()
                
                    strehl = float(get_strehl_from_focal(psf_corr, psf_ref))
                    strehl_uncorr = float(get_strehl_from_focal(psf_uncorr, psf_ref))
                    if timestep/num_iterations > 0.05 and timestep/num_iterations < 0.95: # cut off weird temporal edge behavior
                        strehls.append(strehl)
                        strehls_uncorr.append(strehl_uncorr)
                        
                            
                # Calculate slopes from WFS image
                #wfs_img = large_poisson(wfs_img).astype('float')
                slopes = shwfse.estimate([wfs_photon_img + 1e-10])
                slopes -= slopes_ref
                slopes = slopes.ravel()
                avg_slope = np.mean(slopes)
            
                # Perform wavefront control and set DM actuators
                deformable_mirror.actuators = (1 - leakage) * deformable_mirror.actuators - gain * reconstruction_matrix.dot(slopes)

                # plot: atmosphere, DM, WFS image, corr focal image, uncorr focal image
            
                # Plotting
                if timestep % 1 == 0: # can make this more coarse if needed
                    plt.clf()
            
                    plt.suptitle(f'S={round(strehl,2)}  Gain: {gain}  t = {round(layer.t,3)}  PSF at $\\lambda$={round(wavelength_sci*1e6,2)}µm  WFS at $\\lambda$={round(wavelength_wfs*1e6,2)}µm', y = 0.92)
                    plt.subplots_adjust(wspace=0.5) 
                    
                    plt.subplot(2,3,1)
                    plt.title('DM OPD [$\\mu$m]')
                    imshow_field(deformable_mirror.surface * 1e6, cmap='RdBu', vmin=-1, vmax=1, mask=telescope_pupil)
                    plt.colorbar()
                    plt.xlabel('x [m]')
                    plt.ylabel('y [m]')
            
                    plt.subplot(2,3,2)
                    plt.title('WFS image [counts]')
                    imshow_field(wfs_photon_img, vmin=0, vmax=7, grid_units=1e-3)
                    plt.colorbar()
                    for p in grid.points:
                        d = (grid.x[1]-grid.x[0])*1e3
                        rect = plt.Rectangle(p*1e3 - d/ 2, d, d, linewidth=1, edgecolor=colors.red, facecolor='none')
                        plt.gca().add_patch(rect)
                    plt.xlabel('x [mm]')
                    plt.ylabel('y [mm]')
            
                    plt.subplot(2,3,3)
                    plt.title(f'AO-Corr. PSF $\\mu$m [log]')
                    imshow_field(np.log10(psf_corr/ psf_corr.max()), vmin=-5, vmax=0, grid_units=1e-6) 
                    plt.colorbar()
                    plt.xlabel('x [µm]')
                    plt.ylabel('y [µm]')

                    plt.subplot(2,3,4)
                    plt.title(f'Uncorr. PSF $\\mu$m [log]')
                    imshow_field(np.log10(psf_uncorr/ psf_uncorr.max()), vmin=-5, vmax=0, grid_units=1e-6) 
                    plt.colorbar()
                    plt.xlabel('x [µm]')
                    plt.ylabel('y [µm]')

                    plt.subplot(2,3,5)
                    plt.title('Atmosph. OPD [$\\mu$m]')
                    imshow_field(phase_screen_opd, vmin=-6, vmax=6, cmap='RdBu')
                    plt.colorbar()
                    plt.xlabel('x [m]')
                    plt.ylabel('y [m]')
                    

                writer.grab_frame()
                
        plt.close(fig)
        print(f"Animation saved to: {output_file}")

        strehl_ao = np.mean(strehls)
        strehl_noao = np.mean(strehls_uncorr)

        # save results
        fields=['r0', 'v', 'n_actuators', 'n_lenslets', 'SH_FOV', 'WFS_exp_time', 'WFS_lambda_min', 'WFS_lambda_max',
                'SC_exp_time', 'SC_exp_time', 'SC_lambda_min', 'SC_lambda_max', 'beamsplit', 'Strehl_noAO', 'Strehl_AO']   

        new_row = {'r0': fried_parameter, 
                'v': velocity, 
                'n_actuators': num_actuators,
                'n_lenslets': num_lenslets,
                'SH_FOV': fov_wfs,
                'WFS_exp_time': wfs_exp_time, 
                'WFS_lambda_min': wfs_min, 
                'WFS_lambda_max': wfs_max, 
                'SC_exp_time': sci_exp_time, 
                'SC_lambda_min': sci_min, 
                'SC_lambda_max': sci_max, 
                'beamsplit': wfs_beamsplit, 
                'Strehl_noAO': strehl_noao, 
                'Strehl_AO': strehl_ao}

        with open('ao_performances.txt', 'a') as f:
            writer = DictWriter(f, fieldnames=fields)
            writer.writerow(new_row)
