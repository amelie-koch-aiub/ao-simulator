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
import scipy.integrate as integrate
from shackhartmann import MySquareShackHartmannWavefrontSensorOptics
from transmitted_power import transmitted_power
from csv import DictWriter
import pandas as pd
import yaml
import warnings
import os

warnings.filterwarnings("ignore", category=DeprecationWarning) 

def ao_simulation(simulation_id):

    # load config data
    if not os.path.exists(f'simulations/sim_{simulation_id}'):
        os.mkdir(f'simulations/sim_{simulation_id}')
    config_file = f'configs/config_{simulation_id}.yaml'
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    focal_length = config['focal_length']
    central_obscuration = config['central_obscuration']
    telescope_diameter = config['telescope_diameter']
    spider_width = config['spider_width']
    num_spiders = config['num_spiders']
    fov_wfs = config['fov_wfs']
    fov_sci = config['fov_sci']
    px_scale = float(config['px_scale'])
    px_size = float(config['px_size'])
    num_lenslets = config['num_lenslets']
    n_telescope_mirrors = config['n_telescope_mirrors']
    n_ao_mirrors = config['n_ao_mirrors']
    sci_cam_px_size = float(config['sci_cam_px_size'])
    seeing = config['seeing'] 
    outer_scale = config['outer_scale']
    tau0 = config['tau0']
    velocity = config['velocity'] 
    operation_freq = float(config['operation_freq'])
    wavelength_wfs = float(config['wavelength_wfs']) 
    wfs_bandwidth = float(config['wfs_bandwidth'])
    wfs_beamsplit = config['wfs_beamsplit'] 
    wavelength_sci = float(config['wavelength_sci']) 
    sci_bandwidth = float(config['sci_bandwidth']) 
    sci_exp_time = float(config['sci_exp_time']) 
    wavelength_percentile_to_scale = config['wavelength_percentile_to_scale'] 
    cutoff_below = config['cutoff_below'] 
    interaction_matrix_type = config['interaction_matrix_type'] 
    probe_amp = float(config['probe_amp']) 
    interaction_matrix_output_file = f'simulations/sim_{simulation_id}/dm_calibration.mp4'
    inversion_method = config['inversion_method'] 
    rcond = float(config['rcond']) 
    t_atmos = config['t_atmos'] 
    include_photon_noise = bool(config['include_photon_noise']) 
    mag = config['mag'] 
    gain = config['gain'] 
    leakage = config['leakage'] 
    num_iterations = config['num_iterations'] 
    playback_speed = config['playback_speed'] 
    comments = config['comments'] 

    # set some telescope parameters
    platescale = 206265 / focal_length # [arcsec / m]
    central_obscuration_ratio = central_obscuration / telescope_diameter
    area_aperture = math.pi*(telescope_diameter/2)**2 - math.pi*(central_obscuration/2)**2 - 0.5*(telescope_diameter-central_obscuration)*spider_width*num_spiders # m^2

    # set some WFS parameters
    px_per_subap = round(fov_wfs / px_scale, 0)
    num_pupil_pixels = px_per_subap * num_lenslets 
    pupil_diameter = telescope_diameter

    # get transmission/reflectivity files
    target_spectrum_file = Path(config['target_spectrum_file']) # can be .fits or .csv
    telescope_coating_file = Path(config['coating_file']) # wavelength in [µm]
    ao_coating_file = Path(config['coating_file'])


    # make a telescope pupil
    pupil_grid = make_pupil_grid(num_pupil_pixels, diameter=pupil_diameter)
    pupil_aperture_generator = make_obstructed_circular_aperture(telescope_diameter, 
                                                                central_obscuration_ratio, 
                                                                num_spiders=num_spiders, 
                                                                spider_width=spider_width)
    telescope_pupil = evaluate_supersampled(pupil_aperture_generator, pupil_grid, 16)

    # make an atmosphere
    fried_parameter = seeing_to_fried_parameter(seeing) # [m]
    Cn_squared = Cn_squared_from_fried_parameter(fried_parameter, 500e-9)
    layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared, outer_scale, velocity)

    # get light from target
    hdu = fits.open(target_spectrum_file)
    target_wavelength, target_flux = hdu[1].data['WAVELENGTH'], hdu[1].data['FLUX'] # [angstroms], [FLAM = erg * s^-1 * cm^-2 * angstr^-1]
    target_power = target_flux*area_aperture*1e4 # [erg * s^-1 * angstr^-1]

    # pass it through the system (telescope + AO)
    wavelength_crop1, transmitted_power_telescope = transmitted_power(wavelength = target_wavelength, 
                                                                    in_power = target_power,
                                                                    transmission_file = telescope_coating_file,
                                                                    n_surfaces = n_telescope_mirrors)

    wavelength_crop2, transmitted_power_ao = transmitted_power(wavelength = wavelength_crop1, 
                                                            in_power = transmitted_power_telescope,
                                                            transmission_file = ao_coating_file,
                                                            n_surfaces = n_ao_mirrors)

    # make WFS & SC wavefronts for this target
    wavefronts_target = []
    for i, wlen in enumerate(wavelength_crop2): 
        wf = Wavefront(telescope_pupil, wlen*1e-10) # HCIPy wavefronts take wavelengths in meters only!
        wf.total_power = transmitted_power_ao[i] # [erg * s^-1 * angstr^-1] 
        wavefronts_target.append(wf)
    target_wlens = np.array([wf.wavelength for wf in wavefronts_target])
    target_powers = np.array([wf.total_power for wf in wavefronts_target]) 

    wfs_min, wfs_max = wavelength_wfs - wfs_bandwidth/2, wavelength_wfs + wfs_bandwidth/2
    wfs_wavefronts = np.array(wavefronts_target)[(target_wlens > wfs_min) & (target_wlens < wfs_max)]
    for wf in wfs_wavefronts:
        wf.total_power = wfs_beamsplit*np.array(target_powers)[(target_wlens == wf.wavelength)][0]
    wfs_wls = [wf.wavelength*1e10 for wf in wfs_wavefronts] # convert wavelengths back to angstrom for plotting
    wfs_power = [wf.total_power for wf in wfs_wavefronts]

    target_copy2 = wavefronts_target.copy()
    sci_min, sci_max = wavelength_sci - sci_bandwidth/2, wavelength_sci + sci_bandwidth/2
    sci_wavefronts = np.array(wavefronts_target)[(target_wlens > sci_min) & (target_wlens < sci_max)]
    for wf in sci_wavefronts:
        if (wf.wavelength < wfs_max) and (wf.wavelength > wfs_min):
            wf.total_power = (1 - wfs_beamsplit)*np.array(target_powers)[(target_wlens == wf.wavelength)][0]
    sci_wls = np.array([wf.wavelength*1e10 for wf in sci_wavefronts]) # convert wavelengths back to angstrom for plotting
    sci_power = np.array([wf.total_power for wf in sci_wavefronts])

    # make a grid for the SC focal plane & add two detectors
    spatial_res = 1.22 * sci_max * focal_length / telescope_diameter # all variables use meters
    num_airy = fov_sci/platescale / spatial_res
    q = math.ceil(spatial_res / sci_cam_px_size) # q = number of pixels between rings, num_airy = number of rings that should be visible
    focal_grid_sci = make_focal_grid(q=q, num_airy=num_airy, spatial_resolution=spatial_res)
    n_pixels_sci = len(focal_grid_sci.coords[0])
    science_detector_size = np.sqrt(n_pixels_sci) * sci_cam_px_size
    prop = FraunhoferPropagator(pupil_grid, focal_grid_sci, focal_length=focal_length)
    sci_camera = NoiselessDetector(focal_grid_sci) # uncorrected beam
    sci_camera2 = NoiselessDetector(focal_grid_sci) # AO corrected beam

    h = 6.626e-27 # erg * s
    c = 3e5 * 1e13 # Angstrom / s

    # set some more WFS parameters
    wlen_to_scale = wfs_min + wfs_bandwidth*wavelength_percentile_to_scale
    diff_limit_telescope = 1.22 * wlen_to_scale / telescope_diameter * 206265 # [arcsec]
    spot_size_telescope = diff_limit_telescope / platescale # [m]
    diff_limit_wfs = diff_limit_telescope * num_lenslets # [arcsec]
    n_px_one_spot = round(diff_limit_wfs / px_scale,0)
    spot_size_wfs = n_px_one_spot * px_size # [m]
    wfs_diameter = px_size * px_per_subap * num_lenslets 
    lenslet_diameter = px_size * px_per_subap 
    magnification_pupil = wfs_diameter / telescope_diameter 
    f_coll = focal_length * magnification_pupil 
    magnification_focal = spot_size_wfs / spot_size_telescope
    f_wfs = f_coll * magnification_focal
    f_number = f_wfs / wfs_diameter
    platescale_wfs = platescale / magnification_focal
    magnifier = Magnifier(magnification_pupil)

    # make WFS
    shwfs = MySquareShackHartmannWavefrontSensorOptics(pupil_grid.scaled(magnification_pupil), 
                                                    f_number, 
                                                    num_lenslets, 
                                                    wfs_diameter)
    focal_grid_wfs = pupil_grid.scaled(magnification_pupil)
    n_pixels_wfs = int(len(focal_grid_wfs.coords[0]))
    #print(f'total number of pixels used on the WFS detector: {n_pixels_wfs}')
    camera = NoiselessDetector(focal_grid_wfs)
    wfs_exp_time = 1 / operation_freq
    grid = shwfs.mla_grid 

    # read out reference WFS image
    counts_per_angstr_calm = np.zeros((len(wfs_wavefronts), n_pixels_wfs))
    photon_counts_total = []

    for i, wf in enumerate(wfs_wavefronts):

        E_photon = h * c / (wf.wavelength*1e10) # [erg]
        camera.integrate(shwfs(magnifier(wf)), wfs_exp_time)
        data1 = camera.read_out() # [erg * angstr^-1]
        counts_per_angstr_calm[i] = data1 / E_photon
        photon_counts_total.append(np.sum(data1 / E_photon))

    photon_in_cam = integrate.simpson(photon_counts_total, wfs_wls)
    #print(f'Expecting {round(photon_in_cam*1e-6, 2)}e6 photons/s in the WFS at {int(wfs_min*1e10)}-{int(wfs_max*1e10)} Angstroms, {round(photon_in_cam/(num_lenslets)**2,1)} photons per subaperture at {operation_freq}Hz.')

    photon_counts_calm = Field(np.zeros(len(counts_per_angstr_calm[0])), focal_grid_wfs)
    for i in range(len(counts_per_angstr_calm[0])): # loop over pixels, integrate all wavelengths
        photon_counts_calm[i] = integrate.simpson(counts_per_angstr_calm[:, i], wfs_wls)
    shwfse = ShackHartmannWavefrontSensorEstimator(shwfs.mla_grid, shwfs.micro_lens_array.mla_index)

    # decide which subapertures to use for WFS estimation
    fluxes = ndimage.measurements.sum(photon_counts_calm, shwfse.mla_index, shwfse.estimation_subapertures)
    flux_limit = fluxes.max() * cutoff_below
    estimation_subapertures = shwfs.mla_grid.zeros(dtype='bool')
    estimation_subapertures[shwfse.estimation_subapertures[fluxes > flux_limit]] = True
    shwfse = ShackHartmannWavefrontSensorEstimator(shwfs.mla_grid, shwfs.micro_lens_array.mla_index, estimation_subapertures)
    illuminated_subapertures = len(shwfse.estimation_subapertures)
    photon_count_per_subap = round(np.mean(ndimage.measurements.sum(photon_counts_calm, shwfse.mla_index, shwfse.estimation_subapertures)),1)
    max_snr = round(np.sqrt(np.mean(ndimage.measurements.maximum(photon_counts_calm, shwfse.mla_index, shwfse.estimation_subapertures))),2)
    avg_snr = round(np.sqrt(np.mean(ndimage.measurements.sum(photon_counts_calm, shwfse.mla_index, shwfse.estimation_subapertures))/(px_per_subap)**2),2)

    # estimate reference slopes for central wavelength
    wf = wavefronts_target[245]
    E_photon = h * c / (wf.wavelength*1e10) # [erg]
    camera.integrate(shwfs(magnifier(wf)), wfs_exp_time)
    counts_ref = camera.read_out() # [erg * angstr^-1]
    counts_ref /= E_photon
    slopes_ref = shwfse.estimate([counts_ref]) # can I use photon counts as a unit for slope estimation?

    # choose number of actuators for DM, make DM
    if interaction_matrix_type == 'zonal':
        num_actuators = num_lenslets + 1 #(telescope_diameter/fried_parameter)**2 #len(shwfse.estimation_subapertures) +1 
        actuator_spacing = pupil_diameter / (num_actuators-1)
        actuator_grid = make_actuator_positions(num_actuators,
                                                actuator_spacing=actuator_spacing)
        influence_functions = make_gaussian_influence_functions(pupil_grid,
                                                                num_actuators_across_pupil=num_actuators,
                                                                actuator_spacing=actuator_spacing)

    elif interaction_matrix_type == 'modal':
        num_actuators = (telescope_diameter/fried_parameter)**2
        num_modes = int(round(num_actuators,0))    
        influence_functions = make_zernike_basis(num_modes=num_modes, 
                                    D=telescope_diameter,
                                    grid=pupil_grid, 
                                    use_cache=False) # btw: harmonic disk basis uses fourier bessel basis functions
    else:
        print(f'Error: interaction matrix type not recognized, must be zonal or modal, you gave: {interaction_matrix_type}')

    norm_modes = [mode / np.ptp(mode) for mode in influence_functions]
    dm_modes = ModeBasis(norm_modes, pupil_grid)

    deformable_mirror = DeformableMirror(dm_modes) 

    # calibrate interaction matrix 
    # for this, excite each mode individually and log what happens to the centroids on the detector

    #print(f'DM has stroke {probe_amp}m')
    response_matrix = []

    wf = wavefronts_target[245]
    wf.total_power = 1
    E_photon = h * c / (wf.wavelength*1e10)

    # Set up animation
    fig = plt.figure(figsize=(10, 6))
    writer = FFMpegWriter(
        fps=2,
        metadata={"title": "DM Interaction Matrix Calibration"},
        bitrate=1800
    )

    deformable_mirror.flatten()

    with writer.saving(fig, interaction_matrix_output_file, dpi=100):

        for i in tqdm(range(len(norm_modes))):
            slope = 0
        
            # Probe the phase response
            amps = [-probe_amp, probe_amp]
            for amp in amps:
                deformable_mirror.flatten()
                deformable_mirror.actuators[i] = amp
        
                dm_wf = deformable_mirror.forward(wf)
                wfs_wf = shwfs(magnifier(dm_wf))

                camera.integrate(wfs_wf, 1e-3)
                image = camera.read_out()
                #image /= E_photon
        
                slopes = shwfse.estimate([image])
                #slopes -= slopes_ref
                slope += amp * slopes / np.var(amps)
        
            response_matrix.append(slope.ravel())
        
            # Only show all modes for the first 40 modes
            #if i > 40 and (i + 1) % 20 != 0:
            #  continue
        
            # Plot mode response
            plt.clf()
            plt.suptitle('Mode %d / %d' % (i + 1, num_actuators), y=0.87)
        
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
    print(f"Animation saved to: {interaction_matrix_output_file}")

    # invert response matrix
    if inversion_method == 'tikhonov':
        reconstruction_matrix = inverse_tikhonov(response_matrix.transformation_matrix, rcond=rcond)
    elif inversion_method == 'truncation':
        reconstruction_matrix = inverse_truncated(response_matrix.transformation_matrix, rcond=rcond)
    else:
        print('Error: Inversion method not recognized.')

    # Close the loop, with atmosphere
    layer.reset()
    deformable_mirror.flatten()

    fig = plt.figure(figsize=(12, 8))
    writer = FFMpegWriter(
        fps=int(playback_speed / t_atmos),
        metadata={"title": "Closing the Loop with turbulent atmosphere"},
        bitrate=1800
    )
    #specs_string =f'{int(operation_freq)}Hz_WFS{int(wfs_min*1e10)}A_{int(wfs_max*1e10)}A_Sci{int(sci_min*1e10)}A_{int(sci_max*1e10)}A_BS{int(wfs_beamsplit*100)}pct_SC{sci_exp_time}s_exptime_r0{int(fried_parameter*1e2)}cm_gain{gain}_stroke{round(probe_amp*1e6,2)}microns_{num_iterations}iter_photonnoise_{include_photon_noise}'
    output_file = f"simulations/sim_{simulation_id}/ao_animation.mp4"

    strehls = []
    strehls_uncorr = []

    wfs_energy = 0
    wfs_photon_image = 0
    counts_per_angstr_turb = np.zeros((len(wfs_wavefronts), n_pixels_wfs))
    photon_counts_turb = Field(np.zeros(len(counts_per_angstr_calm[0])), focal_grid_wfs)


    with writer.saving(fig, output_file, dpi=100):
        
        for timestep in tqdm(range(num_iterations)):

            # atmosphere evolution
            layer.t = timestep * t_atmos
            phase_screen_phase = layer.phase_for(wavelength_wfs) # [rad]
            phase_screen_opd = phase_screen_phase * (wavelength_wfs / (2 * np.pi)) * 1e6 # [µm]
        
            # WFS image capture
            for i, wf in enumerate(wfs_wavefronts):

                E_photon = h * c / (wf.wavelength*1e10) # [erg]
                camera.integrate(shwfs(magnifier(deformable_mirror(layer(wf)))), t_atmos) 
                wfs_data = camera.read_out()
                counts_per_angstr_turb[i] = wfs_data / E_photon

            for i in range(len(counts_per_angstr_turb[0])): # for each pixel, integrate photon counts across all wavelengths
                photon_counts_turb[i] += integrate.simpson(counts_per_angstr_turb[:, i], wfs_wls)

            if int(layer.t*10000) % int(wfs_exp_time*10000) == 0:
                wfs_photon_image_to_plot = photon_counts_turb
                photon_counts_turb = Field(np.zeros(len(counts_per_angstr_calm[0])), focal_grid_wfs)

            # Science image capture (corrected and uncorrected)
            psf_ref = 0
            for wf in sci_wavefronts:
                sci_camera.integrate(prop(layer(wf)), t_atmos) # uncorrected
                sci_camera2.integrate(prop(deformable_mirror(layer(wf))), t_atmos) # corrected
                psf_ref += prop.forward(wf).power*sci_exp_time #*prop.forward(wf).wavelength*1e10*5.03e7

            if int(layer.t*1000) % int(sci_exp_time*1000) == 0: # modulo behaves weird with decimals

                psf_uncorr = sci_camera.read_out()
                psf_corr = sci_camera2.read_out()
            
                strehl = float(get_strehl_from_focal(psf_corr, psf_ref))
                strehl_uncorr = float(get_strehl_from_focal(psf_uncorr, psf_ref))
                #if timestep/num_iterations > 0.05:# and timestep/num_iterations < 0.95: # cut off weird temporal edge behavior
            strehls.append(strehl)
            strehls_uncorr.append(strehl_uncorr)
                    
                        
            # Calculate slopes from WFS image
            if include_photon_noise:
                wfs_photon_image_to_plot = large_poisson(wfs_photon_image_to_plot).astype('float')
            slopes = shwfse.estimate([wfs_photon_image_to_plot + 1e-20])
            slopes -= slopes_ref
            slopes_copy = slopes
            slopes = slopes.ravel()
            # TODO: normalize by amplitude applied to DM?
        
            # Set DM actuators
            deformable_mirror.actuators = (1 - leakage) * deformable_mirror.actuators - gain * reconstruction_matrix.dot(slopes)

            # plot: atmosphere, DM, WFS image, corr focal image, uncorr focal image
            if timestep % 1 == 0: # can make this plotting frequency more coarse if needed
                plt.clf()
        
                plt.suptitle(f'Gain: {gain}  PSF at $\\lambda$={round(wavelength_sci*1e6,2)}µm  WFS at $\\lambda$={round(wavelength_wfs*1e6,2)}µm  DM stroke: {round(probe_amp*1e6,2)}µm  WFS freq = {operation_freq}Hz  t = {round(layer.t,3)}  S={round(strehl,2)} (uncorr: {round(strehl_uncorr,2)}', y = 0.92)
                plt.subplots_adjust(wspace=0.5) 
                
                plt.subplot(2,3,1)
                plt.title('DM OPD [$\\mu$m]')
                imshow_field(deformable_mirror.surface * 1e6, cmap='RdBu', vmin=-4, vmax=4, mask=telescope_pupil)
                plt.colorbar()
                plt.xlabel('x [m]')
                plt.ylabel('y [m]')
        
                plt.subplot(2,3,2)
                plt.title('WFS image [counts]')
                imshow_field(wfs_photon_image_to_plot, vmin=0, vmax=np.max(wfs_photon_image_to_plot), grid_units=1e-3)
                plt.colorbar()
                for p in grid.points:
                    d = (grid.x[1]-grid.x[0])*1e3
                    rect = plt.Rectangle(p*1e3 - d/ 2, d, d, linewidth=1, edgecolor=colors.red, facecolor='none')
                    plt.gca().add_patch(rect)
                plt.xlabel('x [mm]')
                plt.ylabel('y [mm]')
        
                plt.subplot(2,3,3)
                plt.title(f'AO-Corr. PSF $\\mu$m [log]')
                imshow_field(np.log10(psf_corr/psf_corr.max()), vmin=-5, vmax=0, grid_units=1e-6) 
                plt.colorbar()
                plt.xlabel('x [µm]')
                plt.ylabel('y [µm]')

                plt.subplot(2,3,4)
                plt.title(f'Uncorr. PSF $\\mu$m [log]')
                imshow_field(np.log10(psf_uncorr/psf_uncorr.max()), vmin=-5, vmax=0, grid_units=1e-6) 
                plt.colorbar()
                plt.xlabel('x [µm]')
                plt.ylabel('y [µm]')

                plt.subplot(2,3,5)
                plt.title('Atmosph. OPD [$\\mu$m]')
                imshow_field(phase_screen_opd, vmin=-4, vmax=4, cmap='RdBu')
                plt.colorbar()
                plt.xlabel('x [m]')
                plt.ylabel('y [m]')

                plt.subplot(2,3,6)
                plt.title('slopes')
                imshow_field(wfs_photon_image_to_plot, vmin=0, vmax=0)
                plt.quiver(shwfs.mla_grid.subset(shwfse.estimation_subapertures).x,
                shwfs.mla_grid.subset(shwfse.estimation_subapertures).y,
                slopes_copy[0,:]*mag, slopes_copy[1,:]*mag,
                color='white')
                plt.xlabel('x [m]')
                plt.ylabel('y [m]')
                

            writer.grab_frame()
            
    plt.close(fig)
    print(f"Animation saved to: {output_file}")

    # calculate average Strehls
    strehl_ao = np.mean(strehls)
    strehl_noao = np.mean(strehls_uncorr)

    # save results & plots
    new_row = {'simulation_ID': [simulation_id],
            'seeing ["]': [seeing],
            'r0 [m]': [fried_parameter], 
            'v [m/s]': [velocity], 
            'n_actuators': [num_actuators],
            'n_lenslets': [num_lenslets],
            'n_illuminated_subapertures': [len(shwfse.estimation_subapertures)],
            'px_per_subap': [px_per_subap],
            'SH_FOV ["]': [fov_wfs],
            'diff_limit_wfs ["]': [diff_limit_wfs],
            'WFS platescale ["/m]': [platescale_wfs],
            'WFS_exp_time [s]': [wfs_exp_time], 
            'WFS_lambda_min [A]': [wfs_min], 
            'WFS_lambda_max [A]': [wfs_max], 
            'SC_exp_time [s]': [sci_exp_time], 
            'SC_lambda_min [A]': [sci_min], 
            'SC_lambda_max [A]': [sci_max], 
            'beamsplit [%]': [wfs_beamsplit], 
            'DM_stroke [m]': [probe_amp],
            'gain': [gain],
            'leakage': [leakage],
            'WFS max_SNR': [max_snr],
            'WFS avg_SNR': [avg_snr],
            'interaction_matrix_type': [interaction_matrix_type],
            'inversion_cutoff': [rcond],
            'inversion_method': [inversion_method],
            'illumination_threshold': [cutoff_below],
            'num_iterations': [num_iterations],
            'photon_noise': [include_photon_noise],
            'avg_Strehl_noAO': [strehl_noao], 
            'avg_Strehl_AO': [strehl_ao],
            'peak_Strehl_AO': [np.max(strehls)],
            'comments': [comments],
            'loop_closure / quality': ['unassessed']}

    if os.path.exists('ao_performance.csv'):
        df = pd.read_csv('ao_performance.csv')
        tmp_df = pd.DataFrame.from_dict(new_row)
        df = pd.concat([df, tmp_df], ignore_index=True)
    else:
        df = pd.DataFrame.from_dict(new_row)
    df.to_csv('ao_performance.csv', index = False)

    times = np.linspace(0, num_iterations*t_atmos, num_iterations)
    df_evol = pd.DataFrame({'time': times, 'strehl_corr': strehls, 'strehl_uncorr': strehls_uncorr})
    df_evol.to_csv(f'simulations/sim_{simulation_id}/strehl_evol.csv')

    plt.plot(times, strehls, label='AO corrected')
    plt.plot(times, strehls_uncorr, label='uncorrected')
    plt.legend()
    plt.xlabel('time [s]')
    plt.ylabel('Strehl')
    plt.title('Strehl value over time as loop closes')
    plt.savefig(f'simulations/sim_{simulation_id}/strehl_evol.png')
    plt.clf()

    plt.imshow(response_matrix)
    plt.xlabel('slope x, slope y')
    plt.ylabel('actuator')
    plt.savefig(f'simulations/sim_{simulation_id}/response_matrix.png')
    plt.clf()
    plt.imshow(reconstruction_matrix)
    plt.xlabel('slope x, slope y')
    plt.ylabel('actuator')
    plt.savefig(f'simulations/sim_{simulation_id}/reconstruction_matrix.png')
    plt.clf()


    x = shwfs.mla_grid.subset(shwfse.estimation_subapertures).x
    y = shwfs.mla_grid.subset(shwfse.estimation_subapertures).y
    act = 25
    vec = reconstruction_matrix[act]
    centroids = vec.reshape(2, -1)
    dx = centroids[0]
    dy = centroids[1]
    demag = 0.5
    plt.figure(figsize=(6,6))
    plt.title(f"Reconstruction matrix actuator {act}")
    plt.scatter(x, y, s=20, color='k')
    plt.quiver(
        x, y,
        dx*demag, dy*demag,
        angles='xy',
        scale_units='xy',
        scale=1,
        color='red'
    )
    plt.gca().set_aspect('equal')
    plt.xlabel("Pupil x")
    plt.ylabel("Pupil y")
    plt.savefig(f'simulations/sim_{simulation_id}/example_reconstruction.png')
    plt.clf()

    svd = np.linalg.svd(response_matrix.transformation_matrix)[1]
    plt.plot(np.arange(0,len(svd)), svd)
    cutoff_index = np.where(svd >= rcond)[0][-1]
    plt.axvline(cutoff_index, linestyle='dashed', color='red')
    plt.yscale("log")
    plt.xlabel('actuator number')
    plt.ylabel('Eigenvalue')
    plt.title('Response matrix SVD with truncation threshold')
    plt.savefig(f'simulations/sim_{simulation_id}/svd_response_matrix.png')
    plt.clf()


    plt.title('SH slopes Flat field')
    im2 = imshow_field(counts_ref)
    plt.quiver(shwfs.mla_grid.subset(shwfse.estimation_subapertures).x,
        shwfs.mla_grid.subset(shwfse.estimation_subapertures).y,
        slopes_ref[0,:], slopes_ref[1,:],
        color='white')
    plt.savefig(f'simulations/sim_{simulation_id}/dm_flat_slope.png')
    plt.clf()


    plt.plot(target_wavelength, target_power, label = 'target spectrum')
    plt.plot(wavelength_crop2, transmitted_power_ao, label = 'output AO')
    plt.plot(wavelength_crop1, transmitted_power_telescope, label='output telescope')
    plt.plot(wfs_wls, wfs_power, color='red', label = 'WFS input')
    plt.fill_between(wfs_wls, wfs_power, color = 'red', alpha=0.2)
    plt.plot(sci_wls, sci_power, color = 'purple', label = 'Sciencecam input')
    plt.fill_between(sci_wls, sci_power, color = 'purple', alpha=0.2)
    plt.legend()
    plt.xlabel('wavelength [angstr.]')
    plt.ylabel('power [erg/s/angstr./m^2]')
    plt.xlim(3500, 15000)
    plt.ylim(0,1.5e-7)
    plt.savefig(f'simulations/sim_{simulation_id}/input_light.png')
    plt.clf()

if __name__ == '__main__':
    ao_simulation(simulation_id='35')
    # ao_simulation(simulation_id='29') 
    # ao_simulation(simulation_id='17')
    # ao_simulation(simulation_id='18')
    # ao_simulation(simulation_id='19')

