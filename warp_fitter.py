import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_setup import *
import pickle
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import utils as mu
import glob
import re
from matplotlib.patches import Circle
from adjustText import adjust_text
from scipy.stats import spearmanr, kendalltau
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C

import json
import subprocess
from astropy.io import fits

# Base map: only one entry per clean display name
base_disc_name_map = {
    'mwc758': "MWC 758",
    'mwc758_wc': "MWC 758",
    'v4046': "V4046 Sgr",
    'aatau': "AA Tau",
    'cqtau': "CQ Tau",
    'hd34282': "HD 34282",
    'dmtau': "DM Tau",
    'hd135344': "HD 135344B",
    'hd143006': "HD 143006",
    'j1604': "J1604",
    'j1615': "J1615",
    'j1842': "J1842",
    'j1852': "J1852",
    'lkca15': "LkCa15",
    'pds66': "PDS 66",
    'sycha': "SY Cha",
	'hd34700': "HD 34700",
}

# 12CO targets with _dbell suffix
dbell_12co = {"aatau", "hd34282", "dmtau", "j1615", "j1842", "lkca15", "sycha"}

# 13CO targets with _dbell suffix (corrected)
dbell_13co = {"aatau", "hd34282", "j1615", "lkca15", "sycha"}  # dmtau and j1842 removed


# 12CO targets with lower resolution suffix
b0p03_12co ={} #{"aatau", "hd34282", "dmtau", "j1615", "lkca15", "sycha"}

# 13CO targets with lower esolutionsuffix (corrected)
b0p03_13co ={} #{"aatau", "hd34282", "dmtau", "j1615", "lkca15", "sycha", "pds66"} 

# Build full map with all suffix combinations
disc_name_map = {}

for key, display in base_disc_name_map.items():
    # Always include the base
    disc_name_map[key] = display

    # Add 12CO _dbell variant if applicable
    disc_name_map[f"{key}_dbell"] = display
	
    disc_name_map[f"{key}_dbell_b0p30"] = display
    disc_name_map[f"{key}_b0p30"] = display
	

    # Add 13CO base + 13CO _dbell variants
    disc_name_map[f"{key}_13co"] = display
    disc_name_map[f"{key}_13co_dbell"] = display
	
    disc_name_map[f"{key}_13co_dbell_b0p30"] = display
    disc_name_map[f"{key}_13co_b0p30"] = display

vmax_dict_12co = {"MWC 758": 0.3, "V4046 Sgr": 0.1, "AA Tau": 0.3, "CQ Tau": 0.4, "HD 34282": 0.3, "DM Tau": 0.15, "HD 135344B": 0.25, "HD 143006": 0.15, "J1604": 0.075, "J1615": 0.15, "J1842": 0.15, "J1852": 0.1, "LkCa15": 0.25, "PDS 66": 0.1, "SY Cha": 0.3, "HD 34700": 0.7}

vmax_dict_13co = {"MWC 758": 0.2, "V4046 Sgr": 0.1, "AA Tau": 0.3, "CQ Tau": 0.3, "HD 34282": 0.3, "DM Tau": 0.1, "HD 135344B": 0.1, "HD 143006": 0.1, "J1604": 0.05, "J1615": 0.15, "J1842": 0.15, "J1852": 0.1, "LkCa15": 0.25, "PDS 66": 0.1, "SY Cha": 0.3}

"""
comparison_data = {
    'AA Tau':    {'log_Mdot': -8.1, 'NAI': 0.120, 'NIR_excess': 4.7,  'NIR_ulim': False,
                  'alpha_S': 6.54, 'alpha_S_err_low': 3.62, 'alpha_S_err_high': 8.11},
    'CQ Tau':    {'log_Mdot': -7.0, 'NAI': 0.111, 'NIR_excess': 25.4, 'NIR_ulim': False},
    'DM Tau':    {'log_Mdot': -8.2, 'NAI': 0.092, 'NIR_excess': 0.6,  'NIR_ulim': True,
                  'alpha_S': 2.19, 'alpha_S_err_low': 1.21, 'alpha_S_err_high': 2.72},
    'HD 135344B':  {'log_Mdot': -8.0, 'NAI': 0.405, 'NIR_excess': 27.2, 'NIR_ulim': False},
    'HD 143006':  {'log_Mdot': -8.1, 'NAI': 0.215, 'NIR_excess': 21.3, 'NIR_ulim': False},
    'HD 34282':   {'log_Mdot': -7.7, 'NAI': 0.114, 'NIR_excess': 9.2,  'NIR_ulim': False,
                  'alpha_S': 3.86, 'alpha_S_err_low': 2.14, 'alpha_S_err_high': 4.79},
    'J1604':     {'log_Mdot': -10.5,'NAI': 0.059, 'NIR_excess': 17.5, 'NIR_ulim': False},
    'J1615':     {'log_Mdot': -8.5, 'NAI': 0.038, 'NIR_excess': 0.9,  'NIR_ulim': True,
                  'alpha_S': 1.20, 'alpha_S_err_low': 0.66, 'alpha_S_err_high': 1.48},
    'J1842':     {'log_Mdot': -8.8, 'NAI': 0.074, 'NIR_excess': 12.3, 'NIR_ulim': False,
                  'alpha_S': 0.403, 'alpha_S_err_low': 0.223, 'alpha_S_err_high': 0.499},
    'J1852':     {'log_Mdot': -8.7, 'NAI': 0.024, 'NIR_excess': 1.1,  'NIR_ulim': True,
                  'alpha_S': 0.362, 'alpha_S_err_low': 0.201, 'alpha_S_err_high': 0.449},
    'LkCa15':    {'log_Mdot': -8.4, 'NAI': 0.053, 'NIR_excess': 13.4, 'NIR_ulim': False,
                  'alpha_S': 0.174, 'alpha_S_err_low': 0.097, 'alpha_S_err_high': 0.216},
    'MWC 758':   {'log_Mdot': -7.15, 'NAI': 0.429, 'NIR_excess': 27.5, 'NIR_ulim': False},
    'PDS 66':     {'log_Mdot': -9.9, 'NAI': 0.014, 'NIR_excess': 7.3,  'NIR_ulim': False,
                  'alpha_S': 0.0806, 'alpha_S_err_low': 0.0446, 'alpha_S_err_high': 0.0999},
    'SY Cha':    {'log_Mdot': -9.2, 'NAI': 0.075, 'NIR_excess': 7.6,  'NIR_ulim': False,
                  'alpha_S': 0.0166, 'alpha_S_err_low': 0.0092, 'alpha_S_err_high': 0.0205},
    'V4046 Sgr':     {'log_Mdot': -9.3, 'NAI': 0.030, 'NIR_excess': 0.9,  'NIR_ulim': True,
                  'alpha_S': 0.111, 'alpha_S_err_low': 0.061, 'alpha_S_err_high': 0.138}
}
"""

comparison_data = {
    'AA Tau':    {'log_Mdot': -8.1,  'NAI': 0.120, 'NIR_excess': 4.7,  'NIR_excess_err': 3.6,  'NIR_ulim': False,
                  'alpha_S': 6.54,   'alpha_S_err_low': 3.62, 'alpha_S_err_high': 8.11},
    'CQ Tau':    {'log_Mdot': -7.0,  'NAI': 0.111, 'NIR_excess': 25.4, 'NIR_excess_err': 2.5,  'NIR_ulim': False},
    'DM Tau':    {'log_Mdot': -8.2,  'NAI': 0.092, 'NIR_excess': 0.6,  'NIR_ulim': True,
                  'alpha_S': 2.19,   'alpha_S_err_low': 1.21, 'alpha_S_err_high': 2.72},
    'HD 135344B':{'log_Mdot': -8.0,  'NAI': 0.405, 'NIR_excess': 27.2, 'NIR_excess_err': 3.1,  'NIR_ulim': False},
    'HD 143006': {'log_Mdot': -8.1,  'NAI': 0.215, 'NIR_excess': 21.3, 'NIR_excess_err': 1.4,  'NIR_ulim': False},
    'HD 34282':  {'log_Mdot': -7.7,  'NAI': 0.114, 'NIR_excess': 9.2,  'NIR_excess_err': 1.0,  'NIR_ulim': False,
                  'alpha_S': 3.86,   'alpha_S_err_low': 2.14, 'alpha_S_err_high': 4.79},
    'J1604':     {'log_Mdot': -10.5, 'NAI': 0.059, 'NIR_excess': 17.5, 'NIR_excess_err': 3.6,  'NIR_ulim': False},
    'J1615':     {'log_Mdot': -8.5,  'NAI': 0.038, 'NIR_excess': 0.9,  'NIR_ulim': True,
                  'alpha_S': 1.20,   'alpha_S_err_low': 0.66, 'alpha_S_err_high': 1.48},
    'J1842':     {'log_Mdot': -8.8,  'NAI': 0.074, 'NIR_excess': 12.3, 'NIR_excess_err': 1.1,  'NIR_ulim': False,
                  'alpha_S': 0.403,  'alpha_S_err_low': 0.223, 'alpha_S_err_high': 0.499},
    'J1852':     {'log_Mdot': -8.7,  'NAI': 0.024, 'NIR_excess': 1.1,  'NIR_ulim': True,
                  'alpha_S': 0.362,  'alpha_S_err_low': 0.201, 'alpha_S_err_high': 0.449},
    'LkCa15':    {'log_Mdot': -8.4,  'NAI': 0.053, 'NIR_excess': 13.4, 'NIR_excess_err': 1.0,  'NIR_ulim': False,
                  'alpha_S': 0.174,  'alpha_S_err_low': 0.097, 'alpha_S_err_high': 0.216},
    'MWC 758':   {'log_Mdot': -7.15, 'NAI': 0.429, 'NIR_excess': 27.5, 'NIR_excess_err': 2.9,  'NIR_ulim': False},
    'PDS 66':    {'log_Mdot': -9.9,  'NAI': 0.014, 'NIR_excess': 7.3,  'NIR_excess_err': 1.4,  'NIR_ulim': False,
                  'alpha_S': 0.0806, 'alpha_S_err_low': 0.0446, 'alpha_S_err_high': 0.0999},
    'SY Cha':    {'log_Mdot': -9.2,  'NAI': 0.075, 'NIR_excess': 7.6,  'NIR_excess_err': 1.1,  'NIR_ulim': False,
                  'alpha_S': 0.0166, 'alpha_S_err_low': 0.0092, 'alpha_S_err_high': 0.0205},
    'V4046 Sgr': {'log_Mdot': -9.3,  'NAI': 0.030, 'NIR_excess': 0.9,  'NIR_ulim': True,
                  'alpha_S': 0.111,  'alpha_S_err_low': 0.061, 'alpha_S_err_high': 0.138}
}



double_bell_labels = [
    disc_name_map[disc] for disc in [
        'aatau_dbell', 'hd34282_dbell', 'dmtau_dbell',
        'j1615_dbell', 'j1842_dbell', 'lkca15_dbell',
        'sycha_dbell'
    ]
]

def format_pi(x, _):
	tick_locs = {
		-1 * np.pi: r"$-\pi$",
		-0.5 * np.pi: r"$-\pi/2$",
		0.0: r"$0$",
		0.5 * np.pi: r"$\pi/2$",
		1.0 * np.pi: r"$\pi$"
	}
	# Allow small numerical tolerance
	for val, label in tick_locs.items():
		if np.isclose(x, val, atol=0.01):
			return label
	return ""

# Convert to cartesian grid
def plot_velocity_map(dv_grid, radii, phi_highres, meta_params, fname):

	print('plotting velocity map')
	print(radii.shape, phi_highres.shape, dv_grid.shape)
	R, Phi = np.meshgrid(radii, phi_highres, indexing="ij")
	X = R * np.cos(Phi)
	Y = R * np.sin(Phi)
	print(radii)

	radii_au = radii
	unit_label = "au"

	"""# Convert units if dist_pc is defined
	if meta_params.get("dist_pc") is not None:
		scale_au = meta_params["dist_pc"]
		X = X * scale_au
		Y = Y * scale_au
		radii_au = radii * scale_au
		unit_label = "AU"
	else:
		radii_au = radii
		unit_label = "native units"
	"""
	dvmax = 0.05
	dv = dvmax / 20.0

	dv_levels = np.arange(-dvmax, dvmax+dv, dv)

	fig, ax = plt.subplots(figsize=(8, 8))
	# Symmetric diverging colormap centered at 0
	vmax = np.nanmax(np.abs(dv_grid))
	pcm = ax.contourf(
		X,
		Y,
		dv_grid,
		cmap="RdBu_r",
		levels=dv_levels
	)
	cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
	cbar.set_label(r"$\delta v_{\rm los}$ [km s$^{-1}$]", fontsize=12)

	# Concentric circles every 20 AU (or appropriate spacing)
	Rmax = np.max(radii_au)
	dr = 20 if Rmax < 200 else 50
	for r in np.arange(dr, Rmax + dr, dr):
		circle = plt.Circle(
			(0, 0),
			r,
			color="k",
			ls=":",
			lw=0.5,
			fill=False,
			zorder=10,
		)
		ax.add_patch(circle)
		ax.text(
			r / np.sqrt(2),
			r / np.sqrt(2),
			f"{int(r)} {unit_label}",
			fontsize=8,
			ha="left",
			va="bottom",
			color="k",
		)

	# Radial lines every 45 deg
	for angle in np.deg2rad(np.arange(0, 360, 45)):
		ax.plot(
			[0, Rmax * np.cos(angle)],
			[0, Rmax * np.sin(angle)],
			color="k",
			ls=":",
			lw=0.5,
			zorder=10,
		)

	# Remove all axes, ticks, and spines
	ax.set_aspect("equal")
	ax.set_xlim(-Rmax, Rmax)
	ax.set_ylim(-Rmax, Rmax)
	ax.set_xticks([])
	ax.set_yticks([])
	for spine in ax.spines.values():
		spine.set_visible(False)
	ax.set_xlabel("")
	ax.set_ylabel("")

	ax.set_title(fname)
	plt.tight_layout()
	plt.show()

def grid_up(fname_full, phi_highres = np.linspace(-np.pi, np.pi, 257), **kwargs):
	trunc_rout = kwargs.get('trunc_rout', np.inf)
	hreset = kwargs.get('hreset', False)
	dist_pc = kwargs.get('dist_pc', None)
	beam_fwhm_arcsec = kwargs.get('beam_fwhm_arcsec', None)
	mstar_norm = kwargs.get('mstar_norm', None)
	channel_spacing = kwargs.get('channel_spacing', None)
	incl = kwargs.get('incl', None)
	rot_sign = kwargs.get('rot_sign', None)
	fname = os.path.splitext(fname_full)[0]
	plot = kwargs.get('plot', False)

	if not os.path.isfile(f"{fname}_meta_params.npy"):
		hreset =True

	if not os.path.isfile(f"{fname}_gridded.npy") or not os.path.isfile(f"{fname}_rgrid.npy") or not  os.path.isfile(f"{fname}_phigrid.npy") or hreset:
		radii, dv_grid = mu.interpolate_to_uniform_azimuth(fname_full, phi_highres= phi_highres, R_trunc=trunc_rout)
		np.save(f"{fname}_gridded.npy", dv_grid)
		np.save(f"{fname}_rgrid.npy", radii)
		np.save(f"{fname}_phigrid.npy", phi_highres)
	else:
		radii = np.load(f"{fname}_rgrid.npy")
		phi_highres =np.load(f"{fname}_phigrid.npy")
		dv_grid = np.load(f"{fname}_gridded.npy")

	recalc_IC = False
	if not os.path.isfile(f"{fname}_meta_params.npy") or hreset:
		meta_params = mu.write_meta(radii, dv_grid,  trunc_rout=trunc_rout, dist_pc=dist_pc, beam_fwhm_arcsec=beam_fwhm_arcsec, mstar_norm=mstar_norm, channel_spacing=channel_spacing, incl=incl, rot_sign=rot_sign)
		np.save(f"{fname}_meta_params.npy", meta_params)
		recalc_IC = True
	else:
		meta_params = np.load(f"{fname}_meta_params.npy", allow_pickle=True).item()

		# --- Check and update channel spacing if needed ---
		if meta_params.get('channel_spacing', None) != channel_spacing:
			print(f"Channel spacing mismatch: {meta_params.get('channel_spacing')} vs {channel_spacing}. Updating and marking for rescore...")
			meta_params['channel_spacing'] = channel_spacing
			recalc_IC = True  


	if not os.path.isfile(f"{fname}_rescaled.npy") or hreset:
		dv_rescaled = dv_grid #DO NOT RESCALE rescale_vfield(radii, dv_grid, meta_params)
		np.save(f"{fname}_rescaled.npy", dv_rescaled)
	else:
		dv_rescaled =  np.load(f"{fname}_rescaled.npy")

	r_scaled = radii/meta_params['R_out']

	if plot:
		plot_velocity_map(dv_grid, radii, phi_highres, meta_params, fname=fname)

	return radii, phi_highres, dv_grid, meta_params, recalc_IC, r_scaled, dv_rescaled




def save_gp_samples_to_csv(samples, r_eval, discname, quantity_name, output_dir='gp_samples'):
    """
    Save GP posterior samples to a CSV file.
    
    Parameters:
    - samples: ndarray of shape (n_radii, n_samples)
    - r_eval: radius grid of shape (n_radii,)
    - discname: name of the disc
    - quantity_name: string like 'delta_i', 'delta_pa', 'theta', 'psi', 'gamma'
    - output_dir: directory where CSVs are saved
    """
    clean_discname = re.sub(r'\W+', '_', discname.strip())
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(samples.T, columns=[f"{r:.3f}" for r in r_eval])
    filename = f"{output_dir}/{clean_discname}_{quantity_name}_gp_samples.csv"
    df.to_csv(filename, index=False)

def fit_gp_from_warp_model(
    rgrid,
    phigrid,
    dv_grid,
    meta_params,
    nmax=1000,
    plot=True,
    noise=None,
    dv_max=2.0,
	discname='',
	rot_sign=1.0,
	clip=False
):
	"""
	Fit Gaussian Processes to delta_i(r) and delta_PA(r) by decomposing
	the residual velocity field into cos(phi) and sin(phi) components.
	"""

	incl = meta_params.get("incl", None)
	if incl is None:
		raise ValueError("Inclination ('incl') must be specified in meta_params.")
	

	beam_fwhm_arcsec = meta_params.get("beam_fwhm_arcsec", 0.15)
	distance_pc = meta_params.get("dist_pc", 100.0)
	beam_radius_au = beam_fwhm_arcsec * distance_pc



	if clip:
		iinc = rgrid<clip
		dv_grid = dv_grid[iinc]
		rgrid = rgrid[iinc]

	M_star = meta_params.get("mstar_norm", 1.0)
	G = 4 * np.pi**2
	v_kep = np.sqrt(G * M_star / rgrid) * 4.74


	cosI = np.cos(incl)
	sinI = np.sin(incl)

	cos_phi = np.cos(phigrid)
	sin_phi = np.sin(phigrid)

	A_list, B_list, A_err, B_err = [], [], [], []

	beam_frac=0.5
	r0 = 0.0

	for i, r in enumerate(rgrid):
		if r>=r0+beam_frac*beam_radius_au:
			dv = rot_sign * dv_grid[i]
			mask = np.abs(dv) < dv_max


			if np.sum(mask) < 5:
				A_list.append(np.nan)
				B_list.append(np.nan)
				A_err.append(np.nan)
				B_err.append(np.nan)
				continue

			nbeams = 2.*np.pi*r/beam_radius_au

			nskip =1
			if nbeams<int(np.sum(mask)):
				nskip = max(1, int(np.sum(mask)/nbeams))
			cos_phi_mask = cos_phi[mask]
			sin_phi_mask = sin_phi[mask]
			X = np.vstack([cos_phi_mask[::nskip], sin_phi_mask[::nskip]]).T
			y = dv[mask][::nskip]

			try:
				coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
				A, B = coeffs
				A_list.append(A)
				B_list.append(B)

				if len(residuals) > 0:
					sigma = np.sqrt(residuals[0] / len(y))
					cov = sigma**2 * np.linalg.pinv(X.T @ X)
					A_err.append(np.sqrt(cov[0, 0]))
					B_err.append(np.sqrt(cov[1, 1]))
				else:
					A_err.append(np.nan)
					B_err.append(np.nan)
			except Exception:
				A_list.append(np.nan)
				B_list.append(np.nan)
				A_err.append(np.nan)
				B_err.append(np.nan)

	A_arr, B_arr = np.array(A_list), np.array(B_list)
	A_err, B_err = np.array(A_err), np.array(B_err)


	delta_i_obs = A_arr / (v_kep * cosI)
	delta_pa_obs = B_arr / (v_kep * sinI)
	delta_i_err = np.absolute(A_err / (v_kep * cosI))
	delta_pa_err = np.absolute(B_err / (v_kep * np.absolute(sinI)))

	# Determine beam size in AU
	beam_fwhm_arcsec = meta_params.get("beam_fwhm_arcsec", 0.15)
	distance_pc = meta_params.get("dist_pc", 100.0)
	beam_radius_au = beam_fwhm_arcsec * distance_pc

	# Beam-based radial sampling
	r_min, r_max = np.min(rgrid), np.max(rgrid)
	r_fit = np.arange(r_min, r_max + beam_radius_au/2., beam_radius_au/2.)
	#r_fit = np.arange(r_min, r_max + beam_radius_au, beam_radius_au)

	# Interpolate delta_i, delta_pa, and their errors to r_fit
	interp_i = interp1d(rgrid, delta_i_obs, bounds_error=False, fill_value=(delta_i_obs[0], delta_i_obs[-1]))
	interp_pa = interp1d(rgrid, delta_pa_obs, bounds_error=False, fill_value=(delta_pa_obs[0], delta_pa_obs[-1]))
	interp_i_err = interp1d(rgrid, delta_i_err, bounds_error=False, fill_value=(delta_i_err[0], delta_i_err[-1]))
	interp_pa_err = interp1d(rgrid, delta_pa_err, bounds_error=False, fill_value=(delta_pa_err[0], delta_pa_err[-1]))

	i_fit = interp_i(r_fit)
	pa_fit = interp_pa(r_fit)
	i_err = interp_i_err(r_fit)
	pa_err = interp_pa_err(r_fit)

	# GP kernel (can be adjusted later if needed)
	kernel_i = C(1.0) * Matern(length_scale=beam_radius_au*2., nu=2.5) # + WhiteKernel(noise_level=i_err)
	kernel_pa = C(1.0) * Matern(length_scale=beam_radius_au*2., nu=2.5)# + WhiteKernel(noise_level=pa_err)

	# Convert errors to variances
	alpha_i = i_err**2
	alpha_pa = pa_err**2

	# Replace NaNs in variances with a large value to down-weight them
	alpha_i[np.isnan(alpha_i)] = 1e10
	alpha_pa[np.isnan(alpha_pa)] = 1e10

	# GP regressors
	gp_i = GaussianProcessRegressor(kernel=kernel_i, alpha=alpha_i, normalize_y=False, n_restarts_optimizer=10)
	gp_pa = GaussianProcessRegressor(kernel=kernel_pa, alpha=alpha_pa, normalize_y=False, n_restarts_optimizer=10)


	gp_i.fit(r_fit.reshape(-1, 1), i_fit)
	gp_pa.fit(r_fit.reshape(-1, 1), pa_fit)

	r_eval = np.linspace(np.min(rgrid), max(np.max(rgrid), np.max(r_fit)), 200)
	i_mean, i_std = gp_i.predict(r_eval.reshape(-1, 1), return_std=True)
	pa_mean, pa_std = gp_pa.predict(r_eval.reshape(-1, 1), return_std=True)


	# Number of posterior samples to draw
	n_post_samples = 200
	r_eval_reshaped = r_eval.reshape(-1, 1)

	i_samples = gp_i.sample_y(r_eval_reshaped, n_post_samples, random_state=42)
	pa_samples = gp_pa.sample_y(r_eval_reshaped, n_post_samples, random_state=42)

	delta_i_ranges = np.ptp(i_samples, axis=0)  # ptp = max - min along r_eval
	delta_pa_ranges = np.ptp(pa_samples, axis=0)

	# Convert to degrees and account for sin(incl) for PA
	i_range_mean = np.rad2deg(np.mean(delta_i_ranges))
	i_range_std = np.rad2deg(np.std(delta_i_ranges))

	pa_range_mean = np.rad2deg(np.mean(delta_pa_ranges)) #* np.sin(incl)
	pa_range_std = np.rad2deg(np.std(delta_pa_ranges)) #* np.sin(incl)

	with open("warp_profile.txt", "w") as f:
		for ir in range(len(r_fit)):
			f.write(f"{r_fit[ir]} {i_fit[ir]} {pa_fit[ir]}\n")

	# Compute ranges
	i_range = (np.min(i_fit), np.max(i_fit))
	pa_range = (np.min(pa_fit), np.max(pa_fit))


	# Compute psi(R) and |Theta(R)| from GP posterior samples
	theta_samples = np.sqrt(i_samples**2 + (np.sin(incl) * pa_samples)**2)
	ln_r_eval = np.log(r_eval)
	di_dlnr_samples = np.gradient(i_samples, ln_r_eval, axis=0)
	dpa_dlnr_samples = np.gradient(pa_samples, ln_r_eval, axis=0)
	psi_samples = np.sqrt(di_dlnr_samples**2 + (np.sin(incl) * dpa_dlnr_samples)**2)

	# Compute ranges over radius for each sample
	#theta_ranges = np.ptp(theta_samples, axis=0)   # shape (n_samples,)
	theta_ranges = np.amax(theta_samples, axis=0)  # shape (n_samples,)
	psi_ranges = np.ptp(psi_samples, axis=0)

	# Get stats
	delta_psi_mean = np.mean(psi_ranges)
	delta_psi_std = np.std(psi_ranges)
	delta_theta_mean =np.mean(theta_ranges)
	delta_theta_std = np.std(theta_ranges)


	theta_mean = np.mean(theta_samples, axis=1)
	theta_std = np.std(theta_samples, axis=1)
	psi_mean = np.mean(psi_samples, axis=1)
	psi_std = np.std(psi_samples, axis=1)
	psi_std_log = np.std(np.log10(psi_samples), axis=1)
	theta_std_log = np.std(np.log10(theta_samples), axis=1)
	log_psi_samples = np.log10(psi_samples)  # shape: (n_radius, n_samples)
	psi_ravg_samples = np.trapz(log_psi_samples, r_eval[:, None], axis=0) / (r_eval[-1] - r_eval[0])
	psi_ravg = np.mean(psi_ravg_samples)
	psi_ravg_std = np.std(psi_ravg_samples)
	log_theta_samples = np.log10(theta_samples)
	theta_ravg_samples = np.trapz(log_theta_samples, r_eval[:, None], axis=0) / (r_eval[-1] - r_eval[0])
	theta_ravg = np.mean(theta_ravg_samples)
	theta_ravg_std = np.std(theta_ravg_samples)

		# --- Compute twist angle gamma(R) from delta_i and delta_PA ---
	# gamma = arctan2(delta_PA * sin(i_0), delta_i)
	gamma_mean = np.arctan2( i_mean, -pa_mean * np.sin(incl))
	gamma_std = np.std(np.arctan2(i_samples, -pa_samples * np.sin(incl)), axis=1)  # shape: (r_eval,)

	# Store gamma samples for optional use
	gamma_samples = np.arctan2(i_samples, -pa_samples * np.sin(incl))

	save_gp_samples_to_csv(i_samples, r_eval, discname, "delta_i")
	save_gp_samples_to_csv(pa_samples, r_eval, discname, "delta_pa")
	save_gp_samples_to_csv(theta_samples, r_eval, discname, "beta")
	save_gp_samples_to_csv(psi_samples, r_eval, discname, "psi")
	save_gp_samples_to_csv(gamma_samples, r_eval, discname, "gamma")



	return {
		"r_eval": r_eval,
		"delta_i_mean": i_mean,
		"delta_i_std": i_std,
		"delta_pa_mean": pa_mean,
		"delta_pa_std": pa_std,
		"gp_delta_i": gp_i,
		"gp_delta_pa": gp_pa,
		"r_fit": r_fit,
		"delta_i_fit": i_fit,
		"delta_pa_fit": pa_fit,
		"delta_i_err": i_err,
		"delta_pa_err": pa_err,
		"pa_range": pa_range,
		"i_range": i_range,
		"pa_range_err": pa_range_std,
		"i_range_err": i_range_std,
		"pa_range_mean": pa_range_mean,
		"i_range_mean": i_range_mean,
		"delta_psi": delta_psi_mean,
		"delta_psi_err": delta_psi_std,
		"delta_theta": delta_theta_mean,
		"delta_theta_err": delta_theta_std,
		"psi_ravg": psi_ravg,
		"psi_ravg_std": psi_ravg_std,
		"theta_ravg": theta_ravg,
		"theta_ravg_std": theta_ravg_std,
		"gamma_mean": gamma_mean,
		"gamma_std": gamma_std
	}



import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import gridspec
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates

def plot_combined_velocity_and_profiles(
	rgrid,
	phigrid,
	dv_grid,
	fit_result,
	meta_params,
	Nxy=400,
	Rmax=None,
	model_incl=None,
	n_post_samples=200,
	cmap="RdBu_r",
	discname="",
	rot_sign=1.0,
	savepath=None,
	clip=False,
	molecule='$^{12}$CO',
	plot_phys=True,
        show_residual=False,
	vmax_dict=vmax_dict_12co,
	**kwargs
):
	incl = model_incl if model_incl is not None else meta_params["incl"]
	M_star = meta_params.get("mstar_norm", 1.0)
	G = 4 * np.pi**2
	v_kep = np.sqrt(G * M_star / rgrid) * 4.74

	cos_phi = np.cos(phigrid)[None, :]
	sin_phi = np.sin(phigrid)[None, :]

	delta_i = interp1d(fit_result["r_eval"], fit_result["delta_i_mean"], bounds_error=False, fill_value=0.0)(rgrid)
	delta_pa = interp1d(fit_result["r_eval"], fit_result["delta_pa_mean"], bounds_error=False, fill_value=0.0)(rgrid)

	model_grid = rot_sign * (v_kep[:, None] * (
		delta_i[:, None] * np.cos(incl) * cos_phi +
		delta_pa[:, None] * np.sin(incl) * sin_phi
	))

	Rmax = Rmax or np.max(rgrid)
	x = np.linspace(-Rmax, Rmax, Nxy)
	y = np.linspace(-Rmax, Rmax, Nxy)
	X, Y = np.meshgrid(x, y)
	R = np.hypot(X, Y)
	PHI = np.arctan2(Y, X)

	r_idx = np.interp(R.flatten(), rgrid, np.arange(len(rgrid)))
	phi_idx = np.interp(PHI.flatten(), phigrid, np.arange(len(phigrid)))
	coords = np.vstack([r_idx, phi_idx])

	dv_obs_xy = map_coordinates(dv_grid, coords, order=1, mode='nearest').reshape(Nxy, Nxy)
	dv_model_xy = map_coordinates(model_grid, coords, order=1, mode='nearest').reshape(Nxy, Nxy)

	np.save(f"{discname}_dv_model.npy", dv_model_xy)
	np.save(f"{discname}_dv_obs.npy", dv_obs_xy)
	np.save(f"{discname}_xygrid.npy", np.array([x, y]))


	mask = R > np.max(rgrid)
	if clip:
		mask = mask|(R>clip)

	dv_obs_xy[mask] = np.nan
	dv_model_xy[mask] = np.nan


	dv_resid_xy = dv_obs_xy - dv_model_xy

	if discname in vmax_dict:
		dv_lim = vmax_dict[discname]
	else:
		#print(discname, [key for key in vmax_dict])
		dv_lim = 0.05
	dv = dv_lim / 10.0  # 10 levels
	levels = np.arange(-dv_lim, dv_lim + dv, dv)
	levels_resid = np.arange(-dv_lim, dv_lim + dv, dv)
	#levels_resid = np.arange(-dv_lim/2., dv_lim/2. + dv/2., dv/2.)

	# Prepare GP sampling
	r_eval = fit_result["r_eval"]
	i_mean = fit_result["delta_i_mean"]
	i_std = fit_result["delta_i_std"]
	pa_mean = fit_result["delta_pa_mean"]
	pa_std = fit_result["delta_pa_std"]
	gp_i = fit_result["gp_delta_i"]
	gp_pa = fit_result["gp_delta_pa"]

	r_eval_reshaped = r_eval.reshape(-1, 1)
	i_samples = gp_i.sample_y(r_eval_reshaped, n_post_samples, random_state=42)
	pa_samples = gp_pa.sample_y(r_eval_reshaped, n_post_samples, random_state=42)

	# Compute psi(R) and Theta(R) for each posterior sample
	theta_samples = np.sqrt(i_samples**2 + (np.sin(incl) * pa_samples)**2)  # shape (n_r, n_samples)

	# Use gradient in ln(R) space
	ln_r_eval = np.log(r_eval)
	di_dlnr_samples = np.gradient(i_samples, ln_r_eval, axis=0)
	dpa_dlnr_samples = np.gradient(pa_samples, ln_r_eval, axis=0)
	psi_samples = np.sqrt(di_dlnr_samples**2 + (np.sin(incl) * dpa_dlnr_samples)**2)  # shape (n_r, n_samples)

	# Compute means and standard deviations for error bars
	theta_mean = np.mean(theta_samples, axis=1)
	theta_std = np.std(theta_samples, axis=1)
	psi_mean = np.mean(psi_samples, axis=1)
	psi_std = np.std(psi_samples, axis=1)


	# === Compute gamma(R) from delta_i and delta_PA (twist angle) ===
	# gamma = arctan2(delta_PA * sin(i), delta_i)
	gamma_samples = np.arctan2(i_samples, -pa_samples * np.sin(incl))+ np.pi  # Wrap to [0, pi] for consistency
	#gamma_samples = np.arctan2(pa_samples * np.sin(incl), i_samples)+ np.pi  # Wrap to [0, pi] for consistency
	
	# Wrap gamma into [-pi/2, pi/2]
	gamma_samples_wrapped = (gamma_samples + 2 * np.pi) % (2 * np.pi)  -np.pi# Now in [0, 2π)
	#gamma_samples_wrapped = (gamma_samples) % np.pi - np.pi/2.  # Wrap to [-pi/2, pi/2]
	gamma_mean = np.median(gamma_samples, axis=1) # Wrap to [-pi/2, pi/2]

	gamma_mean_wrapped = (gamma_mean) - np.pi   # Wrap to [-pi/2, pi/2])
	#gamma_mean_wrapped = (gamma_mean) % np.pi - np.pi/2.   # Wrap to [-pi/2, pi/2])

	# Start plotting
	if show_residual:
		# Always use 5 columns if residual is shown
		width_ratios = [1, 1, 1, 0.05]  # colorbar | observed | model | residual | colorbar
		ncols = 4
		n_vel_cols = slice(1, 4)  # observed, model, residual
	else:
		width_ratios = [1, 1, 0.05]  # observed | model | colorbar
		ncols = 3
		n_vel_cols = slice(0, 2)

	# Number of rows
	base_rows = 1  # always one row for velocity
	phys_rows = 5 if plot_phys else 2  # 2 GP panels always, +2 more if plot_phys
	nrows = base_rows + phys_rows

	# Height ratios: 1 for top row, 0.3 for each phys panel
	height_ratios = [1.0] + [0.3] * phys_rows

	# Final GridSpec
	fig = plt.figure(figsize=(15 if show_residual else 12, 3 + phys_rows * 2))
	spec = gridspec.GridSpec(
		nrows=nrows,
		ncols=ncols,
		height_ratios=height_ratios,
		width_ratios=width_ratios,
		hspace=0.25,
		wspace=0.1 if show_residual else 0.05
	)

	if show_residual:
		#cax_obs = fig.add_subplot(spec[0, 0])
		#ax_obs = fig.add_subplot(spec[0, 1])
		#ax_mod = fig.add_subplot(spec[0, 2], sharex=ax_obs, sharey=ax_obs)
		#ax_resid = fig.add_subplot(spec[0, 3], sharex=ax_obs, sharey=ax_obs)
		cax_obs = fig.add_subplot(spec[0, 3])
		ax_obs = fig.add_subplot(spec[0, 0])
		ax_mod = fig.add_subplot(spec[0, 1], sharex=ax_obs, sharey=ax_obs)
		ax_resid = fig.add_subplot(spec[0, 2], sharex=ax_obs, sharey=ax_obs)
		#cax_resid = fig.add_subplot(spec[0, 4])
	else:
		ax_obs = fig.add_subplot(spec[0, 0])
		ax_mod = fig.add_subplot(spec[0, 1], sharex=ax_obs, sharey=ax_obs)
		cax_obs = fig.add_subplot(spec[0, 2])


	im0 = ax_obs.contourf(X, Y, dv_obs_xy, cmap=cmap, levels=levels)
	ax_obs.set_title("Observed $\\delta v_{\\rm los}$")
	im1 = ax_mod.contourf(X, Y, dv_model_xy, cmap=cmap, levels=levels)
	ax_mod.set_title("Modeled $\\delta v_{\\rm los}$")

	for ax in [ax_obs, ax_mod]:
		ax.set_xlabel("x [AU]")
		ax.set_ylabel("y [AU]")
		ax.set_aspect('equal')
		beam_fwhm_arcsec = meta_params.get("beam_fwhm_arcsec", 0.15)
		distance_pc = meta_params.get("dist_pc", 100.0)
		beam_radius_au = 2.0 * beam_fwhm_arcsec * distance_pc
		beam_circle = Circle((0, 0), radius=beam_radius_au,
								facecolor='gray', edgecolor='red', alpha=0.95,
								zorder=10)
		ax.add_patch(beam_circle)
	
	# Remove y-axis label and ticks from right-hand velocity map
	ax_mod.set_ylabel("")
	ax_mod.set_yticklabels([])
	ax_mod.tick_params(axis='y', left=False)

	for ax in [ax_obs, ax_mod]:
		ax.set_aspect('equal')
		ax.set_xlabel("")
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_ylabel("")
		for spine in ax.spines.values():
			spine.set_visible(False)
	# Choose round radius intervals

	if clip:
		Rmax = min(clip,Rmax)
	if Rmax > 200:
		dr = 50
	else:
		dr = 20

	radii_au = np.arange(dr, Rmax + 1, dr)  # e.g., [50, 100, 150, ...]

	angles_deg = np.arange(0, 360, 45)

	for ax in [ax_obs, ax_mod]:
		# Concentric circles
		for r in radii_au:
			circle = plt.Circle((0, 0), r, color='k', ls=':', lw=0.5, fill=False, zorder=20)
			ax.add_patch(circle)
			ax.text(r/np.sqrt(2), r/np.sqrt(2), f"{int(r)} au", fontsize=8, ha='left', va='bottom', rotation=0, color='k', zorder=21)

		
		# Radial lines
		for angle in angles_deg:
			angle_rad = np.deg2rad(angle)
			ax.plot([0, Rmax * np.cos(angle_rad)], [0, Rmax * np.sin(angle_rad)],
					color='k', ls=':', lw=0.5, zorder=20)
		
		if clip:
			ax.set_xlim([-clip, clip])
			ax.set_ylim([-clip, clip])

	cbar = fig.colorbar(im1, cax=cax_obs)
	cbar.set_label("$\delta v_{\\rm los}$ [km s$^{-1}$]")
	"""if show_residual:
		cbar.ax.yaxis.set_label_position('left')
		cbar.ax.yaxis.tick_left()"""

		# Beam size indicator in observed panel
	beam_size = meta_params['beam_fwhm_arcsec'] * meta_params['dist_pc']
	beam_x = -0.85 * Rmax
	beam_y = -0.85 * Rmax
	beam_indicator = Circle((beam_x, beam_y), radius=beam_size,
							facecolor='gray', edgecolor='black', lw=1,
							alpha=0.7, zorder=11)
	ax_obs.add_patch(beam_indicator)

	iphys = 0
	if show_residual:

			im_resid = ax_resid.contourf(X, Y, dv_resid_xy, cmap=cmap, levels=levels_resid)
			ax_resid.set_title("Residual (Obs - Model)")

			ax_resid.set_aspect('equal')
			ax_resid.set_xlabel("")
			ax_resid.set_xticks([])
			ax_resid.set_yticks([])
			ax_resid.set_ylabel("")
			for spine in ax_resid.spines.values():
				spine.set_visible(False)
			beam_circle = Circle((0, 0), radius=beam_radius_au,
									facecolor='gray', edgecolor='red', alpha=0.95,
									zorder=10)
			ax_resid.add_patch(beam_circle)

			# Concentric circles
			for r in radii_au:
				circle = plt.Circle((0, 0), r, color='k', ls=':', lw=0.5, fill=False, zorder=20)
				ax_resid.add_patch(circle)
				ax_resid.text(r/np.sqrt(2), r/np.sqrt(2), f"{int(r)} au", fontsize=8, ha='left', va='bottom', rotation=0, color='k', zorder=21)

			
			# Radial lines
			for angle in angles_deg:
				angle_rad = np.deg2rad(angle)
				ax_resid.plot([0, Rmax * np.cos(angle_rad)], [0, Rmax * np.sin(angle_rad)],
						color='k', ls=':', lw=0.5, zorder=20)
			
			if clip:
				ax_resid.set_xlim([-clip, clip])
				ax_resid.set_ylim([-clip, clip])

			#cbar2 = fig.colorbar(im_resid, cax=cax_resid)
			#cbar2.set_label("Obs - Model [km s$^{-1}$]")


	# Bottom panels: GP profiles
	ax2 = fig.add_subplot(spec[1+iphys, :])
	ax3 = fig.add_subplot(spec[2+iphys, :], sharex=ax2)

	# Delta i panel
	for i_s in i_samples.T:
		ax2.plot(r_eval, np.rad2deg(i_s), color='orange', alpha=0.05, zorder=1)
	ax2.errorbar(fit_result["r_fit"], np.rad2deg(fit_result["delta_i_fit"]),
					yerr=np.rad2deg(fit_result["delta_i_err"]), fmt='o', alpha=0.5, zorder=2)
	ax2.plot(r_eval, np.rad2deg(i_mean), color='k', lw=1.5, zorder=3)
	ax2.set_ylabel(r"$\delta i$ [deg]")

	

	# Delta PA panel
	for pa_s in pa_samples.T:
		ax3.plot(r_eval,  np.rad2deg(pa_s), color='orange', alpha=0.05, zorder=1)
	ax3.errorbar(fit_result["r_fit"], np.rad2deg(fit_result["delta_pa_fit"]),
					yerr=np.rad2deg(fit_result["delta_pa_err"]), fmt='o', alpha=0.5, zorder=2)
	ax3.plot(r_eval, np.rad2deg(pa_mean), color='k', lw=1.5, zorder=3)
	ax3.set_ylabel(r"$\delta$PA [deg]")
	if plot_phys:


		def plot_wrapped_angle(x, y, ax=None, threshold=np.pi, **kwargs):
			"""
			Plot wrapped angles, breaking the line at large jumps (like from pi to -pi).
			
			Parameters:
			- x, y: arrays of the same shape
			- ax: matplotlib axis to plot on
			- threshold: discontinuity threshold (default: pi)
			- **kwargs: passed to ax.plot()
			"""
			if ax is None:
				ax = plt.gca()

			x = np.asarray(x)
			y = np.asarray(y)

			# Compute the difference between successive points
			dy = np.diff(y)
			breaks = np.where(np.abs(dy) > threshold)[0]
			# Start and end of segments
			segments = np.split(np.arange(len(x)), breaks + 1)

			# Plot each segment individually
			for seg in segments:
				ax.plot(x[seg], y[seg], **kwargs)

		ax4 = fig.add_subplot(spec[3+iphys, :], sharex=ax2)
		ax4_gamma = fig.add_subplot(spec[4+iphys, :], sharex=ax2)
		ax5 = fig.add_subplot(spec[5+iphys, :], sharex=ax2)

		# Panel for psi(R)
		for theta_s in theta_samples.T:
			ax4.plot(r_eval, np.rad2deg(theta_s), color='orange', alpha=0.05, zorder=1)
		ax4.plot(r_eval, np.rad2deg(theta_mean), color='k', lw=1.5, zorder=3)
		ax4.set_ylabel("$\\beta$ [deg]")

		for gamma_s in gamma_samples_wrapped.T:
			plot_wrapped_angle(r_eval, np.rad2deg(gamma_s), ax=ax4_gamma, threshold=70.0, color='orange', alpha=0.05, zorder=1)
		plot_wrapped_angle(r_eval, np.rad2deg(gamma_mean_wrapped), ax=ax4_gamma, threshold=70.0, color='k', lw=1.5, zorder=3)

		
		#ax4_gamma.set_ylim(-np.pi/2, np.pi/2)
		#ax4_gamma.set_ylim(-90.,90.)
		#ax4_gamma.set_yticks([-90, -45, 0, 45, 90])
		ax4_gamma.set_yticks([-180, -90, 0, 90, 180])

		# Custom ticks at multiples of pi/4
		#locs = np.arange(-np.pi/2, np.pi/2 + 0.01, np.pi/4)
		#labels = [r"$-\frac{\pi}{2}$", r"$-\frac{\pi}{4}$", "0", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$"]

		#ax4_gamma.set_yticks(locs)
		#ax4_gamma.set_yticklabels(labels)
		ax4_gamma.set_ylabel(r"$\gamma$ [deg]")
		ax4_gamma.axhline(0.0, color='gray', linestyle='--', lw=0.8)

		# Panel for |Theta|
		for psi_s in psi_samples.T:
			ax5.plot(r_eval, (psi_s), color='orange', alpha=0.05, zorder=1)
		ax5.plot(r_eval, (psi_mean), color='k', lw=1.5, zorder=3)
		ax5.set_yscale('log')
		ax5.set_ylim(1e-3, 1e0)  # Adjusted for log scale
		from matplotlib.ticker import LogLocator, NullFormatter
		ax5.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
		ax5.yaxis.set_minor_formatter(NullFormatter())
		ax5.set_ylabel("$\\psi$ [rad]")
		ax5.set_xlabel("Radius: $R$ [au]")
	else:		
		ax3.set_xlabel("Radius [AU]")


	# Compute max beta for each posterior sample
	theta_max_per_sample = np.max(np.rad2deg(theta_samples), axis=0)  # shape (n_samples,)

	# Compute the median and ±1σ confidence intervals
	theta_max_median = np.median(theta_max_per_sample)
	theta_max_lower = np.percentile(theta_max_per_sample, 16)
	theta_max_upper = np.percentile(theta_max_per_sample, 84)

	# Print result
	print(f"Max β (median ±1σ): {theta_max_median:.2f}° (+{theta_max_upper - theta_max_median:.2f}, -{theta_max_median - theta_max_lower:.2f})")

	fig.suptitle(f"{discname} {molecule}", fontsize=16)

	if savepath:
		import os
		os.makedirs(os.path.dirname(savepath), exist_ok=True)
		fig.savefig(savepath, dpi=150)
		print(f"Saved combined figure to {savepath}")

	plt.tight_layout()
	plt.subplots_adjust(top=0.93, hspace=0.3)
	plfname = f'combined_model_plot_{discname.replace(" ", "")}'
	if molecule!='$^{12}$CO':
		plfname += molecule.translate(str.maketrans('', '', '${}^'))
	plt.savefig(plfname+'.pdf', bbox_inches='tight', format='pdf')
	plt.show()


def load_or_grid(fname, **kwargs):
	folder = kwargs.get("folder", os.getcwd())

	print(kwargs)
	base_name = os.path.basename(fname)
	#result_name = f"{os.path.splitext(base_name)[0]}_fitwaves_result.pkl"
	#result_path = os.path.join(folder, result_name)

	# Handle current working directory case (e.g., mwc758.txt lives in .)
	in_same_directory = (folder == os.getcwd())

	r = None
	p = None
	dv = None
	if in_same_directory:
		r,p,dv, meta_params, _, _, _ = grid_up(base_name,  **kwargs)
	else:
		cwd = os.getcwd()
		os.chdir(folder)
		try:
			r, p, dv, meta_params, _, _, _ =  grid_up(base_name,  **kwargs)
		finally:
			os.chdir(cwd)
	
	return r, p, dv, meta_params

def load_or_run_fit(fname, **kwargs):
	"""
	Handles loading or generating fit results for a velocity residual file.

	If the file is in a subdirectory (as with sorted axisymmetric/testgrid targets),
	we change into that folder. If not, we run everything in-place.
	"""
	folder = kwargs.get("folder", os.getcwd())

	base_name = os.path.basename(fname)
	result_name = f"{os.path.splitext(base_name)[0]}_fitwaves_result.pkl"
	result_path = os.path.join(folder, result_name)

	# Handle current working directory case (e.g., mwc758.txt lives in .)
	in_same_directory = (folder == os.getcwd())

	if os.path.exists(result_path) and not kwargs.get("reset", False):
		with open(result_path, "rb") as f:
			result = pickle.load(f)
			print(f"[loaded] {result_path}")
		if kwargs.get("plot", True):
			if in_same_directory:
				result = fit_waves(base_name,  **kwargs)
			else:
				cwd = os.getcwd()
				os.chdir(folder)
				try:
					result = fit_waves(base_name, **kwargs)
				finally:
					os.chdir(cwd)
	else:
		if in_same_directory:
			result = fit_waves(base_name, **kwargs)
			with open(result_name, "wb") as f:
				pickle.dump(result, f)
				print(f"[saved] {result_name}")
		else:
			cwd = os.getcwd()
			os.chdir(folder)
			print(cwd, os.getcwd())
			try:
				result = fit_waves(base_name, **kwargs)
				with open(result_name, "wb") as f:
					pickle.dump(result, f)
					print(f"[saved] {result_name}")
			finally:
				os.chdir(cwd)

	return result

def load_or_run_gp_fit(fname, **kwargs):
	"""
	Handles loading or generating GP-fitted velocity components.
	Automatically handles directory switching and caching.

	Returns:
		Dictionary of interpolated GP functions:
		{
			'vphi_fn': ..., 'vr_fn': ..., 'vz_fn': ...,
			'vphi_std_fn': ..., 'vr_std_fn': ..., 'vz_std_fn': ...
		}
	"""
	folder = kwargs.get("folder", os.getcwd())

	base_name = os.path.basename(fname)
	result_name = f"{os.path.splitext(base_name)[0]}_gpfit_result.pkl"
	result_path = os.path.join(folder, result_name)

	# Check if we're already in the folder
	in_same_directory = (folder == os.getcwd())

	if os.path.exists(result_path) and not kwargs.get("reset", False):
		with open(result_path, "rb") as f:
			gp_funcs = pickle.load(f)
			print(f"[loaded] {result_path}")
	else:
		# Change directory if needed
		if in_same_directory:
			gp_funcs = fit_gp_velocity_components(base_name, **kwargs)
			with open(result_name, "wb") as f:
				pickle.dump(gp_funcs, f)
				print(f"[saved] {result_name}")
		else:
			cwd = os.getcwd()
			os.chdir(folder)
			try:
				gp_funcs = fit_gp_velocity_components(base_name, **kwargs)
				with open(result_name, "wb") as f:
					pickle.dump(gp_funcs, f)
					print(f"[saved] {result_name}")
			finally:
				os.chdir(cwd)

	return gp_funcs

def load_or_run_warp_fit(fname, **kwargs):
	"""
	Handles loading or fitting the warp GP model for inclination perturbations.

	Parameters
	----------
	fname : str
		Filename (e.g., 'azimuthal_velocity_residuals_cqtau.txt')
	kwargs : dict
		Options for the fit (e.g., incl, reset, folder, etc.)

	Returns
	-------
	fit_result : dict
		Output of fit_gp_from_warp_model
	"""
	folder = kwargs.get("folder", os.getcwd())
	base_name = os.path.basename(fname)
	label = os.path.splitext(base_name)[0]
	rot_sign = kwargs.get("rot_sign", 1.0)
	discname = kwargs.get("discname", "")
	result_name = f"{label}_warpfit_result.pkl"
	result_path = os.path.join(folder, result_name)
	molecule = kwargs.get('molecule', '$^{12}$CO')

	if molecule == '$^{12}$CO':
		vmax_dict = vmax_dict_12co
	elif molecule == '$^{13}$CO':
		vmax_dict = vmax_dict_13co

	print(kwargs)
	print(kwargs['plot'], kwargs.get("plot", False))

	reset = kwargs.get("reset", False)
	hreset = kwargs.get("hreset", False)

	clip  = kwargs.get('clip', False)

	fit_result = None
	if os.path.exists(result_path) and not (reset or hreset):
		with open(result_path, "rb") as f:
			fit_result = pickle.load(f)
			print(f"[loaded warp GP fit] {result_path}")
			if not kwargs.get("plot", False):
				return fit_result

	# Otherwise, run fit
	cwd = os.getcwd()
	try:
		if cwd != folder:
			os.chdir(folder)

		rgrid = np.load(f"{label}_rgrid.npy")
		phigrid = np.load(f"{label}_phigrid.npy")
		dv_grid = np.load(f"{label}_gridded.npy")
		meta = np.load(f"{label}_meta_params.npy", allow_pickle=True).item()
		if "incl" in kwargs:
			meta["incl"] = kwargs["incl"]

		fit_result = fit_gp_from_warp_model(
			rgrid=rgrid,
			phigrid=phigrid,
			dv_grid=dv_grid,
			meta_params=meta,
			discname=discname,
			plot=kwargs.get("plot", False),
			dv_max=kwargs.get("dv_max", 0.5),
			rot_sign =rot_sign, 
			clip=clip
		)

		with open(result_name, "wb") as f:
			pickle.dump(fit_result, f)
			print(f"[saved warp GP fit] {result_name}")

		print('SHOW RESIDUAL:', kwargs.get("show_residuals", False))


		if kwargs.get("plot", False):
			plot_combined_velocity_and_profiles(
				rgrid=rgrid,
				phigrid=phigrid,
				dv_grid=dv_grid,
				fit_result=fit_result,
				meta_params=meta,
				#plot=kwargs.get("plot", False),
				#title_prefix=f"{label} - ",
				discname = discname,
				rot_sign = rot_sign,
				clip=clip, 
				molecule=molecule,
				show_residual=kwargs.get("show_residuals", False),
				vmax_dict=vmax_dict
			)
	finally:
		if cwd != folder:
			os.chdir(cwd)

	return fit_result

def plot_gp_three_panel_comparison(dv_obs, R, PHI, vphi_fn, vr_fn, vz_fn, incl_rad):
	"""
	Plot comparison of observed LOS velocity, GP model, and residuals.

	Parameters:
		dv_obs: 2D array of observed LOS velocity (shape [n_r, n_phi])
		R: 2D array of radius grid
		PHI: 2D array of azimuth grid (in radians)
		vphi_fn, vr_fn, vz_fn: functions of radius r returning GP mean values
		incl_rad: inclination in radians
	"""
	# Evaluate model LOS on the grid
	r_flat = R.flatten()
	phi_flat = PHI.flatten()

	vphi_model = vphi_fn(r_flat)
	vr_model   = vr_fn(r_flat)
	vz_model   = vz_fn(r_flat)

	vlos_model_flat = (
		vphi_model * np.cos(phi_flat) * np.sin(incl_rad) +
		-vr_model   * np.sin(phi_flat) * np.sin(incl_rad) +
		-vz_model   * np.cos(incl_rad)
	)
	dv_fit = vlos_model_flat.reshape(R.shape)
	dv_residual = dv_obs - dv_fit

	# === PLOTTING ===
	fig = plt.figure(figsize=(15, 6))
	gs = gridspec.GridSpec(2, 3, height_ratios=[0.05, 1], hspace=0.3)

	ax0 = plt.subplot(gs[1, 0])
	ax1 = plt.subplot(gs[1, 1])
	ax2 = plt.subplot(gs[1, 2])
	cb0_ax = plt.subplot(gs[0, 0])
	cb1_ax = plt.subplot(gs[0, 1])
	cb2_ax = plt.subplot(gs[0, 2])

	dvmax = int(np.percentile(np.abs(dv_obs.flatten()), 90.0) * 10.0 + 0.99) / 10.0
	levels = np.arange(-dvmax, dvmax + dvmax / 10.0, dvmax / 10.0)

	# Panel 1: Observed LOS
	im0 = ax0.contourf(np.rad2deg(PHI), R, dv_obs, levels=levels, cmap='RdBu_r', extend='both')
	cb0 = plt.colorbar(im0, cax=cb0_ax, orientation='horizontal')
	cb0.set_label('$\delta v_{\\rm los,\\ obs}$ [km s$^{-1}$]')
	ax0.set_xlabel('Azimuth (deg)')
	ax0.set_ylabel('Radius [AU]')

	# Panel 2: GP Model LOS
	im1 = ax1.contourf(np.rad2deg(PHI), R, dv_fit, levels=levels, cmap='RdBu_r', extend='both')
	cb1 = plt.colorbar(im1, cax=cb1_ax, orientation='horizontal')
	cb1.set_label('$v_{\\rm los,\\ model}$ [km s$^{-1}$]')
	ax1.set_xlabel('Azimuth (deg)')

	# Panel 3: Residuals
	im2 = ax2.contourf(np.rad2deg(PHI), R, dv_residual, levels=levels, cmap='RdBu_r', extend='both')
	cb2 = plt.colorbar(im2, cax=cb2_ax, orientation='horizontal')
	cb2.set_label('Residual [km s$^{-1}$]')
	ax2.set_xlabel('Azimuth (deg)')

	plt.tight_layout()
	plt.show()

def plot_gp_components(posterior_dict, r_eval=None, component_labels=None):
		
	if r_eval is None:
		rsort= np.argsort(posterior_dict['r_eval'])
		r_eval = posterior_dict['r_eval'][rsort]
	else:
		rsort= np.argsort(r_eval)


	samples = [
		posterior_dict['vphi_samples'][:,rsort],
		posterior_dict['vr_samples'][:,rsort],
		posterior_dict['vz_samples'][:, rsort]
	]

	if component_labels is None:
		component_labels = ('$v_\\phi$', '$v_r$', '$v_z$')

	fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

	for a, sample_array, label in zip(ax, samples, component_labels):
		a.plot(r_eval, sample_array.T, color='gray', alpha=0.05)
		mean = sample_array.mean(axis=0)
		a.plot(r_eval, mean, color='black', label=label)
		a.set_ylabel(f"{label} [km/s]")
		a.axhline(0, color='k', linestyle='--', linewidth=0.5)
		a.legend()

	ax[-1].set_xlabel("Radius $r$ [AU]")
	fig.suptitle("GP Velocity Component Samples", fontsize=14)
	plt.tight_layout()
	plt.show()

def permutation_corr(x, y, n_permutations=10000, seed=42):
    rng = np.random.default_rng(seed)
    r_obs = np.corrcoef(x, y)[0, 1]
    permuted_rs = np.array([
        np.corrcoef(x, rng.permutation(y))[0, 1]
        for _ in range(n_permutations)
    ])
    p_val = np.mean(np.abs(permuted_rs) >= np.abs(r_obs))
    return r_obs, p_val

	
# Utility function to plot correlations and annotate
def annotate_correlations(ax, x, y, color='black', label=None, ha=None, va=None):
		rho_s, pval_s = spearmanr(x, y)
		tau_k, pval_k = kendalltau(x, y)
		r_perm, p_perm = permutation_corr(x, y)

		if label is None:
			text = (
				f"Sp.  $\\rho/p: {rho_s:.2f}/{pval_s:.3f}$\n"
				f"K.   $\\tau/p: {tau_k:.2f}/{pval_k:.3f}$\n"
				f"P.   $r/p: {r_perm:.2f}/{p_perm:.3f}$"
			)
		else:
			text = (
				f"{label}:\n"
				f"Sp.  $\\rho/p: {rho_s:.2f}/{pval_s:.3f}$\n"
				f"K.   $\\tau/p: {tau_k:.2f}/{pval_k:.3f}$\n"
				f"P.   $r/p: {r_perm:.2f}/{p_perm:.3f}$"
			)

		
		if ha is None and va is None:
			if label == 'All':
				ax.text(0.98, 0.02, text,
						transform=ax.transAxes,
						ha='right', va='bottom',
						fontsize=9, color=color,
						bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
			else:  # 'No DBell'
				ax.text(0.02, 0.98, text,
						transform=ax.transAxes,
						ha='left', va='top',
						fontsize=9, color=color,
						bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
		else:
			if ha=='right':
				xc = 0.98
			elif ha=='left':
				xc = 0.02
			else:
				xc = 0.5
				
			if va=='bottom':
				yc = 0.02
			elif va=='top':
				yc = 0.98
			else:
				yc = 0.5
			ax.text(xc, yc, text,
						transform=ax.transAxes,
						ha=ha, va=va,
						fontsize=9, color=color,
						bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
		

def compare_warp_to_curone(results_dict, disc_name_map,stellar_masses=None, inc_dbell=False, xaxis_log_psi=True):
	"""
	Compare GP warp fit results with axisymmetry index and accretion rate from Curone+ (2025).

	Parameters
	----------
	results_dict : dict
		Dictionary of warp GP fit results with 'i_range' and 'pa_range'.
	disc_name_map : dict
		Mapping from short labels to full disc names.
	comparison_data : dict, optional
		Dictionary containing 'axisym_index' and 'log_Mdot' keyed by long disc name.
	stellar_masses : dict, optional
		Dictionary of stellar masses in solar units keyed by long disc name.
	"""

	print(f"{'Disc':<10} {'i range (deg)':<20} {'PA range (deg)':<20} {'Warp amplitude (deg)':<20} {'A-index':<10} {'log(Mdot)':<10}")
	print("="*70)

	# Prepare data for plotting
	delta_i_norm = []
	i0s = []
	delta_i_vals=[]
	log_Mdots = []
	non_axisym = []
	log_Mdot_norm = []
	delta_i_mags = []
	delta_i_mags_errs = []
	labels = []
	nir_excess=[]
	nir_ulim=[]
	warp_ampl =[]
	warp_ampl_err =[]
	log_psi_vals = []
	log_psi_errs = []
	nir_excess_errs = []


	for label, result in results_dict.items():
		long_name = disc_name_map.get(label, label)
		comp = comparison_data.get(long_name)
		M_star = result['stellar_mass']
		i0  =result['incl']
		if not comp or not M_star:
			print(f'No M_star for {long_name}, masking for plot...')
			#continue

		i_min, i_max = np.rad2deg(result['i_range'])
		pa_min, pa_max = np.rad2deg(result['pa_range'])



		delta_i_mean= result['i_range_mean']
		delta_pa_mean = result['pa_range_mean']

		delta_i_mean_err= result['i_range_err']
		delta_pa_mean_err = result['pa_range_err']

		dtheta = np.rad2deg(result['delta_theta'])
		dtheta_err = np.rad2deg(result['delta_theta_err'])
		warp_amplitude = dtheta # np.sqrt(delta_i_mean**2+(np.sin(i0)**2) * delta_pa_mean**2)
		warp_amplitude_err = dtheta_err #np.sqrt(delta_i_mean**2*delta_i_mean_err**2+(np.sin(i0)**4) * delta_pa_mean**2* delta_pa_mean_err**2)/warp_amplitude

		warp_ampl.append(warp_amplitude)
		warp_ampl_err.append(warp_amplitude_err)

		# Optional psi-based x-axis
		log_psi_val = result.get('psi_ravg', np.nan)
		log_psi_err = result.get('psi_ravg_std', np.nan)

		log_psi_vals.append(log_psi_val)
		log_psi_errs.append(log_psi_err)

		delta_i = i_max - i_min
		delta_pa = pa_max - pa_min

		print(f"{long_name:<10} {i_min:.2f}–{i_max:.2f}     {pa_min:.2f}–{pa_max:.2f}     {warp_amplitude:.2f}$\pm${warp_amplitude_err:.2f}    {comp['NAI']:<10.2f} {comp['log_Mdot']:<10.2f}")

		meta = result.get("meta_params", {})
		rout = meta.get("R_out", np.nan)
		M_star = result.get("stellar_mass", np.nan)

		#if not np.isfinite(rout) or not np.isfinite(M_star):
		#	continue

		delta_i_rout = delta_i / rout

		delta_i_vals.append(delta_i)
		delta_i_norm.append(delta_i_rout)
		log_Mdots.append(comp['log_Mdot'])
		non_axisym.append(comp['NAI'])
		log_Mdot_norm.append(comp['log_Mdot'] - 2 * np.log10(M_star))
		labels.append(long_name)

		i0s.append(i0)
		dl_mag = np.sqrt((np.sin(i0)*delta_pa)**2 + delta_i**2)

		delta_i_mags.append(dtheta)
		delta_i_mags_errs.append(dtheta_err)

		nir_excess.append(comp['NIR_excess'])
		nir_ulim.append(comp['NIR_ulim'])
		
		nir_excess_err = comp.get('NIR_excess_err', np.nan)
		nir_excess_errs.append(nir_excess_err if not comp['NIR_ulim'] else np.nan)

	


	mask_incs = np.ones(len(results_dict), dtype=bool)
	ilab = 0
	for label, result in results_dict.items():
		if not(inc_dbell or (not 'dbell' in label.split('_'))):
			mask_incs[ilab]=False
		ilab+=1
	
	# ======= LaTeX Table Output ========
	print("\nLaTeX Table:")
	print("\\begin{tabular}{lcccccccc}")
	print("\\toprule")
	print("Disc & $M_\\star$ [$M_\\odot$] & $i_0$ [$^\circ$] & $R_\\mathrm{out}$ [au] & $\delta i_{\mathrm{min/max}}$ [$^\circ$] & $\delta$PA$_{\mathrm{min/max}}$  [$^\circ$] & $\\beta_\mathrm{max}$ [$^\circ$] & NAI & $\\log \\dot{M}_\mathrm{acc}$ [$M_\odot$~yr$^{-1}$] & $\langle \log \psi \\rangle_R$ DB?\\\\")
	print("\\midrule")

	i =0
	for label, result in results_dict.items():
		i_min, i_max = np.rad2deg(result['i_range'])
		pa_min, pa_max = np.rad2deg(result['pa_range'])
		M_star = result["stellar_mass"]
		meta = results_dict[label].get("meta_params", {})
		rout = meta.get("R_out", np.nan)
		incl = result['incl']
		if type(incl)==tuple:
			print('Warning; not sure why inclination is a tuple....')
			incl = incl[0]
		warp_amplitude= (warp_ampl[i])
		warp_amplitude_err = (warp_ampl_err[i])
		log_psi = log_psi_vals[i]
		log_psi_err = log_psi_errs[i]
		discname = disc_name_map[label]
		comp = comparison_data.get(discname, None)
		if comp is None:
			continue
		
		db_bool = 'N' if mask_incs[i] else 'Y'
		print(f"{discname} & {M_star:.2f} & {np.rad2deg(incl):.1f} & {rout:.1f} & {i_min:.1f}--{i_max:.1f} & {pa_min:.1f}--{pa_max:.1f} & {warp_amplitude:.2f}$\pm${warp_amplitude_err:.2f} &{log_psi:.2f}$\pm${log_psi_err:.2f} & {comp['NAI']:.2f} & {comp['log_Mdot']:.2f} & {db_bool:s}\\\\")
		i+=1
	print("\\bottomrule")
	print("\\end{tabular}")

	print(delta_i_mags)

	non_axisym = np.array(non_axisym)
	log_Mdots = np.array(log_Mdots)
	log_Mdot_norm = np.array(log_Mdot_norm)
	nir_excess = np.array(nir_excess)
	log_psi_vals = np.array(log_psi_vals)
	log_psi_errs = np.array(log_psi_errs)
	nir_excess_errs = np.array(nir_excess_errs)


	if xaxis_log_psi:
		xlabel = r"Averaged warp amplitude: $\langle \log\,\psi \rangle_{R}$"
		xvals = log_psi_vals
		xerrs = log_psi_errs
	else:
		xlabel = r"Tilt amplitude: ${\beta}_\mathrm{max}$  [deg]"
		xvals = np.array(delta_i_mags)
		xerrs = np.array(delta_i_mags_errs)

	delta_i_plot = np.array(delta_i_mags) #delta_i_vals 
	delta_i_plot_errs = np.array(delta_i_mags_errs) #delta_i_vals 

	NIR_exc=True
	if NIR_exc:
		fig, axes = plt.subplots(4, 1, figsize=(5, 9), sharex=True)

		# Panel 1: NAI
		axes[0].scatter(xvals[mask_incs], non_axisym[mask_incs], color='tab:red')
		axes[0].errorbar(xvals[mask_incs], non_axisym[mask_incs],
                 xerr=xerrs[mask_incs], fmt='o',
                 color='tab:red', ecolor='gray',elinewidth=1, capsize=0)

		for i, txt in enumerate(labels):
			if mask_incs[i]:
				axes[0].annotate(txt, (xvals[i], non_axisym[i]), ha='center', fontsize=9)
		for i, label in enumerate(labels):
			if mask_incs[i]:
				if label in double_bell_labels:
					axes[0].scatter(xvals[i], non_axisym[i], facecolors='none', edgecolors='red', s=100, linewidths=1.5)
		axes[0].set_ylabel("NAI")
		if inc_dbell:
			annotate_correlations(axes[0], xvals, non_axisym, label='All', ha='right', va='bottom')
			mask = [lbl not in double_bell_labels for lbl in labels]
			annotate_correlations(axes[0], np.array(xvals)[mask], np.array(non_axisym)[mask], color='red', label='No DBell', ha='right', va='center')
		else:
			annotate_correlations(axes[0], xvals, non_axisym, label=None)


		# Panel 2: log Mdot
		axes[1].scatter(xvals[mask_incs], log_Mdots[mask_incs], color='tab:blue')
		axes[1].errorbar(xvals[mask_incs], log_Mdots[mask_incs],
                 xerr=xerrs[mask_incs], fmt='o',
                 color='tab:blue', ecolor='gray',elinewidth=1, capsize=0)

		for i, txt in enumerate(labels):
			if mask_incs[i]:
				axes[1].annotate(txt, (xvals[i], log_Mdots[i]), ha='center', fontsize=9)
		axes[1].set_ylabel("$\log \dot{M}_\\mathrm{acc}$  [$M_\\odot$ yr$^{-1}$]")
		if inc_dbell:
			annotate_correlations(axes[1], xvals[mask_incs], log_Mdots[mask_incs], label='All', ha='right', va='bottom')
			annotate_correlations(axes[1], np.array(xvals)[mask&mask_incs], np.array(log_Mdots)[mask&mask_incs], color='red', label='No DBell', ha='right', va='center')
		else:
			annotate_correlations(axes[1], xvals[mask_incs], log_Mdots[mask_incs], label=None, ha='right', va='bottom')

		for i, label in enumerate(labels):
			if mask_incs[i]:
				if label in double_bell_labels:
					axes[1].scatter(xvals[i], log_Mdots[i], facecolors='none', edgecolors='red', s=100, linewidths=1.5)

		# Panel 3: log Mdot / M*^2
		axes[2].scatter(xvals[mask_incs], log_Mdot_norm[mask_incs], color='tab:green')
		axes[2].errorbar(xvals[mask_incs], log_Mdot_norm[mask_incs],
                 xerr=xerrs[mask_incs], fmt='o',
                 color='tab:green', ecolor='gray',  elinewidth=1, capsize=0)

		for i, txt in enumerate(labels):
			if mask_incs[i]:
				axes[2].annotate(txt, (xvals[i], log_Mdot_norm[i]), ha='center', fontsize=9)
		axes[2].set_ylabel("$\log(\\dot{{M}}_\mathrm{acc} / M_*^2)$ [$M_\\odot^{-1}$ yr$^{-1}$]")
		if inc_dbell:
			annotate_correlations(axes[2], xvals[mask_incs], log_Mdot_norm[mask_incs], label='All', ha='right', va='bottom')
			annotate_correlations(axes[2], np.array(xvals)[mask&mask_incs], np.array(log_Mdot_norm)[mask&mask_incs], color='red', label='No DBell', ha='right', va='center')
		else:
			annotate_correlations(axes[2], xvals[mask_incs], log_Mdot_norm[mask_incs], label=None, ha='right', va='bottom')
			
		for i, label in enumerate(labels):
			if mask_incs[i]:
				if label in double_bell_labels:
					axes[2].scatter(xvals[i], log_Mdot_norm[i], facecolors='none', edgecolors='red', s=100, linewidths=1.5)

		# Panel 4: NIR Excess
		"""for i, label in enumerate(labels):
			if mask_incs[i]:
				marker = 'v' if nir_ulim[i] else 'o'
				axes[3].errorbar(xvals[mask_incs], nir_excess[mask_incs],
						xerr=xerrs[mask_incs], fmt='None',
						color='tab:purple', ecolor='gray', alpha=0.3, elinewidth=1, capsize=0, marker='None')
				axes[3].scatter(xvals[i], nir_excess[i], marker=marker, color='tab:purple')
				axes[3].annotate(label, (xvals[i], nir_excess[i]), fontsize=8, ha='center', xytext=(0, 5), textcoords='offset points')"""
		
		for i, label in enumerate(labels):
			if mask_incs[i]:
				if nir_ulim[i]:  # Plot as downward triangle

					axes[3].errorbar(xvals[i], nir_excess[i],xerr=xerrs[i],marker='v',
									fmt='v', color='tab:purple', ecolor='gray',  elinewidth=1,  zorder=2)
				else:  # Include error bar
					axes[3].errorbar(xvals[i], nir_excess[i], yerr=nir_excess_errs[i],xerr=xerrs[i],
									fmt='o', color='tab:purple', ecolor='gray',  elinewidth=1,  zorder=2)
					
				axes[3].annotate(label, (xvals[i], nir_excess[i]), fontsize=8, ha='center', xytext=(0, 5), textcoords='offset points')


		for i, label in enumerate(labels):
			if mask_incs[i]:
				if label in double_bell_labels:
					axes[3].scatter(xvals[i],  nir_excess[i], facecolors='none', edgecolors='red', s=100, linewidths=1.5)

		if inc_dbell:
			annotate_correlations(axes[3], xvals[mask_incs], nir_excess[mask_incs], label='All', ha='right', va='bottom')
			annotate_correlations(axes[3], np.array(xvals)[mask&mask_incs], np.array(nir_excess)[mask&mask_incs], color='red', label='No DBell', ha='right', va='bottom')
		else:
			annotate_correlations(axes[3], xvals[mask_incs], nir_excess[mask_incs], label=None, ha='right', va='bottom')

		axes[3].set_ylabel("NIR Excess [percent]")

		axes[3].set_xlabel(xlabel)
		if not xaxis_log_psi:
			axes[3].set_xscale('log')
		else:
			axes[3].set_xscale('linear')


		plt.tight_layout()
		plt.savefig('correlations_figure_with_nir.pdf', bbox_inches='tight', format='pdf')
		plt.show()

	else:

		# Create 3-panel plot
		fig, axes = plt.subplots(3, 1, figsize=(5, 8), sharex=True)

		# Panel 1: NAI
		axes[0].scatter(xvals[mask_incs], non_axisym[mask_incs], color='tab:red')
		axes[0].errorbar(xvals[mask_incs], non_axisym[mask_incs],
                 xerr=xerrs[mask_incs], fmt='o',
                 color='tab:red', ecolor='gray', elinewidth=1, capsize=0)
		for i, txt in enumerate(labels):
			if mask_incs[i]:
				axes[0].annotate(txt, (xvals[i], non_axisym[i]), ha='center', fontsize=9)
		for i, label in enumerate(labels):
			print(label, mask_incs[i])
			if mask_incs[i]:
				if label in double_bell_labels:
					axes[0].scatter(xvals[i], non_axisym[i], facecolors='none', edgecolors='red', s=100, linewidths=1.5)
		axes[0].set_ylabel("NAI")
		if inc_dbell:
			annotate_correlations(axes[0], xvals, non_axisym, label='All', ha='right', va='top')
			mask = [lbl not in double_bell_labels for lbl in labels]
			annotate_correlations(axes[0], np.array(xvals)[mask], np.array(non_axisym)[mask], color='red', label='No DBell', ha='right', va='center')
		else:
			annotate_correlations(axes[0], xvals, non_axisym, label=None)


		# Panel 2: log Mdot
		axes[1].scatter(xvals[mask_incs], log_Mdots[mask_incs], color='tab:blue')
		axes[1].errorbar(xvals[mask_incs], log_Mdots[mask_incs],
                 xerr=xerrs[mask_incs], fmt='o',
                 color='tab:blue', ecolor='gray',  elinewidth=1, capsize=0)
		for i, txt in enumerate(labels):
			if mask_incs[i]:
				axes[1].annotate(txt, (xvals[i], log_Mdots[i]), ha='center', fontsize=9)
		axes[1].set_ylabel("$\log \dot{M}_\\mathrm{acc}$  [$M_\\odot$ yr$^{-1}$]")
		if inc_dbell:
			annotate_correlations(axes[1], xvals[mask_incs], log_Mdots[mask_incs], label='All')
			annotate_correlations(axes[1], np.array(xvals)[mask&mask_incs], np.array(log_Mdots)[mask&mask_incs], color='red', label='No DBell')
		else:
			annotate_correlations(axes[1], xvals[mask_incs], log_Mdots[mask_incs], label=None)

		for i, label in enumerate(labels):
			if mask_incs[i]:
				if label in double_bell_labels:
					axes[1].scatter(xvals[i], log_Mdots[i], facecolors='none', edgecolors='red', s=100, linewidths=1.5)

		# Panel 3: log Mdot / M*^2
		axes[2].scatter(xvals[mask_incs], log_Mdot_norm[mask_incs], color='tab:green')
		axes[2].errorbar(xvals[mask_incs], log_Mdot_norm[mask_incs],
                 xerr=xerrs[mask_incs], fmt='o',
                 color='tab:green', ecolor='gray', elinewidth=1, capsize=0)

		for i, txt in enumerate(labels):
			if mask_incs[i]:
				axes[2].annotate(txt, (xvals[i], log_Mdot_norm[i]), ha='center', fontsize=9)
		axes[2].set_ylabel("$\log(\\dot{{M}}_\mathrm{acc} / M_*^2)$ [$M_\\odot^{-1}$ yr$^{-1}$]")
		axes[2].set_xlabel("Tilt amplitude: ${\\beta}_\mathrm{max}$  [deg]")
		if inc_dbell:
			annotate_correlations(axes[2], xvals[mask_incs], log_Mdot_norm[mask_incs], label='All')
			annotate_correlations(axes[2], np.array(xvals)[mask&mask_incs], np.array(log_Mdot_norm)[mask&mask_incs], color='red', label='No DBell')
		else:
			annotate_correlations(axes[2], xvals[mask_incs], log_Mdot_norm[mask_incs], label=None)
			
		for i, label in enumerate(labels):
			if mask_incs[i]:
				if label in double_bell_labels:
					axes[2].scatter(xvals[i], log_Mdot_norm[i], facecolors='none', edgecolors='red', s=100, linewidths=1.5)
		
		if not xaxis_log_psi:
			axes[2].set_xscale('log')
		else:
			axes[2].set_xscale('linear')
		axes[2].set_xlabel(xlabel)
		plt.tight_layout()
		print('Saving ', os.getcwd())
		plt.savefig('correlations_figure.pdf', bbox_inches='tight', format='pdf')
		plt.show()


def plot_inclination_vs_pa(results_dict, logspace=False):
	labels, delta_i, delta_pa, incls = [], [], [], []
	delta_i_errs, delta_pa_errs= [], []

	for label, result in results_dict.items():
		long_name = disc_name_map.get(label, label)
		i_range = result['i_range_mean']
		pa_range = result['pa_range_mean']
		i_range_err = result['i_range_err']
		pa_range_err = result['pa_range_err']
		incl = result.get('incl', None)
		if incl is None:
			continue


		labels.append(long_name)
		delta_i.append(i_range)
		delta_pa.append(pa_range)
		delta_i_errs.append(i_range_err)
		delta_pa_errs.append(pa_range_err)
		print(pa_range_err)
		incls.append(np.rad2deg(incl))

	delta_pa= np.abs(delta_pa)
	delta_pa_errs= np.absolute(delta_pa_errs)
	#delta_pa *= np.sin(np.deg2rad(np.absolute(incls))) # Convert to PA amplitude
	#delta_pa_errs *= np.sin(np.deg2rad(np.absolute(incls)))  # Convert to PA amplitude
	delta_i= np.abs(delta_i)
	delta_i_errs= np.absolute(delta_i_errs)

	fig, ax = plt.subplots(figsize=(6, 5))
	norm_center=0.0
	# Compute x and y values
	xvals = np.abs(delta_pa)
	xerrs = np.abs(delta_pa_errs)
	yvals = np.abs(delta_i)
	yerrs = delta_i_errs



	# Add faint error bars first (under the scatter points)
	ax.errorbar(
		xvals, yvals, 
		xerr=xerrs, yerr=yerrs,
		fmt='none', ecolor='gray', elinewidth=1, alpha=0.8, zorder=1
	)
	sc = ax.scatter(delta_pa, np.absolute(delta_i), c=np.absolute(incls), 
        cmap='cool', 
		edgecolors='k', 
		s=60,
		norm=plt.Normalize(vmin= 0.0, vmax=norm_center + 60)
	)
	texts =  []
	for i, label in enumerate(labels):
		print('Label:', label, delta_pa[i], delta_i[i])
		if not logspace:
			if label=='J1615':
				txt = ax.annotate(label, (np.absolute(delta_pa[i])-0.1, np.absolute(delta_i[i])-0.3), ha='center', fontsize=9)
			elif label=='PDS 66':
				txt = ax.annotate(label, (np.absolute(delta_pa[i])-0.1, np.absolute(delta_i[i])-0.3), ha='center', fontsize=9)
			else:
				txt = ax.annotate(label, (np.absolute(delta_pa[i])-0.1, np.absolute(delta_i[i])+0.15), ha='center', fontsize=9)

		else:
			txt = ax.annotate(label, (np.absolute(delta_pa[i])*0.9, np.absolute(delta_i[i])*1.1), ha='center', fontsize=9)
		texts.append(txt)
	for i, label in enumerate(labels):
		if label in double_bell_labels:
			ax.scatter(delta_pa[i], delta_i[i], facecolors='none', edgecolors='red', s=160, linewidths=1.5)

	if not logspace:
		xsp = np.linspace(-1.0, 16.0)
		#plt.plot(xsp, xsp, color='k', linestyle='dashed', linewidth=1)
		#plt.plot(xsp, 0.2*xsp, color='k', linestyle='dotted', linewidth=1)
		#plt.plot(xsp, 5.0*xsp, color='k', linestyle='dotted', linewidth=1)
		ax.set_xlim([0.,11.0])
		ax.set_ylim([0.,16.0])

	else:
		ax.set_xscale('log')
		ax.set_yscale('log')
	xsp = np.linspace(-1.0, 6.0)
	#ax.plot(xsp,xsp/np.sin(np.deg2rad(21.0)), linestyle='dashed', color='k', linewidth=1)
	#ax.set_xlim([-0.4,4.0])

	if logspace:
		adjust_text(texts, ax=ax)
	ax.set_xlabel("Amplitude in PA: $\\Delta$PA [deg]")
	ax.set_ylabel("Amplitude in inclination: $\\Delta i$ [deg]")
	cbar = plt.colorbar(sc, ax=ax)
	cbar.set_label("Global inclination: $|i_0|$ [deg]")
	plt.tight_layout()
	plt.savefig("warp_vs_pa_incl_colored.pdf", bbox_inches='tight')
	plt.show()


def plot_inclination_vs_pa_compare(results_12co, results_13co, labels_12co='12CO', labels_13co='13CO'):
	def extract_data(results_dict):
		labels, delta_i, delta_pa, incls = [], [], [], []
		delta_i_errs, delta_pa_errs = [], []
		for label, result in results_dict.items():
			short_label = label.replace('_13co', '').replace('_dbell', '')
			i_range = result['i_range_mean']
			pa_range = result['pa_range_mean']
			i_range_err = result['i_range_err']
			pa_range_err = result['pa_range_err']
			incl = result.get('incl', None)
			if incl is None:
				continue
			labels.append(short_label)
			delta_i.append(np.abs(i_range))
			delta_pa.append(np.abs(pa_range))
			delta_i_errs.append(np.abs(i_range_err))
			delta_pa_errs.append(np.abs(pa_range_err))
			incls.append(np.rad2deg(incl))
		return labels, np.array(delta_i), np.array(delta_pa), np.array(delta_i_errs), np.array(delta_pa_errs), np.array(incls)

	# Extract data
	labels_12, delta_i_12, delta_pa_12, delta_i_errs_12, delta_pa_errs_12, incls_12 = extract_data(results_12co)
	labels_13, delta_i_13, delta_pa_13, delta_i_errs_13, delta_pa_errs_13, incls_13 = extract_data(results_13co)

	fig, ax = plt.subplots(figsize=(6, 5))
	norm_center = 0.0

	# Plot connecting lines for shared labels
	common_labels = set(labels_12) & set(labels_13)
	label_to_idx_12 = {label: i for i, label in enumerate(labels_12)}
	label_to_idx_13 = {label: i for i, label in enumerate(labels_13)}

	for label in common_labels:
		i12 = label_to_idx_12[label]
		i13 = label_to_idx_13[label]
		x1 = np.abs(np.sin(np.deg2rad(incls_12[i12]))) * delta_pa_12[i12]
		y1 = delta_i_12[i12]
		x2 = np.abs(np.sin(np.deg2rad(incls_13[i13]))) * delta_pa_13[i13]
		y2 = delta_i_13[i13]
		ax.plot([x1, x2], [y1, y2], color='gray', alpha=0.4, zorder=0, linewidth=1)

	# Plot 12CO
	xvals_12 = np.abs(np.sin(np.deg2rad(incls_12))) * delta_pa_12
	yvals_12 = delta_i_12
	sc12 = ax.scatter(xvals_12, yvals_12, c=np.abs(incls_12), cmap='cool',
					edgecolors='k', s=60, marker='o', label=labels_12co,
					norm=plt.Normalize(vmin=0.0, vmax=norm_center + 60))

	# Plot 13CO
	xvals_13 = np.abs(np.sin(np.deg2rad(incls_13))) * delta_pa_13
	yvals_13 = delta_i_13
	sc13 = ax.scatter(xvals_13, yvals_13, c=np.abs(incls_13), cmap='cool',
					edgecolors='k', s=60, marker='^', label=labels_13co,
					norm=plt.Normalize(vmin=0.0, vmax=norm_center + 60))

	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_xlabel("Amplitude in PA: $|\sin i_0 |\cdot \\Delta$PA [deg]")
	ax.set_ylabel("Amplitude in inclination: $\\Delta i$ [deg]")
	cbar = plt.colorbar(sc12, ax=ax)
	cbar.set_label("Global inclination: $|i_0|$ [deg]")
	ax.legend()
	plt.tight_layout()
	plt.savefig("warp_vs_pa_incl_12co_vs_13co.pdf", bbox_inches='tight')
	plt.show()

def plot_warp_amplitude_comparison(results_12co, results_13co):
	"""
	Plot comparison of warp amplitude sqrt(Δi^2 + sin^2(i0) ΔPA^2)
	between 12CO and 13CO with error bars and same-target connecting lines.
	"""

	import numpy as np
	import matplotlib.pyplot as plt

	def compute_amplitude(result):
		incl = result['incl']
		di = result['i_range_mean']
		dpa = result['pa_range_mean']
		di_err = result['i_range_err']
		dpa_err = result['pa_range_err']
		dtheta = np.rad2deg(result['delta_theta'])
		dtheta_err = np.rad2deg(result['delta_theta_err'])
		amp = np.sqrt(di**2 + (np.sin(incl)**2) * dpa**2)
		amp_err = np.sqrt((di * di_err)**2 + ((np.sin(incl)**2 * dpa * dpa_err)**2)) / amp
		return dtheta, dtheta_err

	# Find common base labels (strip suffixes)
	base_labels_12co = {label.replace('_dbell', '') for label in results_12co}
	base_labels_13co = {label.replace('_13co', '').replace('_dbell', '') for label in results_13co}
	shared_labels = sorted(base_labels_12co & base_labels_13co)

	x, xerr, y, yerr, colors, all_labels = [], [], [], [], [], []

	for base_label in shared_labels:
		# Match full label names from each dictionary
		label_12 = next((l for l in results_12co if base_label in l), None)
		label_13 = next((l for l in results_13co if base_label in l), None)

		if label_12 is None or label_13 is None:
			continue

		amp12, amp12_err = compute_amplitude(results_12co[label_12])
		amp13, amp13_err = compute_amplitude(results_13co[label_13])
		i0 = np.absolute(np.rad2deg(results_12co[label_12]['incl']))

		x.append(amp12)
		xerr.append(amp12_err)
		y.append(amp13)
		yerr.append(amp13_err)
		colors.append(i0)
		all_labels.append(base_label)

	fig, ax = plt.subplots(figsize=(7, 4))
	# Error bars
	ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='none', ecolor='gray', elinewidth=1, alpha=0.6, zorder=1)
	# Scatter points
	sc = ax.scatter(x, y, c=colors, cmap='cool', edgecolors='k', s=60, zorder=2,vmin =0.0, vmax=60.0)



	# Identity line
	lims = [min(x + y)*0.8, max(x + y)*1.2]
	ax.plot(lims, lims, 'k--', lw=1)
	ax.set_xlim(lims)
	ax.set_ylim(lims)

	ax.set_xlabel('Tilt amplitude $^{12}$CO: $\\beta_{\mathrm{max},12}$ [deg]')
	ax.set_ylabel('Tilt amplitude $^{13}$CO: $\\beta_{\mathrm{max},13}$ [deg]')
	ax.set_xscale('log')
	ax.set_yscale('log')

	texts = []
	# Annotate and circle DBell
	for i, label in enumerate(all_labels):
		texts.append(ax.annotate(base_disc_name_map[label], (x[i]*0.9, y[i]*1.1), fontsize=9, ha='center'))

		if label in dbell_12co:
			print(label, x[i], y[i])
			ax.scatter(x[i], y[i], facecolors='none', edgecolors='red', s=120, linewidths=1.5, zorder=1)
	adjust_text(texts)
	cbar = plt.colorbar(sc, ax=ax)
	cbar.set_label("Inclination: $|i_0|$ [deg]")

	plt.tight_layout()
	plt.savefig("warp_amplitude_12co_vs_13co.pdf", bbox_inches='tight')
	plt.show()

def plot_beam_amplitude_comparison(results_nominal, results_b0p30, double_bell_labels=None):
	import numpy as np
	import matplotlib.pyplot as plt

	if double_bell_labels is None:
		double_bell_labels = set()

	def compute_amplitude(result):
		incl = result['incl']
		di = result['i_range_mean']
		dpa = result['pa_range_mean']
		di_err = result['i_range_err']
		dpa_err = result['pa_range_err']

		dtheta = np.rad2deg(result['delta_theta'])
		dtheta_err = np.rad2deg(result['delta_theta_err'])
		amp = np.sqrt(di**2 + (np.sin(incl)**2) * dpa**2)
		amp_err = np.sqrt((di * di_err)**2 + ((np.sin(incl)**2 * dpa * dpa_err)**2)) / amp
		return dtheta, dtheta_err

	shared = sorted(set(label.replace('_b0p30', '') for label in results_b0p30) & set(results_nominal))

	x, xerr, y, yerr, colors, all_labels = [], [], [], [], [], []

	for base in shared:
		label_nom = base
		label_b03 = base + '_b0p30'

		amp_nom, err_nom = compute_amplitude(results_nominal[label_nom])
		amp_b03, err_b03 = compute_amplitude(results_b0p30[label_b03])
		i0 = np.absolute(np.rad2deg(results_nominal[label_nom]['incl']))

		x.append(amp_nom)
		xerr.append(err_nom)
		y.append(amp_b03)
		yerr.append(err_b03)
		colors.append(i0)
		all_labels.append(base)

	fig, ax = plt.subplots(figsize=(7, 4))
	ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='none', ecolor='gray', elinewidth=1, alpha=0.6, zorder=1)
	sc = ax.scatter(x, y, c=colors, cmap='cool', edgecolors='k', s=60, zorder=2, vmin=0., vmax=60.0)

	texts =[]
	for i, label in enumerate(all_labels):
		texts.append(ax.annotate(disc_name_map[label], (x[i], y[i]), fontsize=9, ha='center', xytext=(0, 5), textcoords='offset points'))
		if label in double_bell_labels:
			ax.scatter(x[i], y[i], facecolors='none', edgecolors='red', s=120, linewidths=1.5, zorder=3)

	lims = [min(x + y)*0.8, max(x + y)*1.2]
	ax.plot(lims, lims, 'k--', lw=1)
	#ax.set_xlim(lims)
	#ax.set_ylim(lims)
	#ax.set_xscale('log')
	#ax.set_yscale('log')
	ax.set_xlabel("Tilt amplitude (0.15''): $ \\beta_{\mathrm{max},0.15}$ [deg]")
	ax.set_ylabel("Tilt amplitude (0.30''): $ \\beta_{\mathrm{max},0.30}$ [deg]")

	cbar = plt.colorbar(sc, ax=ax)
	cbar.set_label("Inclination $|i_0|$ [deg]")
	plt.tight_layout()
	plt.savefig("warp_amplitude_beam_comparison.pdf", bbox_inches='tight')
	plt.show()


def plot_warp_vs_mdot_norm(results_dict, disc_name_map,  inc_dbell=True):
	"""
	Plot normalized mass accretion rate vs psi_ravg (log) for discs.

	Parameters
	----------
	results_dict : dict
		Dictionary of warp GP fit results including 'psi_ravg', 'psi_ravg_std', and 'stellar_mass'.
	disc_name_map : dict
		Mapping from short labels to full disc names.
	comparison_data : dict
		Dictionary with keys by long disc name, containing 'log_Mdot'.
	inc_dbell : bool, optional
		Whether to include double bell discs in the plot.
	"""

	labels, xvals, xerrs, yvals = [], [], [], []

	# Handle optional masking of DBell discs
	mask_incs = []
	double_bell_labels = [lbl for lbl in results_dict if 'dbell' in lbl.lower()]
	is_dbell = []

	for label, result in results_dict.items():
		long_name = disc_name_map.get(label, label)
		comp = comparison_data.get(long_name, None)

		if comp is None or 'psi_ravg' not in result or 'stellar_mass' not in result:
			continue

		if not inc_dbell and label in double_bell_labels:
			continue

		psi_ravg = result['psi_ravg']
		psi_ravg_std = result['psi_ravg_std']
		M_star = result['stellar_mass']
		log_mdot = comp.get('log_Mdot')

		if not np.isfinite(psi_ravg) or not np.isfinite(psi_ravg_std) or not np.isfinite(log_mdot) or not np.isfinite(M_star):
			continue

		labels.append(long_name)
		xvals.append(psi_ravg)
		xerrs.append(psi_ravg_std)
		yvals.append(log_mdot - 2 * np.log10(M_star))
		is_dbell.append(label in double_bell_labels)
	is_dbell = np.array(is_dbell)
	# Convert lists to numpy arrays
	xvals = np.array(xvals)
	xerrs = np.array(xerrs)
	yvals = np.array(yvals)
	if not inc_dbell:
		xvals = xvals[~is_dbell]
		xerrs = xerrs[~is_dbell]
		yvals = yvals[~is_dbell]
	fig, ax = plt.subplots(figsize=(5, 4))

	ax.errorbar(xvals, yvals, xerr=xerrs, fmt='o', color='tab:green', ecolor='gray', alpha=0.7, capsize=2)

	for i, label in enumerate(labels):
		ax.annotate(label, (xvals[i], yvals[i]), fontsize=8, ha='center', xytext=(0, 5), textcoords='offset points')
		if is_dbell[i]:
				ax.scatter(xvals[i], yvals[i], facecolors='none', edgecolors='red', s=120, linewidths=1.5, zorder=3)

	# Correlation annotations
	if not inc_dbell:
		annotate_correlations(ax, xvals, yvals)
	else:
		annotate_correlations(ax, xvals[~is_dbell], yvals[~is_dbell], label='No DBell', color='tab:red')
		annotate_correlations(ax, xvals, yvals, label='All', color='k')

	ax.set_xlabel(r'Averaged warp gradient: $\langle \log\,\psi \rangle_{R}$')
	ax.set_ylabel(r'$\log\,(\dot{M}_\mathrm{acc} / M_*^2)$ [$M_\odot^{-1}$ yr$^{-1}$]')

	plt.tight_layout()
	plt.savefig('warp_vs_mdot_norm.pdf', bbox_inches='tight')
	plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_warp_comparison(results_dict, rad_unit='au'):
	"""
	Plot delta_i_fit and delta_pa_fit from multiple warp results.

	Parameters
	----------
	results_dict : dict
		Dictionary containing warp fit results for multiple discs.
		Each value must be a dictionary with keys:
		- 'r_fit'
		- 'delta_i_fit'
		- 'delta_pa_fit'
		- 'delta_i_err'
		- 'delta_pa_err'
		- 'incl' (to scale PA)

	rad_unit : str
		Label for the radial unit. Default is 'au'.
	"""
	fig, axes = plt.subplots(2, 1, figsize=(7, 4), sharex=True)

	label2labels = {'mwc758': 'MWC 758', 'mwc758_wc': 'Warped coordinates'}

	for label_, result in results_dict.items():
		r = result['r_fit']
		delta_i = np.rad2deg(result['delta_i_fit'])
		delta_pa = np.rad2deg(result['delta_pa_fit']) * np.sin(result['incl'])
		err_i = np.rad2deg(result['delta_i_err'])
		err_pa = np.rad2deg(result['delta_pa_err']) * np.sin(result['incl'])

		if label_ in label2labels:
			label = label2labels[label_]
		else:
			label = label_
		

		axes[0].errorbar(
			r, delta_i, yerr=err_i,
			fmt='o', markersize=5, capsize=3,
			label=label, alpha=0.7
		)
		axes[1].errorbar(
			r, delta_pa, yerr=err_pa,
			fmt='o', markersize=5, capsize=3,
			label=label, alpha=0.7
		)

	axes[0].set_ylabel(r"$\delta i$ [deg]")
	axes[1].set_ylabel(r"$\delta$ PA [deg]")
	axes[1].set_xlabel(f"Radius [{rad_unit}]")

	for ax in axes:
		ax.legend(frameon=False, loc='best', fontsize='small')

	axes[1].set_xlim([39.0, 270.0])
	plt.tight_layout()
	plt.savefig('warp_coordinates.pdf', bbox_inches='tight', format='pdf')
	plt.show()



def plot_warp_vs_alphaS(results_dict, disc_name_map, inc_dbell=True):
	"""
	Plot log10(alpha_S) vs <log10(psi)> for discs.

	Parameters
	----------
	results_dict : dict
		Dictionary of warp GP fit results including 'psi_ravg', 'psi_ravg_std'.
	disc_name_map : dict
		Mapping from short labels to full disc names.
	inc_dbell : bool, optional
		Whether to include double bell discs in the plot.
	"""
	labels, xvals, xerrs, yvals, yerr_lows, yerr_highs = [], [], [], [], [], []
	is_dbell = []

	double_bell_labels = [lbl for lbl in results_dict if 'dbell' in lbl.lower()]

	for label, result in results_dict.items():
		long_name = disc_name_map.get(label, label)
		comp = comparison_data.get(long_name, None)

		if comp is None or 'psi_ravg' not in result:
			continue

		if 'alpha_S' not in comp or 'alpha_S_err_low' not in comp or 'alpha_S_err_high' not in comp:
			continue

		if not inc_dbell and label in double_bell_labels:
			continue

		psi_ravg = result['psi_ravg']
		psi_ravg_std = result['psi_ravg_std']
		alpha_S = comp.get('alpha_S')*1e-3
		err_low = comp.get('alpha_S_err_low')*1e-3
		err_high = comp.get('alpha_S_err_high')*1e-3

		if not np.all(np.isfinite([psi_ravg, psi_ravg_std, alpha_S, err_low, err_high])):
			continue

		labels.append(long_name)
		xvals.append(psi_ravg)
		xerrs.append(psi_ravg_std)
		yvals.append(np.log10(alpha_S))
		yerr_lows.append(np.log10(alpha_S) - np.log10(alpha_S - err_low))
		yerr_highs.append(np.log10(alpha_S + err_high) - np.log10(alpha_S))
		is_dbell.append(label in double_bell_labels)

	print(xvals)

	xvals = np.array(xvals)
	xerrs = np.array(xerrs)
	yvals = np.array(yvals)
	yerr_lows = np.array(yerr_lows)
	yerr_highs = np.array(yerr_highs)
	is_dbell = np.array(is_dbell, dtype=bool)

	if not inc_dbell:
		xvals = xvals[~is_dbell]
		xerrs = xerrs[~is_dbell]
		yvals = yvals[~is_dbell]
		yerr_lows = yerr_lows[~is_dbell]
		yerr_highs = yerr_highs[~is_dbell]
		labels = [lab for i, lab in enumerate(labels) if not is_dbell[i]]

	print(xvals)

	fig, ax = plt.subplots(figsize=(5, 4))

	ax.errorbar(
		xvals, yvals,
		xerr=xerrs, yerr=[yerr_lows, yerr_highs],
		fmt='o', color='tab:blue', ecolor='gray',
		capsize=2, alpha=0.8, label='Discs'
	)

	for i, label in enumerate(labels):
		ax.annotate(label, (xvals[i], yvals[i]), fontsize=8, ha='center', xytext=(0, 5), textcoords='offset points')
		if is_dbell[i]:
			ax.scatter(xvals[i], yvals[i], facecolors='none', edgecolors='red', s=120, linewidths=1.5, zorder=3)

	"""if not inc_dbell:
		annotate_correlations(ax, xvals, yvals)
	else:
		annotate_correlations(ax, xvals[~is_dbell], yvals[~is_dbell], label='No DBell', color='tab:red')
		annotate_correlations(ax, xvals, yvals, label='All', color='k')"""

	ax.set_xlabel(r'Averaged warp gradient: $\langle \log\,\psi \rangle_{R}$')
	ax.set_ylabel(r'$\log \alpha$')

	plt.tight_layout()
	plt.savefig('warp_vs_alphaS.pdf', bbox_inches='tight')
	plt.show()


def _register_target(target, vmax=0.2):
	base_disc_name_map.update({target: target})
	vmax_dict_12co.update({target: vmax})
	vmax_dict_13co.update({target: 0.5 * vmax})
	comparison_data.update(
		{
			target: {  #Just a placeholder rn
				"log_Mdot": -1.0,
				"NAI": 1.0,
				"NIR_excess": 1.0,
				"NIR_excess_err": 1.0,
				"NIR_ulim": False,
				"alpha_S": 1.0,
				"alpha_S_err_low": 1.0,
				"alpha_S_err_high": 1.0,
			}
		}
	)
	print(base_disc_name_map)

if __name__=='__main__':

	import argparse

	# Optional: Use pyfiglet for fancy font if installed
	try:
		from pyfiglet import Figlet
		def print_title():
			f = Figlet(font='slant')
			print(f.renderText('Warp Fitter'))
	except ImportError:
		def print_title():
			print('='*40)
			print('               Warp Fitter              ')
			print('='*40)


	#practice_2()
	#exit()
	parser = argparse.ArgumentParser()
	parser.add_argument('--plot', action='store_true')
	parser.add_argument('--reset', action='store_true')
	parser.add_argument('--Nbeam', type=int, default=1, help='Number of beams to sample')
	#parser.add_argument('--target', choices=['default', 'mwc758', 'v4046', 'both', 'testgrid', 'testaxi'], default='both')
	parser.add_argument('--target', type=str, default='all', help="Comma-separated list of targets: mwc758,v4046,both,testgrid,testaxi,testheight,default")
	parser.add_argument('--warp', action='store_true', help='Run warp analysis with GP fit to inclination profile')
	parser.add_argument('--co13', action='store_true', help='Use 13CO directory versions and filenames')
	parser.add_argument('--clip', type=float, help='Clip without scaling (just in warp analysis for now)')
	parser.add_argument('--unclip', action='store_true', help='Unclip -- overrides clip')
	parser.add_argument('--grid', action='store_true', help='Grid the txt files')
	parser.add_argument('--isocomp', action='store_true', help='Compare Delta i and Delta PA between 12CO and 13CO')
	parser.add_argument('--beamcomp', action='store_true', help='Compare warp amplitudes for different beam sizes (e.g. b0p30 vs default)')
	parser.add_argument('--residual', action='store_true', help='Show residuals of the velocity field')
	parser.add_argument('--wcomp', action='store_true', help='Compare warp fits between targets')

	parser.add_argument("--curone", action="store_true", help="Compare warp fit results with axisymmetry indices and accretion rates from Curone+2025")
	parser.add_argument("--parfile", action="store_true", help= "Infer target name and parameters from local parfile.json")
	parser.add_argument("--initdiscminer", action="store_true", help="Make target folders and initialise discminer files from scratch")
	
	args = parser.parse_args()

	target_list = [t.strip().lower() for t in args.target.split(',')]
	co13_suffix = '_13co' if args.co13 else ''

	def suffix_name(name, co13=args.co13):
		"""Return a name with _13co and/or _dbell suffixes based on the molecule and predefined lists."""
		name = name.lower()
		suffix = ''

		if co13:
			suffix += '_13co'
			if name in dbell_13co:
				suffix += '_dbell'
			if name in b0p03_13co:
				suffix += '_b0p30'
		else:
			if name in dbell_12co:
				suffix += '_dbell'
			if name in b0p03_12co:
				suffix += '_b0p30'

		return f"{name}{suffix}"
	
	clip_13co_dict = {'mwc758': 170.0, 'v4046': 200.0, 'hd34282': 450.0, 'aatau': 300.0, 'cqtau': 135.0, 'dmtau': 300.,
			  'hd135344': 180.0, 'hd143006': 140.0, 'j1604': False, 'j1615': 350., 'j1842': 200.0, 'j1852': 170.0, 
			  'lkca15':False, 'pds66': 70.0, 'sycha': 200.0}

	def clip_13co(name, co13=args.co13):
		if co13 and not args.unclip:
			return clip_13co_dict[name]
		return False

	if 'all' in target_list:
		#target_list.extend(['mwc758', 'v4046', 'aatau_dbell', 'cqtau', 'hd34282_dbell', 'dmtau', 'hd135344', 'hd143006', 'j1604', 'j1615', 'j1842', 'j1852',  'lkca15', 'pds66' ])
		target_list.extend(['mwc758', 'v4046', 'aatau', 'cqtau', 'hd34282', 'dmtau', 'hd135344', 'hd143006', 'j1604', 'j1615', 'j1842', 'j1852',  'lkca15', 'pds66', 'sycha' ])

	
	targets = []

	if args.parfile:
		
		with open("parfile.json") as json_file:
			pars = json.load(json_file)

		best = pars["best_fit"]
		meta = pars["metadata"]
		discname = meta["disc"]
		dpcpar = meta["dpc"]
		inclpar = best["orientation"]["incl"]
		mstarpar = best["velocity"]["Mstar"]
		velsignpar = best["velocity"]["vel_sign"]
		outerbound = 0.95 * best["intensity"]["Rout"]
		header = fits.getheader(meta["file_data"])
		bsize = 0.2 * header["BMAJ"] * 3600  # For radial binning
		chansp = np.abs(header["CDELT3"])  # Assumed in km/s

		if args.initdiscminer:
			subprocess.run(
				f"rm -rf velocity_residuals/azimuthal_velocity_residuals_{discname}",
				shell=True,
				check=True,
			)
			subprocess.run(
				f"mkdir -p velocity_residuals/azimuthal_velocity_residuals_{discname}",
				shell=True,
				check=True,
			)

			subprocess.run("discminer channels -mb 1 -so 0", shell=True, check=True)
			subprocess.run("discminer moments1d", shell=True, check=True)
			subprocess.run(
				"discminer azimprof -i 0.2 -o 0.95 -ig 1 -t residuals -w 1 -so 0",
				shell=True,
				check=True,
			)
			subprocess.run(
				f"mv azimuthal_velocity_residuals_{discname}.txt velocity_residuals/azimuthal_velocity_residuals_{discname}/",
				shell=True,
				check=True,
			)

		label = suffix_name(discname, co13=args.co13)
		targets.append(
			(
				label,
				f"azimuthal_velocity_residuals_{label}.txt",
				dpcpar,
				bsize,
				mstarpar,
				outerbound,  # Outer analysis boundary
				chansp,  # Channel spacing
				inclpar,
				velsignpar,
				clip_13co(meta["disc"]),
			)
		)

		_register_target(discname, vmax=pars["custom"]["vlim"])
		target_list = [discname]

	if not os.path.isdir("velocity_residuals"):
		print("Error: 'velocity_residuals' directory not found.")
		exit()

	os.chdir("velocity_residuals")
				
	
	if 'mwc758' in target_list and not args.parfile:
		label = suffix_name('mwc758', co13=args.co13)
		bsize = 0.15
		if label.split('_')[-1]=='b0p30':
			bsize=0.30
		targets.append((label, f'azimuthal_velocity_residuals_{label}.txt', 155.9, bsize, 1.40, 270.0, 0.02, 0.337866, 1.0, clip_13co('mwc758')))
		#(_, _, dpc, beamsize, mstar, Rout, channel_spacing, inclination, velsign)

	if 'mwc758_wc' in target_list and not args.parfile:
		label = suffix_name('mwc758', co13=args.co13)
		label = label + '_wc'
		bsize = 0.15
		if label.split('_')[-1]=='b0p30':
			bsize=0.30
		targets.append((label, f'azimuthal_velocity_residuals_{label}.txt', 155.9, bsize, 1.40, 270.0, 0.02, 0.337866, 1.0, clip_13co('mwc758')))

	if 'v4046' in target_list and not args.parfile:
		label = suffix_name('v4046', co13=args.co13)
		bsize = 0.15
		if label.split('_')[-1]=='b0p30':
			bsize=0.30
		targets.append((label, f'azimuthal_velocity_residuals_{label}.txt', 71.5, bsize, 1.73, np.inf, 0.02, -0.586914, 1.0, clip_13co('v4046')))

	if 'hd34282' in target_list and not args.parfile:
		label = suffix_name('hd34282', co13=args.co13)
		bsize = 0.15
		if label.split('_')[-1]=='b0p30':
			bsize=0.30
		targets.append((label, f'azimuthal_velocity_residuals_{label}.txt', 309.0, bsize, 1.62, 750.0, 0.02, -1.017799, 1.0, clip_13co('hd34282')))


	if 'hd34700' in target_list and not args.parfile:
		#NOTE THIS IS WRONG!
		label = suffix_name('hd34700', co13=args.co13)
		bsize = 0.15
		if label.split('_')[-1]=='b0p30':
			bsize=0.30
		targets.append((label, f'azimuthal_velocity_residuals_{label}.txt', 350.5, bsize, 3.59, np.inf, 0.02, 0.616, 1.0, clip_13co('hd34282')))


	if 'aatau' in target_list and not args.parfile:
		label = suffix_name('aatau', co13=args.co13)
		bsize = 0.15
		if label.split('_')[-1]=='b0p30':
			bsize=0.30
		targets.append((label, f'azimuthal_velocity_residuals_{label}.txt', 135.0, bsize, 0.79, 500.0, 0.02, -1.024645, 1.0, clip_13co('aatau')))

	if 'cqtau' in target_list and not args.parfile:
		label = suffix_name('cqtau', co13=args.co13)
		bsize = 0.15
		if label.split('_')[-1]=='b0p30':
			bsize=0.30
		targets.append((label, f'azimuthal_velocity_residuals_{label}.txt', 149.0, bsize, 1.40, np.inf, 0.02, -0.632653, -1.0, clip_13co('cqtau')))

	if 'dmtau' in target_list and not args.parfile:
		label = suffix_name('dmtau', co13=args.co13)
		bsize = 0.15
		if label.split('_')[-1]=='b0p30':
			bsize=0.30
		targets.append((label, f'azimuthal_velocity_residuals_{label}.txt', 144.0, bsize, 0.45, 540.0, 0.02, 0.702772, 1.0, clip_13co('dmtau')))

	if 'hd135344' in target_list and not args.parfile:
		label = suffix_name('hd135344', co13=args.co13)
		bsize = 0.15
		if label.split('_')[-1]=='b0p30':
			bsize=0.30
		targets.append((label, f'azimuthal_velocity_residuals_{label}.txt', 135.0, bsize, 1.61, np.inf, 0.02, -0.281219, -1.0, clip_13co('hd135344')))

	if 'hd143006' in target_list and not args.parfile:
		label = suffix_name('hd143006', co13=args.co13)
		bsize = 0.15
		if label.split('_')[-1]=='b0p30':
			bsize=0.30
		targets.append((label, f'azimuthal_velocity_residuals_{label}.txt', 167.0, bsize, 1.56, np.inf, 0.02, -0.2952398962673608, 1.0, clip_13co('hd143006')))

	if 'j1604' in target_list and not args.parfile:
		label = suffix_name('j1604', co13=args.co13)
		bsize = 0.15
		if label.split('_')[-1]=='b0p30':
			bsize=0.30
		targets.append((label, f'azimuthal_velocity_residuals_{label}.txt', 145.0, bsize, 1.29, np.inf, 0.02, 0.103879, 1.0, clip_13co('j1604')))

	if 'j1615' in target_list and not args.parfile:
		label = suffix_name('j1615', co13=args.co13)
		bsize = 0.15
		if label.split('_')[-1]=='b0p30':
			bsize=0.30
		targets.append((label, f'azimuthal_velocity_residuals_{label}.txt', 156.0, bsize, 1.14, 540.0, 0.02, 0.804316, 1.0, clip_13co('j1615')))

	if 'j1842' in target_list and not args.parfile:
		label = suffix_name('j1842', co13=args.co13)
		bsize = 0.15
		if label.split('_')[-1]=='b0p30':
			bsize=0.30
		targets.append((label, f'azimuthal_velocity_residuals_{label}.txt', 151.0, bsize, 1.07, np.inf, 0.02, 0.687349, -1.0, clip_13co('j1842')))

	if 'j1852' in target_list and not args.parfile:
		label = suffix_name('j1852', co13=args.co13)
		bsize = 0.15
		if label.split('_')[-1]=='b0p30':
			bsize=0.30
		targets.append((label, f'azimuthal_velocity_residuals_{label}.txt', 147.0, bsize, 1.03, 250.0, 0.02, -0.570361, 1.0, clip_13co('j1852')))

	if 'lkca15' in target_list and not args.parfile:
		label = suffix_name('lkca15', co13=args.co13)
		bsize = 0.15
		print(label, label.split('_')[-1])
		if label.split('_')[-1]=='b0p30':
			bsize=0.30
		targets.append((label, f'azimuthal_velocity_residuals_{label}.txt', 157.0, bsize, 1.17, 700.0, 0.02, 0.880406, -1.0, clip_13co('lkca15')))

	if 'pds66' in target_list and not args.parfile:
		label = suffix_name('pds66', co13=args.co13)
		bsize = 0.15
		if label.split('_')[-1]=='b0p30':
			bsize=0.30
		targets.append((label, f'azimuthal_velocity_residuals_{label}.txt', 98.0, bsize, 1.28, np.inf, 0.02, -0.556906, -1.0, clip_13co('pds66')))

	if 'sycha' in target_list and not args.parfile:
		label = suffix_name('sycha', co13=args.co13)
		bsize = 0.15
		if label.split('_')[-1]=='b0p30':
			bsize=0.30
		targets.append((label, f'azimuthal_velocity_residuals_{label}.txt', 182.0, bsize, 0.81, 540.0, 0.02, -0.884481, -1.0, clip_13co('sycha')))

	if 'testgrid' in target_list and not args.parfile:
		grid_files = sorted(glob.glob("**/azimuthal_velocity_residuals_incl*deg_PA*deg_xc*_yc*.txt", recursive=True))
		for full_path in grid_files:
			if '_axisym_' not in full_path:
				fname = os.path.basename(full_path)
				if fname != 'azimuthal_velocity_residuals_incl0deg_PA0deg_xc0_yc0.txt':
					match = re.search(r'incl(\d+)deg_PA(\d+)deg_xc(\d+)_yc(\d+)', fname)
					if match:
						label = f"incl{match.group(1)}_PA{match.group(2)}_xc{match.group(3)}_yc{match.group(4)}"
						targets.append((label, fname, 100.0, 0.15, 1.0, np.inf, 0.02, 0.52, 1.0, False))

	if 'testheight' in target_list and not args.parfile:
		height_dirs = sorted(glob.glob("azimuthal_velocity_residuals_z*_p*_q*_Rb*", recursive=True))
		print(height_dirs)
		for folder in height_dirs:
			fname = os.path.basename(folder) + ".txt"
			full_txt_path = os.path.join(folder, fname)
			if not os.path.isfile(full_txt_path):
				continue  # Skip if the .txt file doesn't exist

			match = re.search(
				r'azimuthal_velocity_residuals_z([-+]?[0-9]*\.?[0-9]+)_p([-+]?[0-9]*\.?[0-9]+)_q([-+]?[0-9]*\.?[0-9]+)_Rb([-+]?[0-9]*\.?[0-9]+).txt',
				fname
			)
			if match:
				z0, p, q, Rb = match.groups()
				label = f"z0={z0}_p={p}_q={q}_Rb={Rb}"
				targets.append((label, fname,100.0, 0.15, 1.0, np.inf, 0.02, 0.52, 1.0, False))

	if 'testaxi' in target_list and not args.parfile:
		axi_files = sorted(glob.glob("**/*_axisym_*.txt", recursive=True))
		for full_path in axi_files:
			fname = os.path.basename(full_path)
			match = re.search(r'axisym_inc_(\d+)_vr_([0-9.]+)_vz_([0-9.]+)_vphi_([0-9.]+)\.txt$', fname)
			if match:
				inc, vr, vz, vphi = match.groups()
				label = f"inc{inc}_vr{vr}_vz{vz}_vphi{vphi}"
				targets.append((label, fname, 100.0, 0.15, 1.0, np.inf, 0.02, 0.52, 1.0, False))

	print_title()
	print("Author: Andrew Winter\n")

	# Summarise run mode
	modes = []
	if args.grid: modes.append("grid")
	if args.plot: modes.append("plot")
	if args.warp: modes.append("warp")

	mode_summary = ", ".join(modes) if modes else "idle (no processing requested)"
	print(f"Mode: {mode_summary}")

	# Show target summary
	print("\nTargets:")
	for label, fname, dist, beam, mstar, rout, channel_spacing, incl, rot_sign, clip13 in targets:
		print(f"  - {label:<20}  file: {fname:<55}  dist: {dist:>5} pc  beam: {beam}''  M⋆: {mstar} M☉  i: {incl}")
	print()

	result = None

	results = {}
	results_warp = {}
	for label, fname, dist, beam, mstar, rout, channel_spacing, incl, rot_sign, clip13 in targets:
		folder = os.path.splitext(fname)[0]
		
		if args.grid:
			r_, p_, dv_, meta_ = load_or_grid(fname, 
				dist_pc=dist,
				beam_fwhm_arcsec=beam,
				mstar_norm=mstar,
				incl=incl,
				trunc_rout=rout,
				plot=args.plot,
				channel_spacing=channel_spacing,
				Nbeam=args.Nbeam,
				folder=folder,
				reset=args.reset,
				hreset  =args.reset,
				rot_sign = rot_sign
			)
		

		if not result is None:
			results[label] = result
			results[label]['stellar_mass'] = mstar

		if args.warp:
			print(f">>> Running warp analysis for: {label}")

			meta_path = f"{folder}_meta_params.npy"
			if args.co13:
				molecule = '$^{13}$CO'
			else:
				molecule = '$^{12}$CO'

			if label in disc_name_map:
				dnl = disc_name_map[label]
			else:
				dnl = label
			fit_result = load_or_run_warp_fit(
				fname=fname,
				folder=folder,
				plot=args.plot,
				reset=args.reset,
				incl=incl,
				hreset=args.reset,
				dv_max=0.5,
				discname=dnl,
				rot_sign=rot_sign,
				clip=clip13,
				molecule=molecule,
				show_residuals=args.residual
			)

			if label in disc_name_map and disc_name_map[label] in comparison_data:
				results_warp[label] = fit_result
				results_warp[label]['stellar_mass'] =mstar
				results_warp[label]['incl'] =incl
				results_warp[label]['meta_params'] = np.load(folder+'/'+meta_path, allow_pickle=True).item()
			else:
				print(f"'{label}' not found in the disc name map.")



	#plot_warp_vs_alphaS(results_warp, disc_name_map, inc_dbell=True)

	if args.wcomp:
		plot_warp_comparison(results_warp)
		
	results_warp_12co = {}
	results_warp_13co = {}

	if args.isocomp:
		for label, fname, dist, beam, mstar, rout, channel_spacing, incl, rot_sign, clip13 in targets:

			target =label.split('_')[0]
			for molecule_label, use_13co in [('12co', False), ('13co', True)]:
				label = suffix_name(target, co13=use_13co)
				fname = f'azimuthal_velocity_residuals_{label}.txt'
				folder = os.path.splitext(fname)[0]

				if not os.path.exists(os.path.join(folder, fname)):
					print(f"Skipping {label} because file was not found.")
					continue

				print(f"Loading for comparison: {label}")
				fit_result = load_or_run_warp_fit(
					fname=fname,
					folder=folder,
					plot=False,
					reset=args.reset,
					incl=incl,  # can leave None if stored in metadata
					hreset=args.hardreset,
					dv_max=0.5,
					discname=disc_name_map.get(label, label),
					rot_sign=rot_sign,
					clip=clip_13co(target, co13=use_13co)
				)

				meta_path = f"{folder}_meta_params.npy"

				if fit_result is not None and label in disc_name_map:
					fit_result['incl'] = incl
					fit_result['meta_params'] = np.load(folder + '/' + meta_path, allow_pickle=True).item()
					fit_result['stellar_mass'] = mstar

					if use_13co:
						results_warp_13co[label] = fit_result
						results_warp_13co[label]['incl'] = incl
					else:
						results_warp_12co[label] = fit_result
						results_warp_12co[label]['incl'] = incl
		
		if args.plot:
			plot_warp_amplitude_comparison(results_warp_12co, results_warp_13co)
			plot_inclination_vs_pa_compare(results_warp_12co, results_warp_13co)
		
	if args.beamcomp:
		results_015 = {}
		results_030 = {}
		beamset = b0p03_13co if args.co13 else b0p03_12co
		for label, fname, dist, beam, mstar, rout, channel_spacing, incl, rot_sign, clip13 in targets:
				target = label.split('_')[0]
				if target not in (b0p03_13co if args.co13 else b0p03_12co):
					continue  # only consider targets that have b0p30 versions

				# --------- 0.30" (default) run ---------
				label_030 = suffix_name(target, co13=args.co13)
				fname_030 = f"azimuthal_velocity_residuals_{label_030}.txt"
				folder_030 = os.path.splitext(fname_030)[0]

				if os.path.exists(os.path.join(folder_030, fname_030)):
					print(f">>> [b030] Running warp fit for {label_030} (beam=0.30\")")
					result_030 = load_or_run_warp_fit(
						fname=fname_030,
						folder=folder_030,
						plot=False,
						reset=args.reset,
						incl=incl,
						hreset=args.hardreset,
						dv_max=0.5,
						discname=disc_name_map.get(label_030, label_030),
						rot_sign=rot_sign,
						clip=clip13,
						molecule='$^{13}$CO' if args.co13 else '$^{12}$CO'
					)
					if result_030 is not None:
						meta_030_path = f"{folder_030}/{folder_030}_meta_params.npy"
						if os.path.exists(meta_030_path):
							result_030['meta_params'] = np.load(meta_030_path, allow_pickle=True).item()
						result_030['stellar_mass'] = mstar
						result_030['incl'] = incl
						results_030[label_030] = result_030
				else:
					print(f"[b030] Skipping {label_030} — file not found.")

				# --------- 0.15" run (manually strip _b0p30) ---------
				label_015 = label_030.replace('_b0p30', '')
				fname_015 = f"azimuthal_velocity_residuals_{label_015}.txt"
				folder_015 = os.path.splitext(fname_015)[0]

				if os.path.exists(os.path.join(folder_015, fname_015)):
					print(f">>> [b015] Running warp fit for {label_015} (beam=0.15\")")
					result_015 = load_or_run_warp_fit(
						fname=fname_015,
						folder=folder_015,
						plot=False,
						reset=args.reset,
						incl=incl,
						hreset=args.hardreset,
						dv_max=0.5,
						discname=disc_name_map.get(label_015, label_015),
						rot_sign=rot_sign,
						clip=clip13,
						molecule='$^{13}$CO' if args.co13 else '$^{12}$CO'
					)
					if result_015 is not None:
						meta_015_path = f"{folder_015}/{folder_015}_meta_params.npy"
						if os.path.exists(meta_015_path):
							result_015['meta_params'] = np.load(meta_015_path, allow_pickle=True).item()
						result_015['stellar_mass'] = mstar
						result_015['incl'] = incl
						results_015[label_015] = result_015
				else:
					print(f"[b015] Skipping {label_015} — file not found.")

		if args.plot:
			plot_beam_amplitude_comparison(results_015, results_030)

	#plot_warp_vs_mdot_norm(results_warp, disc_name_map)

	# After warp analysis:
	if args.curone:
		compare_warp_to_curone(results_warp, disc_name_map, xaxis_log_psi=True)
		compare_warp_to_curone(results_warp, disc_name_map, xaxis_log_psi=False)
	if args.plot:
		plot_inclination_vs_pa(results_warp)
