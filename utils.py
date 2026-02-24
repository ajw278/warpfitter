
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from mpl_setup import *

def load_discdat_old(fname):
	data  =np.loadtxt(fname, delimiter=' ', dtype=str) 

	return data


def load_discdat(fname):
    rows = []

    with open(fname, "r") as f:
        for iline, line in enumerate(f, start=1):
            line = line.strip()

            # Skip blank/comment lines
            if not line or line.startswith("#"):
                continue

            # Split on arbitrary whitespace (more robust than delimiter=' ')
            parts = line.split()

            if len(parts) == 3:
                # Original format:
                # radius   phi1,phi2,...   dv1,dv2,...
                rows.append(parts)

            elif len(parts) > 3:
                # New format:
                # radius phi1 phi2 ... phiN dv1 dv2 ... dvN
                n = len(parts)

                if (n - 1) % 2 != 0:
                    raise ValueError(
                        f"Line {iline}: expected 1 + 2*N entries, got {n} entries."
                    )

                n_az = (n - 1) // 2

                radius = parts[0]
                phi_vals = parts[1:1 + n_az]
                dv_vals = parts[1 + n_az:]

                # Re-pack into the old 3-field format so downstream code still works
                rows.append([radius, ",".join(phi_vals), ",".join(dv_vals)])

            else:
                raise ValueError(
                    f"Line {iline}: expected either 3 entries (old format) "
                    f"or 1+2*N entries (new format), got {len(parts)}."
                )

    return np.array(rows, dtype=object)

# === BEAM-SAMPLED INTERPOLATION (GRID-BASED) ===
def beam_sample_highres(R_arr, Phi_arr, dv_interp2d, meta_params, Nbeam=4.0):

	beam_fwhm_arcsec = meta_params['beam_fwhm_arcsec']
	dist_pc = meta_params['dist_pc']
	R_out_phys = meta_params['R_out']

	R_out = np.amax(R_arr)
	if R_out!=1.0:
		raise Warning("Radial grid should be normalised before performing beam sampling")
	
	
	arcsec_per_au = 1.0 / dist_pc
	beam_fwhm_au = beam_fwhm_arcsec / arcsec_per_au
	beam_width_norm = beam_fwhm_au/R_out_phys # Normalized to outer radius = 1


	# Generate a Cartesian grid in x, y
	x = np.arange(-1, 1 + beam_width_norm/4., beam_width_norm/float(Nbeam))
	y = np.arange(-1, 1 + beam_width_norm/4., beam_width_norm/float(Nbeam))
	X, Y = np.meshgrid(x, y)

	#De-grid to avoid Fourier residuals
	X_rand = (np.random.uniform(size=X.shape)-0.5)*np.diff(x)[0]
	Y_rand = (np.random.uniform(size=X.shape)-0.5)*np.diff(y)[0]
	X+=X_rand
	Y+=Y_rand

	R_cart = np.sqrt(X**2 + Y**2)
	Phi_cart = np.arctan2(Y, X)%(2.*np.pi)-np.amax(Phi_arr)

	R_min = np.amin(R_arr)
	R_max = np.amax(R_arr)
	mask = (R_cart <= R_max) & (R_cart >= R_min)
	R_samples = R_cart[mask]
	Phi_samples = Phi_cart[mask]

	interp_func = RegularGridInterpolator(
		(R_arr, Phi_arr),
		dv_interp2d
	)

	pts = np.stack([R_samples, Phi_samples], axis=-1)
	dv_samples = interp_func(pts)

	dv_99 = np.percentile(np.absolute(dv_samples), 99.0 )

	ifilt = np.absolute(dv_samples)<dv_99

	sampled_array = np.stack([R_samples[ifilt], Phi_samples[ifilt], dv_samples[ifilt]])  # Shape (3, N)


	return sampled_array

# === 2D INTERPOLATION TO UNIFORM POLAR GRID ===
def interpolate_to_uniform_azimuth(fname, phi_highres= np.linspace(-np.pi, np.pi, 257), R_trunc=np.inf):

	data = load_discdat(fname)

	# Arrays to store results
	radii = []
	dv_rows = []

	for row in data:
		radius = float(row[0])


		phi = np.array(row[1].split(','), dtype=float)*np.pi/180.0
		dv = np.array(row[2].split(','), dtype=float)

		# Sort for consistency
		sort_idx = np.argsort(phi)
		phi = phi[sort_idx]

		# Sort by azimuth to ensure correct ordering
		sort_idx = np.argsort(phi)
		phi = phi[sort_idx]
		dv = dv[sort_idx]

		dv = np.where(np.isnan(dv), 0.0, dv)

		# Interpolate dv onto the high-resolution phi grid
		dv_interp = np.interp(phi_highres, phi, dv)

		#Check periodicity
		if dv_interp[-1] != dv_interp[0]:
			if np.absolute(dv_interp[-1]) > np.absolute(dv_interp[0]):
				dv_interp[0] = dv_interp[-1]
			else:
				dv_interp[-1] = dv_interp[0]


		radii.append(radius)
		dv_rows.append(dv_interp)

	# Convert to numpy arrays
	radii = np.array(radii)
	dv_grid = np.array(dv_rows)  # Shape (n_radii, n_phi)

	iinc = radii<R_trunc


	radii = radii[iinc]
	dv_grid = dv_grid[iinc, :]

	# Ensure sorted
	radii_sort_idx = np.argsort(radii)
	radii = radii[radii_sort_idx]
	dv_grid = dv_grid[radii_sort_idx, :]

	if len(np.unique(radii)) != len(radii):
		print("Warning: duplicate radii found!")

	return radii, dv_grid


def downsample_and_interpolate(xx, yy, fn, max_size=150):
    """
    Downsample a large grid, apply a function, and interpolate the result
    back to the original resolution.

    Parameters
    ----------
    xx, yy : ndarray
        Meshgrids of shape (N, M).
    fn : function
        Function to apply on the downsampled meshgrid.
        Must return an array of shape (N_down, M_down).
    max_size : int
        Maximum number of pixels in each dimension for downsampling.

    Returns
    -------
    interp_result : ndarray
        Interpolated result on the original (xx, yy) grid.
    """
    x = xx[0]
    y = yy[:, 0]

    N, M = len(y), len(x)
    factor_x = max(1, M // max_size)
    factor_y = max(1, N // max_size)

    # Downsample
    x_small = x[::factor_x]
    y_small = y[::factor_y]
    xx_small, yy_small = np.meshgrid(x_small, y_small)

    # Apply the function
    result_small = fn(xx_small, yy_small)

    # Interpolate back
    interp = RegularGridInterpolator((y_small, x_small), result_small, bounds_error=False, fill_value=np.nan)
    coords = np.stack([yy.ravel(), xx.ravel()], axis=-1)
    interp_result = interp(coords).reshape(xx.shape)

    return interp_result

def write_meta(radii, dv_grid,  trunc_rout=np.inf, dist_pc=None, beam_fwhm_arcsec=None, mstar_norm=1.0, rot_sign=1.0, channel_spacing=0.1, incl=None):

	#As currently written, the radii are truncated first -- so R_out_init always equals R_out
	R_out_init = np.amax(radii)
	R_trunc = trunc_rout
	R_out = min(R_out_init,R_trunc)

	dvtrunc = dv_grid[radii<R_trunc, :] 
	radii = radii[radii<R_trunc]

	v_K = Keplerian(radii, mstar_norm)

	dv_norm = dvtrunc/v_K[:, np.newaxis]

	dv_variance = np.var(dv_norm)

	meta_params = {'R_out_init': R_out_init, 'R_out': R_out, 'R_trunc': R_trunc}
	meta_params['dist_pc'] = dist_pc
	meta_params['beam_fwhm_arcsec'] = beam_fwhm_arcsec
	meta_params['mstar_norm'] = mstar_norm
	meta_params['dv_variance'] = dv_variance
	meta_params['channel_spacing'] = channel_spacing
	meta_params['incl'] = incl
	meta_params['rot_sign'] = rot_sign

	return meta_params

def Keplerian(R_au, M_star):
	G = 4.302e-3  # pc (Msol)^-1 (km/s)^2
	R_pc = R_au / 206265.0  # convert au to pc
	v_kep = np.sqrt(G * M_star / R_pc)  # km/s
	return v_kep

