# Warp fitter

Fit simple radial warp profiles ($\delta i(R)$, $\delta \mathrm{PA}(R)$) to line-of-sight velocity **residual maps** using lightweight Gaussian-process smoothing. Produces diagnostic plots and CSVs of GP posterior samples for analysis.

> Repo: [https://github.com/ajw278/warpfitter](https://github.com/ajw278/warpfitter)

---

## Quick start

```bash
# 1) Clone and enter
git clone https://github.com/ajw278/warpfitter
cd warpfitter

# 2) Prepare your input file (see “Input format” below)
#    Place it under: velocity_residuals/azimuthal_velocity_residuals_YYYY/
#    and name it:    azimuthal_velocity_residuals_YYYY.txt

# 3) Build grids (creates cached .npy files)
python warp_fitter.py --target YYYY --grid

# 4) Fit the warp model and make plots
python warp_fitter.py --target YYYY --warp --plot
```

This will:

* Grid your residuals to a uniform $(R,\phi)$ mesh and cache arrays like `*_gridded.npy`, `*_rgrid.npy`, `*_phigrid.npy`, plus `*_meta_params.npy`.
* Fit $\delta i(R)$ and $\delta \mathrm{PA}(R)$ via a simple cos/sin decomposition and GP smoothing.
* Save posterior samples to CSV and produce a multi-panel diagnostic PDF (observed/model/residual maps + profiles).

---

## Input format & folder layout

WarpFitter expects a **single TXT file** per target inside a directory named after that file (for caching):

```
velocity_residuals/
  azimuthal_velocity_residuals_YYYY/
    azimuthal_velocity_residuals_YYYY.txt     # <- your input
```

Where **`YYYY`** is any label you choose (e.g. a disc name or model tag). The TXT file must be a "discminer"-type export with rows of the form:

```
R_au   phi_deg[0] phi_deg[1] ...   dv_kms[0] dv_kms[1] ...
```

* `R_au`: radius in astronomical units for that row
* `phi_deg[*]`: list of azimuths (degrees)
* `dv_kms[*]`: velocity residuals from Keplerian (km/s) at those azimuths

WarpFitter will interpolate to a uniform azimuth grid.

---

## Installation

Python ≥ 3.9 recommended.

**Dependenciess** (as used in `warp_fitter.py`):

* numpy, scipy, pandas
* scikit-learn (GaussianProcessRegressor, Matérn kernel)
* matplotlib, adjustText

**Optional:**

* pyfiglet 

The repo also expects local modules in the root: `mpl_setup.py` (plot styling) and `utils.py` (I/O + gridding helpers).

---

## CLI usage

```bash
python warp_fitter.py [flags]
```

**Most common flags**

* `--target YYYY` : Comma-separated list or a single label. Use the same `YYYY` as in your filenames.
* `--grid`       : Build/read the gridded caches for the target(s).
* `--warp`       : Fit the GP warp model and write outputs.
* `--plot`       : Show and save diagnostic figures.

**Other useful flags**

* `--co13`           : Use 13CO naming/suffix logic for built-in targets.
* `--clip <R_au>`    : Clip the maps at radius `R_au` during the warp analysis.
* `--unclip`         : Disable any default clipping for 13CO runs.
* `--residual`       : Add an Obs−Model residual panel to the top-row figure.
* `--reset`          : Recompute and overwrite cached results for the target(s).
* `--wcomp`          : Compare warp fits across selected targets in a single figure.
* `--isocomp`        : Compare 12CO vs 13CO warp amplitudes for shared targets.
* `--beamcomp`       : Compare amplitudes across nominal vs larger-beam variants.
* `--Nbeam <int>`    : Sampling density control for gridding (advanced).

Run `python warp_fitter.py -h` for full help.

---

## Required file name

Your input filename **must** be:

```
azimuthal_velocity_residuals_YYYY.txt
```

and it must live inside the folder of the same stem:

```
azimuthal_velocity_residuals_YYYY/
```

WarpFitter will operate from `velocity_residuals/` and change into that per-target folder during processing.

---

## Adding a new target (temporary hardcoded list)

At the moment, the script uses a hardcoded list of targets in `warp_fitter.py` (look for the big `targets.append(( ... ))` blocks). If your custom `YYYY` is **not** one of the built-ins, add a tuple for it following the same pattern:

```python
# somewhere near other targets.append calls
label = 'YYYY'  # your unique label; must match your file stem
fname = f'azimuthal_velocity_residuals_{label}.txt'

targets.append((
    label,         # label (used for folder and plots)
    fname,         # input filename (inside the same-named folder)
    100.0,         # dist_pc         (distance in pc)
    0.15,          # beam_fwhm_arcsec
    1.0,           # mstar_norm      (M_★ in solar units)
    np.inf,        # R_out_trunc_au  (outer-radius clip for gridding)
    0.02,          # channel_spacing (km/s)
    np.deg2rad(45),# inclination [radians]; sign matters only for sin/cos use
    1.0,           # rot_sign        (+1 if clockwise on the sky, −1 otherwise)
    False          # clip_13co radius or False
))
```

> If you prefer not to edit the code, you can also re-use an existing block and just change `label`, `fname`, and the meta-parameters. A future version will move this to a YAML/JSON config.

**About `rot_sign`**

* Use `+1.0` if increasing azimuth (the way you defined it) corresponds to the *clockwise* rotating side in the residual convention you adopted.
* Use `−1.0` if it is the opposite. If your $\cos\phi$/$\sin\phi$ projections look flipped, try toggling this.

---

## Using an existing Discminer model of your target

If you already have a `discminer` model of your target, you can automatically generate the required `azimuthal_velocity_residuals_YYYY.txt` file and fit a warp with `warp_fitter.py` as follows:

1. Place the following files inside your **warpfitter** folder:
   - A reduced (prototype) **datacube** of the disc.
   - A `parfile.json` file with disc parameters.

These are typically generated with:  
   ```bash
   python prepare_data.py
   discminer parfile
   ```
   (see [Discminer example](https://github.com/andizq/discminer/tree/main/example)).

2. Run `warpfitter`:
   ```bash
   python warp_fitter.py --warp --grid --plot --parfile --initdiscminer
   ```
   
- `--parfile` : read target parameters from the existing `parfile.json` and pass them to `warpfitter`.  
- `--initdiscminer` : compute the required residuals file and move it to `azimuthal_velocity_residuals_YYYY/` (only needed once per target).  

Subsequent runs for the same disc do not require `--initdiscminer`.  


## What the code does 

1. **Gridding (`--grid`)**

   * Reads your TXT; interpolates residuals to a uniform azimuth grid $\phi\in[-\pi,\pi]$ per radius row.
   * Writes:

     * `*_gridded.npy`   : 2D array of $\delta v_{los}(R,\phi)$
     * `*_rgrid.npy`     : 1D radius array (au)
     * `*_phigrid.npy`   : 1D azimuth array (rad)
     * `*_meta_params.npy` : dict with `dist_pc`, `beam_fwhm_arcsec`, `mstar_norm`, `incl`, `R_out`, `channel_spacing`, etc.

2. **Decomposition & GP fit (`--warp`)**

   * At each radius, fits $\delta v(\phi) \approx A\cos\phi + B\sin\phi$, then maps $A\to \delta i$ and $B\to \delta\mathrm{PA}$ using Keplerian scaling with $M_\star$ and `incl`.
   * Smooths $\delta i(R)$, $\delta\mathrm{PA}(R)$ using 1D GPs with Matérn kernels (beam-sized length scales).
   * Samples 200 GP realizations to derive profile uncertainties and derived curves:

     * $\beta(R) = |\Theta(R)| = \sqrt{\delta i^2 + (\sin i_0\,\delta\mathrm{PA})^2}$
     * $\psi(R) = \big|d\Theta/d\ln R\big|$ (log-space gradient)
     * $\gamma(R)$ (twist angle; defined from $\delta i,\delta \mathrm{PA}$)

3. **Plotting (`--plot`)**

   * 2D maps: Observed $\delta v$, model $\delta v$, optional residual.
   * Radial panels: $\delta i(R)$, $\delta \mathrm{PA}(R)$, and optionally $\beta(R)$, $\gamma(R)$, $\psi(R)$ with posterior sample overlays.

---

## Outputs

Within the per-target folder (`velocity_residuals/azimuthal_velocity_residuals_YYYY/`):

* Caches: `*_gridded.npy`, `*_rgrid.npy`, `*_phigrid.npy`, `*_meta_params.npy`, `*_rescaled.npy`
* GP fit: `*_warpfit_result.pkl` (pickle of arrays & scalars)
* Posterior samples (CSV):

  * `gp_samples/<DiscName>_delta_i_gp_samples.csv`
  * `gp_samples/<DiscName>_delta_pa_gp_samples.csv`
  * `gp_samples/<DiscName>_beta_gp_samples.csv`
  * `gp_samples/<DiscName>_psi_gp_samples.csv`
  * `gp_samples/<DiscName>_gamma_gp_samples.csv`
* Gridded model/obs planes ($x,y$) as NPY: `<DiscName>_dv_model.npy`, `<DiscName>_dv_obs.npy`, `<DiscName>_xygrid.npy`
* PDFs: `combined_model_plot_<DiscName>.pdf` (and variants for 13CO/beam tests), plus any comparison figures triggered by flags.

> If you see a **channel spacing mismatch** warning, the meta file is updated and the code will rescore as needed.

---

## Worked example

Suppose you have residuals for **`cqtau`** in a file named:

```
velocity_residuals/azimuthal_velocity_residuals_cqtau/
  azimuthal_velocity_residuals_cqtau.txt
```

Add an entry in the targets list (inside `warp_fitter.py`) like:

```python
label = 'cqtau'
fname = f'azimuthal_velocity_residuals_{label}.txt'
# dist=149pc, beam=0.15", M*=1.40Msun, R_out=∞, dV=0.02km/s, incl=-36.25° (in rad), rot_sign=-1
targets.append((label, fname, 149.0, 0.15, 1.40, np.inf, 0.02, -0.632653, -1.0, False))
```

Then run:

```bash
python warp_fitter.py --target cqtau --grid
python warp_fitter.py --target cqtau --warp --plot
```

---

## Troubleshooting

* **`Error: 'velocity_residuals' directory not found.`**
  Ensure you are running the script from the repo root (where `warp_fitter.py` lives), and that the `velocity_residuals/` folder exists.

* **My target isn’t recognized.**
  For now, you must add a `targets.append((...))` block for your `YYYY` (see above). Make sure `label` and the file stem match.

* **Sign conventions / flipped patterns.**
  Try switching `rot_sign` between `+1.0` and `-1.0`. Also double-check your azimuth definition and disk major-axis orientation used to compute residuals.

* **Axis labels/units look odd.**
  Distance and beam size propagate into the velocity-map panels (via AU conversion). Verify `dist_pc` and `beam_fwhm_arcsec` in your target tuple.

---

## Future changes:

* Move the per-target configuration to an external YAML/JSON.
* Friendlier I/O layer for different residual formats.
* Optional Bayesian model with full likelihood (beyond GP smoothing step).

---

## How to cite

If you use WarpFitter in a publication, please cite the repository and the associated paper https://ui.adsabs.harvard.edu/abs/2025ApJ...990L..10W/abstract :

```
A. J. Winter et al. 2025, ApJL, 990, L10
```
