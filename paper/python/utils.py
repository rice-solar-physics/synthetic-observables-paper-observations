"""
Basic utils used throughout paper
"""
import seaborn
import numpy as np
import astropy.units as u
from sunpy.map import GenericMap


# Color palette for heating functions
def heating_palette(n_colors=3):
    return seaborn.color_palette(palette='colorblind', n_colors=n_colors,)


# General qualitative color palette
def qualitative_palette(n):
    return seaborn.color_palette('Dark2', n)


def make_slope_map(emcube, temperature_lower_bound=None, em_threshold=None):
    """
    Fit emission measure distribution in every pixel

    The fit is computed between `temperature_lower_bound`
    and the temeperature at which the EM is maximum.
    If the total emission (over all temperature bins) is less than
    `em_threshold`, no fit is calculated for that pixel.

    Parameters
    ----------
    emcube: `EMCube`
        Emission measure map as a function space and temperature
    temperature_lower_bound: `~astropy.units.Quantity`
    em_threshold: `~astropy.units.Quantity`, optional
        If the total EM in a pixel is below this, no slope is calculated

    Returns
    -------
    slope_map: `~sunpy.map.GenericMap`
    rsquared_map: `~sunpy.map.GenericMap`
    """
    if em_threshold is None:
        em_threshold = u.Quantity(1e25, u.cm**(-5))
    i_valid = np.where(
        u.Quantity(emcube.total_emission.data, emcube[0].meta['bunit']) > em_threshold)
    em_valid = np.log10(emcube.as_array()[i_valid])
    # Set any NaNs or Infs to zero. These will be weighted as zero during the fitting
    em_valid[~np.isfinite(em_valid)] = 0.0
    i_peak = em_valid.argmax(axis=1)
    log_temperature_bin_centers = np.log10(emcube.temperature_bin_centers.value)
    if temperature_lower_bound is None:
        i_lower = 0
    else:
        # Find closest bin to selected lower bound on temperature
        i_lower = np.fabs(emcube.temperature_bin_centers - temperature_lower_bound).argmin()
    slopes, rsquared = [], []
    for emv, ip in zip(em_valid, i_peak):
        t_fit = log_temperature_bin_centers[i_lower:ip]
        em_fit = emv[i_lower:ip]
        # Do not give any weight to bins with no emission. This includes the NaN/Inf
        # bins we set to zero earlier.
        w = np.where(em_fit > 0, 1, 0)
        # Ignore fits on two or less points or where all but two or less of the
        # weights are zero
        if t_fit.size < 3 or np.where(w == 1)[0].size < 3:
            slopes.append(np.nan)
            rsquared.append(0.)
            continue
        coeff, rss, _, _, _ = np.polyfit(t_fit, em_fit, 1, full=True, w=w)
        # Calculate the zeroth-order fit in order to find the correlaton
        _, rss_flat, _, _, _ = np.polyfit(t_fit, em_fit, 0, full=True, w=w)
        slopes.append(coeff[0])
        rsquared.append(1-rss[0]/rss_flat[0])

    slopes_data = np.zeros(emcube.total_emission.data.shape)
    slopes_data[i_valid] = slopes
    rsquared_data = np.zeros(emcube.total_emission.data.shape)
    rsquared_data[i_valid] = rsquared

    return GenericMap(slopes_data, emcube[0].meta,), GenericMap(rsquared_data, emcube[0].meta)
