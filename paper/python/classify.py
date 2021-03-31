"""
Apply random forest classifier to time lag, cross-correlation and slope data
"""
import copy
import os

import numpy as np
import astropy.units as u
from sunpy.map import Map
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from synthesizAR.analysis.dem import EMCube

from utils import make_slope_map


def prep_data(top_dir, channel_pairs, heating,
              correlation_threshold=0.1, rsquared_threshold=0.75,):
    """
    Import and reshape model and observational data for training

    This function imports the time lag, cross-correlation, and EM slope data,
    finds the pixels where any of these quantities is invalid, and reshapes the
    valid data into an n-by-p data matrix that can be passed to scikit-learn.
    It does this for both the simulation and model data.

    Parameters
    ----------
    top_dir : `str`
    channel_pairs : `list`
    heating : `list`
    correlation_threshold : `float`, optional
    rsquared_threshold : `float`, optional

    Returns
    -------
    X : `~numpy.NDArray`
        Simulated data matrix with shape `(31,n)`, where `n` is the total
        number of pixels in the simulated image times the number of heating
        frequencies minus the number of bad pixels
    Y : `~numpy.NDArray`
        Heating frequency labels for each pixel with shape (1,n)
    X_observation : `~numpy.NDarray`
        Observed data matrix with shape `(31,m)`, where `m` is the total
        number of pixels in the real image minus the number of bad
        pixels
    bad_pixels : `~numpy.NDArray`
        Composite mask for the observed data with shape `(mx,my)`
    """
    file_format = os.path.join(top_dir, '{}', '{}_{}_{}.fits')

    # Calculate slope maps for model results and observations
    # the r^2. If the r^2 value is below the given threshold or
    # it is infinite or NaN, mask that pixel
    slope_maps = {}
    tpeak_maps = {}
    for h in heating + ['observations']:
        em_threshold = 1e27 * u.cm**(-5) if h == 'observations' else 1e24 * u.cm**(-5)
        em_cube = EMCube.restore(os.path.join(top_dir, h, 'em_cube.h5'))
        tpeak_maps[h] = Map(em_cube.temperature_bin_centers[np.argmax(em_cube.as_array(), axis=2)],
                            copy.deepcopy(em_cube.all_meta()[0]))
        s, r2 = make_slope_map(
            em_cube,
            em_threshold=em_threshold,
            temperature_lower_bound=8e5*u.K,
        )
        mask = ~np.logical_and(r2.data >= rsquared_threshold, np.isfinite(r2.data))
        slope_maps[h] = Map(s.data, s.meta, mask=mask)

    # Create mask for simulated data from cross-correlations and r^2 values to EM slopes
    # If any pixel is masked in any frequency, the pixel is masked in all frequencies
    correlations = np.stack([
        Map(file_format.format(h, 'correlation', *cp)).data for h in heating
        for cp in channel_pairs
    ])
    # If cross-correlation less than threshold in any channel pair or slope is masked,
    # it is bad
    bad_pixels = np.stack((
        (correlations < correlation_threshold).any(axis=0,),
        np.stack([slope_maps[h].mask for h in heating]).any(axis=0),
    )).any(axis=0)

    # Load, flatten, and stack the timelags, cross-correlation values and slopes for
    # all three simulated datasets, excluding the bad pixels
    X_timelag = np.stack([
        np.hstack([
            Map(file_format.format(h, 'timelag', *cp)).data[np.where(~bad_pixels)].flatten()
            for h in heating])
        for cp in channel_pairs
    ], axis=1)
    X_correlation = np.stack([
        np.hstack([
            Map(file_format.format(h, 'correlation', *cp)).data[np.where(~bad_pixels)].flatten()
            for h in heating])
        for cp in channel_pairs
    ], axis=1)
    X_slope = np.hstack([slope_maps[h].data[np.where(~bad_pixels)].flatten() for h in heating])
    X_slope = X_slope[:, np.newaxis]
    X_tpeak = np.hstack([tpeak_maps[h].data[np.where(~bad_pixels)].flatten() for h in heating])
    X_tpeak = X_tpeak[:, np.newaxis]
    X = np.hstack((X_timelag, X_correlation, X_slope, X_tpeak))

    # Load heating frequency labels into something the same shape as our data
    Y = np.hstack([np.where(~bad_pixels)[0].shape[0]*[h] for h in heating])

    # Create mask for real data from cross-correlations and r^2 values to EM slopes
    correlations = np.stack([
        Map(file_format.format('observations', 'correlation', *cp)).data
        for cp in channel_pairs
    ])
    # If cross-correlation less than threshold in any channel pair or slope
    # is below given threshold, it is bad
    bad_pixels = np.stack((
        (correlations < correlation_threshold).any(axis=0),
        slope_maps['observations'].mask,
    )).any(axis=0)

    # Load, flatten, and stack timelags, cross-correlation values and slopes for
    # observations, excluding the bad pixels
    X_timelag = np.stack([
        Map(file_format.format('observations', 'timelag', *cp)).data[np.where(~bad_pixels)].flatten()
        for cp in channel_pairs
    ], axis=1)
    X_correlation = np.stack([
        Map(file_format.format('observations', 'correlation', *cp)).data[np.where(~bad_pixels)].flatten()
        for cp in channel_pairs
    ], axis=1)
    X_slope = slope_maps['observations'].data[np.where(~bad_pixels)].flatten()
    X_slope = X_slope[:, np.newaxis]
    X_tpeak = tpeak_maps['observations'].data[np.where(~bad_pixels)].flatten()
    X_tpeak = X_tpeak[:, np.newaxis]
    X_observation = np.hstack((X_timelag, X_correlation, X_slope, X_tpeak))

    return X, Y, X_observation, bad_pixels


def classify_ar(classifier_params, X_model, Y_model, X_observation, bad_pixels, **kwargs):
    """
    Train random forest classifier on simulation data, apply to real data

    Parameters
    ----------
    classifier_params {[type]} -- [description]
    X_model {[type]} -- [description]
    Y_model {[type]} -- [description]
    X_observation {[type]} -- [description]
    bad_pixels {[type]} -- [description]
    """
    # Encode labels
    le = LabelEncoder()
    le.fit(Y_model)
    Y_model = le.transform(Y_model)
    # Split training and test data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_model, Y_model, test_size=kwargs.get('test_size', 0.33))
    # Fit classifier
    clf = RandomForestClassifier(**classifier_params)
    clf.fit(X_train, Y_train)
    test_error = 1. - clf.score(X_test, Y_test)
    # Classify observations
    Y_observation = clf.predict(X_observation)
    Y_observation_prob = clf.predict_proba(X_observation)
    # Frequency map
    data = np.empty(bad_pixels.shape)
    data[bad_pixels] = np.nan
    data[~bad_pixels] = Y_observation
    class_map = data.copy()
    # Probability maps
    probability_maps = {}
    for i, c in enumerate(le.inverse_transform(clf.classes_)):
        data = np.empty(bad_pixels.shape)
        data[bad_pixels] = np.nan
        data[~bad_pixels] = Y_observation_prob[:, i]
        probability_maps[c] = data.copy()

    return class_map, probability_maps, clf, test_error
