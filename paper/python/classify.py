"""
Apply random forest classifier to timelag, cross-correlation and slope data
"""
import os

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.map import Map, GenericMap
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, scale
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def prep_data(top_dir, channel_pairs, heating, correlation_threshold=0.1):
    """
    Import and reshape model and observational data

    Parameters
    ----------
    top_dir : `str`
    channel_pairs : `list`
    heating : `list`
    correlation_threshold : `float`, optional
    """
    file_format = os.path.join(top_dir, '{}', '{}_{}_{}.fits')
    # Create mask for training data
    all_correlations = np.stack([Map(file_format.format(h, 'correlation', *cp)).data
                                 for h in heating for cp in channel_pairs])
    correlation_mask = (all_correlations < correlation_threshold).any(axis=0,)
    all_slopes = np.stack([Map(os.path.join(top_dir, f'{h}', 'em_slope.fits')).data
                          for h in heating])
    slope_mask = np.isnan(all_slopes).any(axis=0)
    bad_pixels = np.stack((correlation_mask, slope_mask),).any(axis=0)
    # Load all three training datasets
    X_timelag = np.stack([np.hstack(
        [Map(file_format.format(h, 'timelag', *cp)).data[np.where(~bad_pixels)].flatten()
         for h in heating]) for cp in channel_pairs], axis=1)
    X_correlation = np.stack([np.hstack(
        [Map(file_format.format(h, 'correlation', *cp)).data[np.where(~bad_pixels)].flatten()
         for h in heating]) for cp in channel_pairs], axis=1)
    X_slope = np.hstack(
        [Map('../paper/data/{}/em_slope.fits'.format(h)).data[np.where(~bad_pixels)].flatten()
         for h in heating])
    X_slope = X_slope[:, np.newaxis]
    X = np.hstack((X_timelag, X_correlation, X_slope))
    # Load labels
    Y = np.hstack([np.where(~bad_pixels)[0].shape[0]*[h] for h in heating])
    # Create mask for real data
    all_correlations = np.stack([Map(file_format.format('observations', 'correlation', *cp)).data
                                 for cp in channel_pairs])
    correlation_mask = (all_correlations < correlation_threshold).any(axis=0,)
    slope_mask = np.isnan(Map(os.path.join(top_dir, 'observations', 'em_slope.fits')).data)
    bad_pixels = np.stack((correlation_mask, slope_mask),).any(axis=0)
    # Load all three real datasets
    X_timelag = np.stack(
        [Map(file_format.format('observations', 'timelag', *cp)).data[np.where(~bad_pixels)].flatten()
         for cp in channel_pairs], axis=1)
    X_correlation = np.stack(
        [Map(file_format.format('observations', 'correlation', *cp)).data[np.where(~bad_pixels)].flatten()
         for cp in channel_pairs], axis=1)
    X_slope = Map('../paper/data/observations/em_slope.fits').data[np.where(~bad_pixels)].flatten()
    X_slope = X_slope[:, np.newaxis]
    X_observation = np.hstack((X_timelag, X_correlation, X_slope))

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
    # Scale and split training data
    X_scaled = scale(X_model, axis=0, with_mean=True, with_std=True,)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, Y_model, test_size=kwargs.get('test_size', 0.33))
    # Scale observed data
    X_observation_scaled = scale(X_observation, axis=0, with_mean=True, with_std=True)
    # Fit classifier
    clf = RandomForestClassifier(**classifier_params)
    clf.fit(X_train, Y_train)
    test_error = 1. - clf.score(X_test, Y_test)
    # Classify observations
    Y_observation = clf.predict(X_observation_scaled)
    Y_observation_prob = clf.predict_proba(X_observation_scaled)
    # Frequency map
    data = np.empty(bad_pixels.shape)
    data[bad_pixels] = np.nan
    data[~bad_pixels] = Y_observation
    frequency_map = data.copy()
    # Probability maps
    probability_maps = {}
    for i, c in enumerate(le.inverse_transform(clf.classes_)):
        data = np.empty(bad_pixels.shape)
        data[bad_pixels] = np.nan
        data[~bad_pixels] = Y_observation_prob[:, i]
        probability_maps[c] = data.copy()

    return frequency_map, probability_maps, clf, test_error