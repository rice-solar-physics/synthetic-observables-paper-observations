"""
This provides a quick estimate of the bounds on the mean waiting times for each
heating heating frequency for the longest and shortest loops in the model AR. The
data and scripts needed to run this can be found in the repo for the first paper,
Barnes, Bradshaw, and Viall (2019): https://github.com/rice-solar-physics/synthetic-observables-paper-models.
"""
import sys
import copy

import numpy as np
import astropy.units as u

import synthesizAR

# NOTE: you need to have the appropriate version of the AR skeleton and then
# set the path here. You'll also need the wait-time-dependent heating model
# and then set the path for that.
heating_model_path = 'scripts'
skeleton_path = 'paper/data/base_noaa1158'

sys.path.append(heating_model_path)
from constrained_heating_model import CustomHeatingModel


base_heating_options = {
    'duration': 200.0,
    'duration_rise': 100.0,
    'duration_decay': 100.0,
    'stress_level': 1.,
    'power_law_slope': -2.5,
}

frequencies = [0.1, 1, 5]

skeleton = synthesizAR.Field.restore(skeleton_path)

all_lengths = u.Quantity([l.full_length for l in skeleton.loops])
loop_min = skeleton.loops[np.argmin(all_lengths)]
loop_max = skeleton.loops[np.argmax(all_lengths)]

for l in [loop_min, loop_max]:
    print('------------')
    print(l.full_length.to('Mm'))
    for f in frequencies:
        conf = copy.deepcopy(base_heating_options)
        conf['frequency_parameter'] = f
        print('---------------')
        print('Frequency = ', f)
        h = CustomHeatingModel(conf)
        t_cool = h.cooling_time(l)
        print('Mean waiting time = ', f*t_cool, ' s')
