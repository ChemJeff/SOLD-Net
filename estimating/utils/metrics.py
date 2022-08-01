import numpy as np


def calc_azimuth_error(pred, gt, unit='deg'):
    assert unit in ['deg', 'rad']
    cycle = 360 if unit == 'deg' else np.pi*2
    candidates = np.zeros(3)
    candidates[0] = pred - (gt - cycle)
    candidates[1] = pred - gt
    candidates[2] = pred - (gt + cycle)
    minabspos = np.argmin(abs(candidates))
    return candidates[minabspos]