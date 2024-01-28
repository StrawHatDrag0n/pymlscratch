import numpy as np
from scipy import stats

def calculate_mode(x):
    return stats.mode(x).mode

def calculate_mean(x):
    return np.mean(x)