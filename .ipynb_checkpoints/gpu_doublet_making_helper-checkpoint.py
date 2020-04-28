"""
This file contains useful helper functions for the doublet making functions.
"""

import numpy as np
import cupy as cp

def filter_layers(layer_ids, layer_range):
    """
    Returns boolean value indicating whether or not the hit is in the layer range
    """
    return cp.isin(layer_ids, layer_range)

def filter_phi(outer_phi, inner_phi, nPhiSlices):
    return (((inner_phi - 1) == outer_phi) |
            ((inner_phi + 1) == outer_phi) |
             (inner_phi == outer_phi) |
            ((inner_phi == 0) & outer_phi == nPhiSlices - 2) |
            ((inner_phi == nPhiSlices - 2) & outer_phi == 0))


def filter_doublet_length(inner_r, outer_r, minDoubletLength, maxDoubletLength):
    return (((outer_r - inner_r) < maxDoubletLength) & ((outer_r - inner_r) > minDoubletLength))


def filter_horizontal_doublets(inner_r, inner_z, outer_r, outer_z, maxCtg):
    return cp.abs((outer_z - inner_z)/(outer_r - inner_r)) < maxCtg


def filter_z(outer_layer, outer_z, layer_range, z_ranges):
    return (outer_z > z_ranges[outer_layer][:, 0]) & (outer_z < z_ranges[outer_layer][:, 1])


def get_layer_range():
    '''
    This function, given a inner hit, returns a list of layers that may contain valid outer hits
    '''
    layer_range = cp.empty((gpu_hit_table.shape[0], 4), dtype=cp.int32)
    layer_range[:, 0] = gpu_hit_table[:, 1] + 1
    layer_range[:, 1] = gpu_hit_table[:, 1] + 2
    layer_range[:, 2] = gpu_hit_table[:, 1] - 1
    layer_range[:, 3] = gpu_hit_table[:, 1] - 2

    layers = cp.arange(nLayers)
    layer_range[~ cp.isin(layer_range, layers)] = FALSE_INT


def get_z_ranges():
    '''
    This function, given an inner hit, calculates the z region of interest for all valid layers in layer_range
    '''
    z_ranges = cp.empty((gpu_hit_table.shape[0], 2*nLayers))
    # A loop over 2*(# of layers), not great but somewhat unavoidable
    for i in cp.arange(z_ranges.shape[1], step=2):
        z_ranges[:, i]   = zMinus + gpu_refCoords[i//2] * (gpu_hit_table[:, 4] - zMinus) // gpu_hit_table[:, 3]
        z_ranges[:, i+1] = zPlus  + gpu_refCoords[i//2] * (gpu_hit_table[:, 4] - zPlus) // gpu_hit_table[:, 3]
        z_ranges[:, i], z_ranges[:, i+1] = cp.amin(z_ranges[:, i:i+2], axis=1), cp.amax(z_ranges[:, i:i+2], axis=1)


def z_mask(layer_range, z_ranges, layerModels, FALSE_INT):
    '''
    This function sets the elements in layer_range and z_ranges to FLASE_INT if their corresponding layer geometries
    are outside the z range
    '''
    for idx in range(len(layer_range)):
        if not layer_range[idx] == FALSE_INT:
            if not (layerModels[layer_range[idx]][3] > z_ranges[idx][0] and layerModels[layer_range[idx]][2] < z_ranges[idx][1]):
                layer_range[idx] = FALSE_INT
                z_ranges[idx][0] = FALSE_INT
                z_ranges[idx][1] = FALSE_INT


def contains(val, lst):
    '''
    This function is the numba friendly implementation of the contains built in funtion in python
    '''
    for elem in lst:
        if val == elem:
            return True
    return False


def where(val, lst):
    '''
    This function is the numba friendly implementation of the where function in numpy
    '''
    index = 0
    for elem in lst:
        if val == elem:
            return index
        index += 1
    return index #will cause out of range error


def zip(x, y):
    '''
    This function is the numba friendly implementation of the zip function in python
    '''
    zipped = np.array([])
    for idx in range(len(x)):
        zipped = append(zipped, np.array([x[idx], y[idx]]))
    return zipped


def append(arr, val):
    '''
    This function is the numba friendly implementation of the append function in numpy
    '''
    new_arr = np.zeros(len(arr) + 1, dtype=int64)
    for idx in range(len(arr)):
        new_arr[idx] = arr[idx]
    new_arr[idx+1] = val
    return new_arr

def bin_phi(x, y, nbins):
    phi = np.arctan2(y, x)
    for idx in range(len(phi)):
        if phi[idx] < 0:
            phi[idx] += 2*np.pi
    return (phi * float(nbins-1) / (2*np.pi)).astype(np.int)

def approx_doublet_length(nhits):
    a3 = (-1.073) * pow(10, -9)
    a2 = (1.616)  * pow(10, -3)
    a1 = (-8.490) * pow(10,-2)
    a0 = 9000
    return int(a3 * pow(nhits,3) + a2 * pow(nhits,2) + a1 * nhits + a0)


def triple_space():
    print()
    print()
    print()