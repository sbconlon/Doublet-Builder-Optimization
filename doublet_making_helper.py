"""
This file contains useful helper functions for the doublet making functions.
"""

import numpy as np
import pandas as pd
import trackml.dataset
from numba import jit, guvectorize, prange
from numba import int64, float32, boolean


@jit(nopython=True)
def filter_layers(layer_id, layer_range, verbose=False):
	return contains(layer_id, layer_range)

@jit(nopython=True)
def filter_phi(outer_phi, inner_phi, nPhiSlices):
	return ((outer_phi - 1) == inner_phi or
		    (outer_phi + 1) == inner_phi or
			 outer_phi == inner_phi or
		    (outer_phi == 0 and inner_phi == nPhiSlices - 2) or
		    (outer_phi == nPhiSlices - 2 and inner_phi == 0))

@jit(nopython=True)
def filter_doublet_length(inner_r, outer_r, minDoubletLength, maxDoubletLength):
	return (((outer_r - inner_r) < maxDoubletLength) and ((outer_r - inner_r) > minDoubletLength))

@jit(nopython=True)
def filter_horizontal_doublets(inner_r, inner_z, outer_r, outer_z, maxCtg):
	return np.abs((outer_z - inner_z)/(outer_r - inner_r)) < maxCtg

@jit(nopython=True)
def filter_z(outer_layer, outer_z, layer_range, z_ranges):
	return (outer_z > z_ranges[outer_layer][0] and outer_z < z_ranges[outer_layer][1])

@jit(nopython=True)
def get_layer_range(inner_hit, layer_radii, nLayers, maxDoubletLength, FALSE_INT):
	'''
	This function, given a inner hit, returns a list of layers that may contain valid outer hits
	'''
	valid_layers = []
	for layer_id in range(nLayers):
		if (layer_id == inner_hit[1]+1 or layer_id == inner_hit[1]+2 or
		    layer_id == inner_hit[1]-1 or layer_id == inner_hit[1]-2):
			valid_layers.append(layer_id)
		else:
			valid_layers.append(FALSE_INT)
	return valid_layers

@jit(nopython=True)
def get_z_ranges(inner_hit, refCoords, layer_range, zMinus, zPlus, FALSE_INT):
	'''
	This function, given an inner hit, calculates the z region of interest for all valid layers in layer_range
	'''
	z_ranges = np.zeros((len(layer_range), 2), dtype=int64)
	for idx in range(len(layer_range)):
		if layer_range[idx] == FALSE_INT:
			z_ranges[idx][0], z_ranges[idx][1] = FALSE_INT, FALSE_INT
		else:
			z_minus = zMinus + refCoords[idx] * (inner_hit[4] - zMinus) // inner_hit[3]
			z_plus  = zPlus  + refCoords[idx] * (inner_hit[4] - zPlus) // inner_hit[3]
			z_ranges[idx][0], z_ranges[idx][1] = min(z_minus, z_plus), max(z_minus, z_plus)
	return z_ranges

@jit(nopython=True)
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

@jit(nopython=True)
def contains(val: int64, lst: int64[:]):
	'''
	This function is the numba friendly implementation of the contains built in funtion in python
	'''
	for elem in lst:
		if val == elem:
			return True
	return False

@jit(nopython=True)
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

@jit(nopython=True)
def zip(x, y):
	'''
	This function is the numba friendly implementation of the zip function in python
	'''
	zipped = np.array([])
	for idx in range(len(x)):
		zipped = append(zipped, np.array([x[idx], y[idx]]))
	return zipped

@jit(nopython=True)
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