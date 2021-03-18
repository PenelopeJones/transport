import numpy as np
from scipy.stats import norm, invwishart, multivariate_normal

import pdb

# Compute inter-ion distance given the particle positions
def radial_distance(x_i, y_i, z_i, x_j, y_j, z_j, box_length):
    """
    Calculates the effective distance between two particles using periodic BCs and the
    minimum image convention.

    :param x_i: x_coordinate of ion i (float).
    :param y_i: y_coordinate of ion i (float).
    :param z_i: z_coordinate of ion i (float).
    :param x_j: x_coordinate of ion j (float).
    :param y_j: y_coordinate of ion j (float).
    :param z_j: z_coordinate of ion j (float).
    :param box_length: box size (float).
    :return: distance between the two points (float).
    """
    delta_x = min(((x_i - x_j) % box_length), ((x_j - x_i) % box_length))
    delta_y = min(((y_i - y_j) % box_length), ((y_j - y_i) % box_length))
    delta_z = min(((z_i - z_j) % box_length), ((z_j - z_i) % box_length))
    return np.sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)

# Compute g(r)
def rdf(r, prefactor, bin_size, ion_size, min_r_value, max_r_value, smoothed=False):
    """
    For a given array of central ion A - ion B distances this calculates
    the A-B (smoothed or not smoothed) radial distribution function g(r) as an array.
    :param r: Vector of distances of ions of type A from the central ion of type B.
    :param prefactor: Scalar value equal to the reciprocal of average number density
                      in ideal gas with the same overall density (float).
    :param bin_size: Histogram bin size (float).
    :param ion_size: Size of each ion, used to calculate the smoothed RDF (float).
    :param min_r_value: The minimum x value to be considered in the histogram (float).
    :param max_r_value: The maximum x value to be considered in the histogram (float).
    :param smoothed: If true, the smoothed RDF is calculated; if false, the standard RDF
                     is calculated.
    :return:
    """
    number_of_bins = int((max_r_value - min_r_value) / bin_size)
    gs = []
    lower_r_bound = min_r_value
    upper_r_bound = min_r_value + bin_size
    for i in range(0, number_of_bins):
        r_mean = 0.5 * (lower_r_bound + upper_r_bound)
        V_shell = 4 * np.pi * r_mean ** 2 * bin_size

        if smoothed:
            x = norm.cdf(upper_r_bound, loc=r, scale=ion_size) - \
                norm.cdf(lower_r_bound, loc=r, scale=ion_size)
            number_in_bin = np.sum(x)
        else:
            number_in_bin = ((lower_r_bound < r) & (r < upper_r_bound)).sum()

        g = prefactor * number_in_bin / V_shell
        gs.append(g)
        lower_r_bound = upper_r_bound
        upper_r_bound = upper_r_bound + bin_size
    return gs


def static_feature_vector(typeA, typeB, typeA_id, box_length, bin_size=0.5, ion_size=0.5,
                          min_r_value=0.5, max_r_value=5.0, smoothed=True):
    """
    typeA: np array of positions of type A ions [nA, 3]
    typeB: np array of positions of type B ions [nB, 3]
    typeA_id: int giving the id of the type A ion whose feature vector we are computing
    """
    prefactor_aa = box_length ** 3 / (typeA.shape[0] - 1)
    prefactor_ab = box_length ** 3 / (typeB.shape[0])

    # Step 1: Compute typeA-typeA feature vector
    distances_aa = []
    for j in range(typeA.shape[0]):
        if j == typeA_id:
            continue
        r = radial_distance(typeA[typeA_id, 0], typeA[typeA_id, 1], typeA[typeA_id, 2],
                            typeA[j, 0], typeA[j, 1], typeA[j, 2], box_length)
        distances_aa.append(r)
    distances_aa = np.asarray(distances_aa)
    x_aa = rdf(distances_aa, prefactor_aa, bin_size, ion_size, min_r_value, max_r_value, smoothed)

    # Step 2: Compute typeA-typeB feature vector
    distances_ab = []
    for j in range(typeB.shape[0]):
        r = radial_distance(typeA[typeA_id, 0], typeA[typeA_id, 1], typeA[typeA_id, 2],
                            typeB[j, 0], typeB[j, 1], typeB[j, 2], box_length)
        distances_ab.append(r)
    distances_ab = np.asarray(distances_ab)
    x_ab = rdf(distances_ab, prefactor_ab, bin_size, ion_size, min_r_value, max_r_value, smoothed)

    return np.concatenate((x_aa, x_ab))


# TODO: Write a 'velocity correlation function' descriptor
