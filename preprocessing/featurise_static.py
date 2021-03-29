import os
import argparse
import time
import sys
sys.path.append('../../')

import numpy as np
from scipy.stats import norm, invwishart, multivariate_normal

from utils.feature_util import radial_distance, static_feature_vector

import pdb


def main(args):
    ptl = args.path + 'positions/' + args.subdir + '/'

    # Load ion positions
    anion_paths = np.load(ptl + 'anion_positions.npy')
    cation_paths = np.load(ptl + 'cation_positions.npy')

    """
    sigma = 2.5A / 0.97
    cutoff is 1.6*sigma
    residence time of dynamics ion pairs determined by classifying ions as neighbours 
    if they fall within 2.5*sigma of each other. 
    residence time of static ion pairs classified as ions within 1.6*sigma of each other.
    """

    sigma = 1
    r_c_static = 1.6 * sigma
    r_c_dynamic = 2.5 * sigma
    box_length = 50000.0 ** (1 / 3) * sigma

    assert anion_paths.shape == cation_paths.shape
    (n_snapshots, n_anions, _) = anion_paths.shape
    n_cations = cation_paths.shape[1]

    # Identify which ions are classified as being in a static ion pair at each snapshot
    static_pairs_dict = {}
    dynamic_pairs_dict = {}
    x_an_dict = {}
    x_cat_dict = {}

    for snapshot_id in range(args.n_snapshots):

        t0 = time.time()
        # Select ion positions at a given snapshot
        anions = anion_paths[snapshot_id, :, :]
        cations = cation_paths[snapshot_id, :, :]

        static_pairs = []
        dynamic_pairs = []
        x_cat = []
        x_an = []

        for anion_id in range(n_anions):
            for cation_id in range(n_cations):
                r = radial_distance(anions[anion_id, 0], anions[anion_id, 1], anions[anion_id, 2],
                                    cations[cation_id, 0], cations[cation_id, 1], cations[cation_id, 2],
                                    box_length)
                if r <= r_c_dynamic:
                    dynamic_pairs.append((anion_id, cation_id))
                    if r <= r_c_static:
                        static_pairs.append((anion_id, cation_id))

                        # Compute the feature vectors for the two ions in the ion pair i.e. [g_like, g_unlike]
                        x_cat.append(static_feature_vector(cations, anions, cation_id, box_length))
                        x_an.append(static_feature_vector(anions, cations, anion_id, box_length))

        static_pairs = np.vstack(static_pairs)
        dynamic_pairs = np.vstack(dynamic_pairs)
        x_cat = np.vstack(x_cat)
        x_an = np.vstack(x_an)
        static_pairs_dict[snapshot_id] = np.copy(static_pairs)
        dynamic_pairs_dict[snapshot_id] = np.copy(dynamic_pairs)
        x_an_dict[snapshot_id] = np.copy(x_an)
        x_cat_dict[snapshot_id] = np.copy(x_cat)

        if snapshot_id % args.print_freq == 0:
            print('Snapshot {}\t Time: {:.2f}'.format(snapshot_id, (time.time() - t0)))

    length_dict = {}

    # Calculate the number of snapshots for which an ion pair remains paired
    for idx0 in range(args.n_snapshots - args.cutoff):
        pairs0 = static_pairs_dict[idx0]
        length = np.zeros(pairs0.shape[0])
        for i in range(pairs0.shape[0]):
            for idx in range(idx0 + 1, idx0 + args.cutoff):
                match = False
                for j in range(dynamic_pairs_dict[idx].shape[0]):
                    if np.array_equal(pairs0[i], dynamic_pairs_dict[idx][j]):
                        match = True
                if match:
                    length[i] += 1
                else:
                    break

        length_dict[idx0] = length

    # Create numpy matrix
    X = []
    y = []
    for i in range(args.n_snapshots - args.cutoff):
        X.append(np.concatenate((x_an_dict[i], x_cat_dict[i]), axis=1))
        y.append(length_dict[i])

    # Save as .npy files
    X = np.vstack(X)
    y = np.hstack(y)

    if not os.path.exists(args.pts):
        os.makedirs(args.pts)

    np.save(args.pts + 'X_{}.npy'.format(args.subdir), X)
    np.save(args.pts + 'y_{}.npy'.format(args.subdir), y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='../../data/md-trajectories/',
                        help='Path to directory containing data.')
    parser.add_argument('--pts', type=str, default='../../data/processed/',
                        help='Path to directory containing data.')
    parser.add_argument('--subdir', type=str, default='5',
                        help='Sub directory of interest.')
    parser.add_argument('--print_freq', type=int, default=5,
                        help='Print every N snapshots.')
    parser.add_argument('--n_snapshots', type=int, default=200,
                        help='Number of snapshots.')
    parser.add_argument('--cutoff', type=int, default=50,
                        help='Expected maximum number of snapshots over which ions are paired.')

    args = parser.parse_args()

    main(args)