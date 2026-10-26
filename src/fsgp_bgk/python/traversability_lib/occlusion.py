# Artificial lidar occlusion (docs/fusion_occlusion_plan.md): the FSGP-BGK paper's simulation condition, "15 % of the
# point cloud data intentionally occluded, mimicking real-world LiDAR limitations such as shadowed regions or sparse
# sensor scans" (Hou et al., §IV-A.2). The paper doesn't say how; two modes:
# - sectors: remove whole azimuth wedges (real gaps, like a shadow or a blocked part of the field of view). k random
#   wedge centres; every point is ranked by its angular distance to the nearest centre and the closest fraction of the
#   points are removed, so all k wedges grow together until exactly that share of the points is gone
# - random: drop that share of the points uniformly at random (only thins the cloud)

import numpy as np

MODES = ('sectors', 'random')


# xyz: (P, 3) points in the sensor / robot frame (azimuth about z). Returns the kept points
def occlude_points(xyz, fraction, rng, mode='sectors', sectors=3):
    n = len(xyz)
    n_remove = int(round(fraction * n))
    if fraction <= 0 or n_remove == 0:
        return xyz
    if mode == 'random':
        drop = rng.choice(n, n_remove, replace=False)
    elif mode == 'sectors':
        centres = rng.uniform(-np.pi, np.pi, size=max(int(sectors), 1))
        azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
        diff = np.abs((azimuth[:, None] - centres[None, :] + np.pi) % (2 * np.pi) - np.pi)
        drop = np.argsort(diff.min(axis=1), kind='stable')[:n_remove]
    else:
        raise ValueError(f'Unknown occlusion mode {mode!r}, choose from {MODES}')
    keep = np.ones(n, dtype=bool)
    keep[drop] = False
    return xyz[keep]
