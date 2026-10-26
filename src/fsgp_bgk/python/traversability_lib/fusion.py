# Multi-frame fusion of each map cell's risk factor samples (docs/fusion_occlusion_plan.md)
# Input for C cells observed by up to F frames: factors (F, C, N, 3) joint samples of (slope, flatness, step height),
# NaN where a frame didn't observe the cell, and the frames' times (F,). Output: fused samples (C, N, 3) (NaN where no
# frame observed the cell) and the number of frames per cell. A cell seen by one frame keeps that frame's samples.
# Traversability samples are then w . factors with the usual weights, so every scenario downstream reads the same data:
# - fuse_paper: FSGP-BGK paper eqs 9-10 (Hou et al. 2025), applied as written per factor, frame by frame in time order;
#   samples = the newest frame's, shifted / scaled per factor to the fused mean / variance. The variance update is an
#   average (never shrinks), and the confidence weight 1 / (sigma^2_prev + eps) is unit-dependent
# - fuse_information: information-form (precision-weighted) Gaussian fusion (Thrun, Burgard & Fox 2005, §3.4-3.5;
#   per-cell Kalman fusion as in Fankhauser et al. 2014): Sigma_f^-1 = sum w_i Sigma_i^-1, mu_f = Sigma_f sum w_i
#   Sigma_i^-1 mu_i; samples = all frames' samples pooled (weights w_i), resampled, whitened by the pool's mean /
#   covariance and re-coloured to (mu_f, Sigma_f), so the shape (skew, tails) comes from the real samples while the
#   mean and covariance match the fused estimate exactly (up to clipping to [0, 1])
# Frames sharing points aren't independent, so the information form can be overconfident (conservative alternative:
# covariance intersection, Julier & Uhlmann 1997)
# Time decay w = exp(-(t_ref - t) / decay_time); decay_time None = no decay (a static survey's frames are taken minutes
# apart but describe the same, unchanging scene)

import numpy as np

PAPER_EPS = 1e-6        # eps in the paper's confidence weight 1 / (sigma^2 + eps)
COV_JITTER = 1e-6       # added to every covariance diagonal before inverting / Cholesky (saturated factors have 0 spread)


def _decay(times, t_ref, decay_time):
    if decay_time is None or decay_time <= 0:
        return np.ones_like(times, dtype=np.float64)
    return np.exp(-(t_ref - times) / decay_time)


def _observed(factors):
    return np.isfinite(factors).all(axis=(2, 3))         # (F, C)


def _restyle(samples, mean, var):
    # Shift / scale each factor's samples (C, N, 3) to the given per-factor mean / variance (C, 3); zero-spread factors
    # become the mean
    s_mean = samples.mean(axis=1, keepdims=True)
    s_std = samples.std(axis=1, ddof=1, keepdims=True)
    z = np.where(s_std > 0, (samples - s_mean) / np.where(s_std > 0, s_std, 1.0), 0.0)
    return mean[:, None, :] + z * np.sqrt(np.maximum(var, 0.0))[:, None, :]


def fuse_paper(factors, times, decay_time=None):
    factors = np.asarray(factors, dtype=np.float64)
    F, C, N, K = factors.shape
    obs = _observed(factors)
    order = np.argsort(times, kind='stable')
    M = np.full((C, K), np.nan)
    V = np.full((C, K), np.nan)
    t_prev = np.full(C, np.nan)
    newest = np.full((C, N, K), np.nan)
    for f in order:
        cells = obs[f]
        mu = factors[f, cells].mean(axis=1)
        var = factors[f, cells].var(axis=1, ddof=1)
        first = cells & np.isnan(t_prev)
        later = cells & ~np.isnan(t_prev)
        sub_first = first[cells]
        M[first], V[first] = mu[sub_first], var[sub_first]
        if later.any():
            sub = later[cells]
            w_t = _decay(t_prev[later], times[f], decay_time)[:, None] if decay_time else np.ones((later.sum(), 1))
            w_s = 1.0 / (V[later] + PAPER_EPS)
            M[later] = (w_t * w_s * M[later] + mu[sub]) / (w_t * w_s + 1.0)        # eq. 9
            V[later] = (w_t * V[later] + var[sub]) / (w_t + 1.0)                     # eq. 10
        t_prev[cells] = times[f]
        newest[cells] = factors[f, cells]
    n_frames = obs.sum(axis=0)
    fused = np.full((C, N, K), np.nan)
    seen = n_frames > 0
    fused[seen] = np.clip(_restyle(newest[seen], M[seen], V[seen]), 0.0, 1.0)
    single = n_frames == 1
    fused[single] = newest[single]
    return fused, n_frames


def fuse_information(factors, times, decay_time=None, seed=0):
    factors = np.asarray(factors, dtype=np.float64)
    F, C, N, K = factors.shape
    obs = _observed(factors)
    n_frames = obs.sum(axis=0)
    w = _decay(np.asarray(times, dtype=np.float64), float(np.max(times)), decay_time)     # (F,)
    eye = np.eye(K)

    # Per-frame mean and covariance (unobserved: identity, weight 0)
    filled = np.where(obs[:, :, None, None], factors, 0.0)
    mu = filled.mean(axis=2)                                                # (F, C, K)
    centred = filled - mu[:, :, None, :]
    cov = np.einsum('fcnk,fcnl->fckl', centred, centred) / (N - 1) + COV_JITTER * eye
    cov[~obs] = eye
    prec = np.linalg.inv(cov)
    weight = (w[:, None] * obs)[:, :, None, None]                          # (F, C, 1, 1)
    P = (weight * prec).sum(axis=0)                                         # (C, K, K)
    P[n_frames == 0] = eye
    cov_f = np.linalg.inv(P)
    mu_f = np.einsum('ckl,cl->ck', cov_f, (weight[..., 0] * np.einsum('fckl,fcl->fck', prec, mu)).sum(axis=0))

    rng = np.random.default_rng(seed)
    fused = np.full((C, N, K), np.nan)
    for c in np.flatnonzero(n_frames >= 1):
        frames = np.flatnonzero(obs[:, c])
        if len(frames) == 1:
            fused[c] = factors[frames[0], c]
            continue
        pool = factors[frames, c].reshape(-1, K)                            # (n*N, K)
        p = np.repeat(w[frames], N)
        p = p / p.sum()
        draw = pool[rng.choice(len(pool), N, replace=True, p=p)]
        # Whiten by the draw's own mean / covariance, so the output has exactly (mu_f, Sigma_f) up to clipping
        m_d = draw.mean(axis=0)
        L_d = np.linalg.cholesky(np.cov(draw.T) + COV_JITTER * eye)
        L_f = np.linalg.cholesky(cov_f[c] + COV_JITTER * eye)
        z = np.linalg.solve(L_d, (draw - m_d).T)
        fused[c] = np.clip(mu_f[c] + (L_f @ z).T, 0.0, 1.0)                 # re-colour to the fused estimate
    return fused, n_frames


FUSIONS = {'bgk_paper': fuse_paper, 'information': fuse_information}


# Live fusion for the FSGP-BGK node (docs/fusion_occlusion_plan.md, phase 2 step 3). Each frame's factor grid (map-
# aligned, robot-centred, (nx, ny, 3N) as node_ros2's global_factor_grid: slope_1..N, flatness_1..N, step_1..N, NaN =
# no factor data) is keyed by world cell (round((pose + cell offset) / resolution)); each world cell keeps its last
# `history` frames' samples and the published grid gets the fused samples. No time decay within the short history;
# frames are fused oldest to newest. Cells further than prune_radius from the robot are forgotten
class LiveFusion:
    def __init__(self, method, history, resolution, prune_radius):
        if method not in FUSIONS:
            raise ValueError(f'Unknown fusion method {method!r}, choose from {list(FUSIONS)}')
        from collections import deque
        self._deque = deque
        self.fuse = FUSIONS[method]
        self.history = int(history)
        self.resolution = float(resolution)
        self.prune_radius = float(prune_radius)
        self.cells = {}                     # (ix, iy) -> deque of (N, 3) samples

    def update(self, factor_grid, cell_dx, cell_dy, pose_xy):
        nx, ny, n3 = factor_grid.shape
        n = n3 // 3
        valid = np.isfinite(factor_grid).all(axis=-1)
        if not valid.any():
            return factor_grid, np.zeros((nx, ny), dtype=np.int16)
        wx = pose_xy[0] + cell_dx[valid]
        wy = pose_xy[1] + cell_dy[valid]
        keys = list(zip(np.rint(wx / self.resolution).astype(int).tolist(), np.rint(wy / self.resolution).astype(int).tolist()))
        samples = factor_grid[valid].reshape(-1, 3, n).transpose(0, 2, 1)         # (C, N, 3)
        for key, s in zip(keys, samples):
            store = self.cells.get(key)
            if store is None:
                store = self.cells[key] = self._deque(maxlen=self.history)
            store.append(s)
        # Forget cells the robot has left behind
        r_cells = self.prune_radius / self.resolution
        cx, cy = pose_xy[0] / self.resolution, pose_xy[1] / self.resolution
        current = set(keys)     # never drop a cell this frame observed (the factor grid can reach the window corners)
        for key in [k for k in self.cells if k not in current and (k[0] - cx) ** 2 + (k[1] - cy) ** 2 > r_cells ** 2]:
            del self.cells[key]
        # (F, C, N, 3), newest frame last for every cell
        depth = max(len(self.cells[k]) for k in keys)
        stack = np.full((depth, len(keys), n, 3), np.nan)
        for c, key in enumerate(keys):
            store = self.cells[key]
            stack[depth - len(store):, c] = np.stack(store)
        fused, n_frames = self.fuse(stack, np.arange(depth, dtype=np.float64))
        out = factor_grid.copy()
        out[valid] = fused.transpose(0, 2, 1).reshape(len(keys), n3)
        frames = np.zeros((nx, ny), dtype=np.int16)
        frames[valid] = n_frames
        return out, frames
