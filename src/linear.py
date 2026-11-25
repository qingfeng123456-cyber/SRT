"""Date-equal ridge sufficient statistics and common response coordinates."""
import numpy as np


def normalization(rows, key, floor):
    usable = [np.asarray(r[key], dtype=float) for r in rows if len(r[key])]
    mean = np.mean([x.mean(0) for x in usable], axis=0)
    second = np.mean([(x*x).mean(0) for x in usable], axis=0)
    scale = np.maximum(np.sqrt(np.maximum(second-mean*mean, 0)), floor)
    return mean, scale


def design(raw, mean, scale, channel):
    x = (np.asarray(raw, dtype=float)-mean)/scale
    x = np.column_stack([np.ones(len(x)), x])
    if channel == 1:
        x = x-x.mean(0)
    return x


def fit_ridge(rows, x_key, target_fn, masks_fn, penalty, norm):
    """All outputs in one solve; every date has equal total stock weight."""
    mean, scale = norm
    p = len(mean)+1
    first = np.asarray(target_fn(rows[0]))
    if first.ndim == 2:
        first = first[:, :, None]
    outputs = first.shape[2]
    gram = np.zeros((2, p, p))
    cross = np.zeros((2, p, outputs))
    counts = np.zeros(2, dtype=int)
    for row in rows:
        target = np.asarray(target_fn(row), dtype=float)
        if target.ndim == 2:
            target = target[:, :, None]
        masks = np.asarray(masks_fn(row), dtype=bool)
        for channel in range(2):
            mask = masks[:, channel]
            if not mask.any():
                continue
            x = design(row[x_key], mean, scale, channel)[mask]
            y = target[mask, channel]
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise FloatingPointError('Nonfinite ridge data')
            gram[channel] += x.T@x/len(x)
            cross[channel] += x.T@y/len(x)
            counts[channel] += 1
    coefficients = np.zeros_like(cross)
    for channel in range(2):
        if counts[channel] == 0:
            raise ValueError('No observed dates for ridge channel')
        reg = penalty*np.eye(p)
        if channel == 0:
            reg[0, 0] = 0
        coefficients[channel] = np.linalg.solve(gram[channel]/counts[channel]+reg,
                                                 cross[channel]/counts[channel])
    return {'mean': mean, 'scale': scale, 'coefficients': coefficients,
            'date_counts': counts, 'penalty': penalty}


def predict_ridge(raw, model):
    return np.stack([design(raw, model['mean'], model['scale'], c)@model['coefficients'][c]
                     for c in range(2)], axis=1)


def column_scales(rows, basis_fn, floor):
    squares = np.mean([np.mean(basis_fn(r)**2, axis=0) for r in rows], axis=0)
    return np.maximum(np.sqrt(squares), floor)


def make_basis(row, name):
    # All arrays N x channel x direction; columns are deployable outputs.
    delta0 = (row['K']-row['D'])[:, :, None]
    if name == 'R1':
        return np.concatenate([delta0, row['E']-row['K'][:, :, None]], axis=2)
    if name == 'R0':
        return np.concatenate([delta0, row['raw_c']], axis=2)
    if name == 'RES':
        return np.concatenate([delta0, row['res_E']], axis=2)
    if name == 'GATE':
        return (row['E'].mean(2)-row['D'])[:, :, None]
    if name == 'H':
        return np.stack([design(row['h'], row['h_mean'], row['h_scale'], c)
                         for c in range(2)], axis=1)
    raise KeyError(name)


def moments(row, basis, scales, channel, minimum):
    mask = row['mask'][:, channel]
    if int(mask.sum()) < minimum:
        return None
    d = basis[:, channel]/scales[channel]
    r = row['truth'][:, channel]-row['D'][:, channel]
    x = d[mask]
    return {'t': int(row['t']), 'a': x.T@r[mask]/len(x), 'B': x.T@x/len(x),
            'n': len(x), 'release': int(row['t'])+5}
