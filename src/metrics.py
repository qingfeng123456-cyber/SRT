"""Ground truth ranks, fixed date-block tests, and conservative missing slots."""
import numpy as np
from scipy.stats import rankdata


def standardized_rank(values, zero_if_constant=False):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        return np.zeros(len(values)) if zero_if_constant else None
    ranks = rankdata(values, method='average')
    std = ranks.std()
    if std <= 1e-12:
        return np.zeros(len(values)) if zero_if_constant else None
    return (ranks-ranks.mean())/std


def rank_ic(scores, truth, strict=True):
    v = standardized_rank(truth)
    if v is None or not np.isfinite(scores).all():
        return None
    if strict:
        # Input stocks are in stable ID order. mergesort gives ID tie breaking.
        order = np.argsort(scores, kind='stable')
        ranks = np.empty(len(order), dtype=float)
        ranks[order] = np.arange(len(order))
        u = (ranks-ranks.mean())/ranks.std()
    else:
        u = standardized_rank(scores)
        if u is None:
            return None
    return float(np.mean(u*v))


def mean_defined(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def block_indices(length, settings, rng):
    block, repeats = settings['block_length'], settings['replicates']
    starts = rng.integers(0, length, size=(repeats, int(np.ceil(length/block))))
    return ((starts[:, :, None]+np.arange(block)) % length).reshape(repeats, -1)[:, :length]


def paired_test(delta, indices, settings):
    delta = np.asarray(delta, dtype=float)
    observed_mask = np.isfinite(delta)
    count = int(observed_mask.sum())
    if count < 2*settings['block_length']:
        return {'status': 'unavailable', 'dates': count, 'calendar_origins': len(delta), 'mean_improvement': None,
                'ci_low': None, 'ci_high': None, 'p_value': None, 'holm_p_value': None}
    observed = float(delta[observed_mask].mean())
    centered_values = np.where(observed_mask, delta-observed, 0.)
    counts = observed_mask[indices].sum(1)
    good = counts > 0
    centered = centered_values[indices].sum(1)[good]/counts[good]
    alpha = (1-settings['confidence'])/2
    lo, hi = np.quantile(centered, [alpha, 1-alpha], method='linear')
    p = (1+int((np.abs(centered) >= abs(observed)).sum()))/(len(centered)+1)
    return {'status': 'estimated_exploratory', 'dates': count, 'calendar_origins': len(delta),
            'bootstrap_draws_used': len(centered), 'mean_improvement': observed,
            'ci_low': float(observed-hi), 'ci_high': float(observed-lo), 'p_value': p,
            'holm_p_value': None}


def holm(rows):
    # Missing tests occupy their registered family slots; display them as unavailable.
    order = sorted(range(len(rows)), key=lambda i: rows[i]['p_value'] if rows[i]['p_value'] is not None else 1.)
    previous = 0.
    for position, i in enumerate(order):
        p = rows[i]['p_value'] if rows[i]['p_value'] is not None else 1.
        previous = max(previous, min(1., (len(rows)-position)*p))
        if rows[i]['p_value'] is not None:
            rows[i]['holm_p_value'] = previous
