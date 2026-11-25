"""Frozen financial features, historical correctors and matched ridge responses."""
import time
from pathlib import Path
import numpy as np
import torch
from .neural import tensor
from .linear import normalization, fit_ridge, predict_ridge, column_scales, make_basis
from .io_utils import save_json


@torch.no_grad()
def teacher_forecast(model, data, t, cfg, audit=False, released_at=None):
    predictions, available = [], []
    x = tensor(data.window(t), cfg)
    for k in cfg['training']['future_k']:
        slots, prefix, valid = (data.audit_future(t, k, released_at) if audit else data.future(t, k))
        out = model(x, tensor(slots, cfg), k).cpu().numpy().astype(float)
        out[:, 0] += prefix
        out[:, 0] /= data.final_sigma
        out[:, 1] -= out[:, 1].mean()
        predictions.append(out)
        available.append(valid)
    weights = np.asarray(cfg['training']['future_weights'])
    return np.einsum('k,kij->ij', weights, predictions), np.logical_and.reduce(available)


@torch.no_grad()
def feature_rows(data, models, cfg):
    rows = []
    segments = ['B', 'C', 'REL', 'CHECK', 'WARM', '2020H2', '2021']
    for segment in segments:
        begin, end = data.bounds[segment]
        for t in range(begin, end):
            if len(data.cohorts[t]) < cfg['data']['minimum_stocks']:
                continue
            x = tensor(data.window(t), cfg)
            h = models['D-ALL'].encode(x)
            origin_h = models['T0'].encode(x)
            native = models['D-ALL'].output(h)*models['D-ALL'].output_scale
            origin = models['T0'].output(origin_h)*models['T0'].output_scale
            native, origin = native.cpu().numpy().astype(float), origin.cpu().numpy().astype(float)
            for output in (native, origin):
                output[:, 0] /= data.final_sigma
                output[:, 1] -= output[:, 1].mean()
            z = data.label(t, segment)
            row = {'t': int(t), 'date': str(data.days[t]), 'segment': segment,
                   'ids': data.cohorts[t].copy(), 'h': h.cpu().numpy(),
                   'origin_h': origin_h.cpu().numpy(), 'native': native, 'T0': origin,
                   'truth': np.column_stack([z['y']/data.final_sigma, z['v']]),
                   'mask': np.column_stack([z['mask'], z['rank_mask']]),
                   'training_mask': np.column_stack([z['mask'], z['rank_mask']])}
            if segment in ('B', 'C') and t+5 < data.bounds[segment][1]:
                value, valid = teacher_forecast(models['TF'], data, t, cfg)
                row['teacher'] = value
                row['training_mask'] = row['training_mask'] & valid[:, None]
            else:
                row['teacher'] = np.zeros_like(row['truth'])
                row['training_mask'] = np.zeros_like(row['mask'])
            rows.append(row)
            if len(rows) % 100 == 0:
                print({'event': 'frozen_features', 'dates': len(rows), 'latest': row['date']}, flush=True)
    return rows


def fit_dictionary(rows, cfg, folder):
    started = time.monotonic()
    folder = Path(folder)
    B = [r for r in rows if r['segment'] == 'B']
    C = [r for r in rows if r['segment'] == 'C' and r['training_mask'].any()]
    if not B or not C:
        raise ValueError('Empty historical or head fitting segment')
    teacher_norm = normalization(B, 'origin_h', cfg['ridge']['scale_floor'])
    head_norm = normalization(C, 'h', cfg['ridge']['scale_floor'])
    future_correctors, ordinary_correctors, counts = [], [], []
    for begin, end in cfg['environments']:
        candidates = [r for r in B if begin <= r['date'] <= end]
        # The next environment may not contribute labels to this environment.
        env_stop = max(r['t'] for r in candidates)+1
        chosen = [r for r in candidates if r['t']+5 < env_stop and r['training_mask'].any()]
        if len(chosen) < 40:
            raise ValueError('Insufficient purged dates in fixed environment '+begin)
        future_correctors.append(fit_ridge(chosen, 'origin_h', lambda r: r['truth']-r['teacher'],
                                           lambda r: r['training_mask'], cfg['ridge']['penalty'], teacher_norm))
        ordinary_correctors.append(fit_ridge(chosen, 'origin_h', lambda r: r['truth']-r['T0'],
                                             lambda r: r['training_mask'], cfg['ridge']['penalty'], teacher_norm))
        counts.append({'begin': begin, 'end': end, 'dates': len(chosen),
                       'last_target_t': max(r['t']+5 for r in chosen)})
    eta = cfg['training']['distillation_weight']
    for row in rows:
        # R0 carries the same eta as the actual corrected-supervision response.
        row['raw_c'] = eta*np.concatenate([predict_ridge(row['origin_h'], c) for c in future_correctors], axis=2)
        row['ordinary_c'] = eta*np.concatenate([predict_ridge(row['origin_h'], c) for c in ordinary_correctors], axis=2)

    def targets(row):
        q = (1-eta)*row['truth']+eta*row['teacher']
        return np.concatenate([row['truth'][:, :, None], q[:, :, None], q[:, :, None]+row['raw_c']], axis=2)

    heads = fit_ridge(C, 'h', targets, lambda r: r['training_mask'], cfg['ridge']['penalty'], head_norm)
    ordinary_projection = fit_ridge(C, 'h', lambda r: r['ordinary_c'], lambda r: r['training_mask'],
                                    cfg['ridge']['penalty'], head_norm)
    for row in rows:
        prediction = predict_ridge(row['h'], heads)
        row['D'], row['K'], row['E'] = prediction[:, :, 0], prediction[:, :, 1], prediction[:, :, 2:]
        row['res_E'] = predict_ridge(row['h'], ordinary_projection)
        row['h_mean'], row['h_scale'] = head_norm
    scales = {key: column_scales(C, lambda r, key=key: make_basis(r, key),
                                cfg['ridge']['response_scale_floor'])
              for key in ('R0', 'R1', 'RES', 'GATE', 'H')}
    result = {'heads': heads, 'future_correctors': future_correctors,
              'ordinary_correctors': ordinary_correctors, 'ordinary_projection': ordinary_projection,
              'scales': scales, 'environments': counts, 'head_dates': len(C),
              'elapsed_seconds': time.monotonic()-started,
              'financial_statistical_fits': '8 dual correctors +6 matched dual heads +4 ordinary response projections',
              'return_unit': 'A-fit final_sigma; convert squared loss by sigma^2',
              'rank_unit': 'standardized observed-subset midrank; origin-universe centering',
              'ordinary_dictionary': 'eta times origin-teacher historical residual projected into same student H'}
    save_json(folder/'dictionary_parameters.json', result)
    return scales, result


def save_bank(rows, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fields = ('t', 'date', 'segment', 'ids', 'h', 'origin_h', 'native', 'T0', 'teacher', 'truth', 'mask',
              'training_mask', 'D', 'K', 'E', 'res_E', 'raw_c', 'ordinary_c')
    for row in rows:
        extra = {key: row[key] for key in ('known_mean', 'hidden_state') if key in row}
        np.savez_compressed(folder/(str(row['t'])+'.npz'), **{key: row[key] for key in fields}, **extra)


@torch.no_grad()
def audit_teacher_student(rows, models, data, cfg, folder):
    """Runs only after deployment forecasts/controllers are committed to disk."""
    from .io_utils import append_csv
    folder = Path(folder)
    audit = []
    (folder/'retrospective_targets').mkdir(parents=True, exist_ok=True)
    eta = cfg['training']['distillation_weight']
    for row in rows:
        if row['segment'] not in ('REL', 'CHECK', 'WARM', '2020H2', '2021') or row['t']+5 >= len(data.days):
            continue
        teacher, valid = teacher_forecast(models['TF'], data, row['t'], cfg, True, row['t']+5)
        correction = row['raw_c']/eta
        np.savez_compressed(folder/'retrospective_targets'/(str(row['t'])+'.npz'),
                            teacher=teacher, available=valid, release=row['t']+5)
        for channel in range(2):
            mask = row['mask'][:, channel] & valid
            if not mask.any():
                continue
            y = row['truth'][mask, channel]
            for env in range(4):
                base_teacher = teacher[mask, channel]
                corrected = base_teacher+correction[mask, channel, env]
                fk, fe, direct = row['K'][mask, channel], row['E'][mask, channel, env], row['D'][mask, channel]
                audit.append({'date': row['date'], 't': row['t'], 'segment': row['segment'],
                              'channel': channel, 'environment': env, 'observed': int(mask.sum()),
                              'teacher_gain': float(np.mean((y-base_teacher)**2-(y-corrected)**2)),
                              'student_gain': float(np.mean((y-fk)**2-(y-fe)**2)),
                              'gain_vs_direct': float(np.mean((y-direct)**2-(y-fe)**2)),
                              'diagnostic_only': True})
    append_csv(folder/'teacher_student_same_origin.csv', audit)
    return {'rows': len(audit), 'retrospective_only': True, 'controller_access': False}
