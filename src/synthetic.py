"""Eight mechanism classes, ten fixed panels, controlled student spaces; no neural fit."""
import copy
import hashlib
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from .bank import fit_dictionary, save_bank
from .metrics import standardized_rank, mean_defined
from .replay import replay
from .analysis import evaluate_predictions, overlap_diagnostics
from .io_utils import save_json, append_csv


def generate(case, variant, cfg):
    s = cfg['synthetic']
    counts = [4*s['environment_dates'], s['head_dates'], s['reliability_dates'],
              s['warm_dates'], s['evaluation_dates']]
    ends = np.cumsum(counts)
    total, n = int(ends[-1]), s['stocks']
    code = int(hashlib.sha256((case+variant+str(cfg['random_state'])).encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(code)
    x = rng.normal(size=(total, n, 2))
    if case == 'covariate_only':
        x[ends[3]:] *= 2
    market = .4*np.sin(np.arange(total)/17)+rng.normal(scale=.2, size=total)
    x1, x2 = x[:, :, 0], x[:, :, 1]
    shared = np.stack([x1, x2, np.tanh(x1*x2), x1*x1-1, x2*x2-1,
                       np.broadcast_to(market[:, None], (total, n))], axis=2)
    mean = x1+.4*x2+.15*np.tanh(x1*x2)
    for e in range(4):
        sl = slice(e*s['environment_dates'], (e+1)*s['environment_dates'])
        mean[sl] = (.8+.2*e)*x1[sl]+(.4-.1*e)*x2[sl]+.15*(-1)**e*np.tanh(x1[sl]*x2[sl])
    hidden = np.zeros(total, dtype=int)
    if case == 'teacher_projection':
        mean = x1+x2
        mean[:ends[0]] = -.2*x1[:ends[0]]+x2[:ends[0]]
    elif case == 'recurrent_sign':
        hidden = (np.arange(total)//48) % 3
        coefficient = np.asarray([-1., .2, 1.])[hidden]
        mean = coefficient[:, None]*x1+.4*x2
    elif case == 'unforeseen_flip':
        flip = int(ends[3]+counts[4]//2)
        mean[flip:] = -x1[flip:]+.4*x2[flip:]
        hidden[flip:] = 1
    elif case == 'common_market':
        mean = x1+1.5*market[:, None]
    elif case == 'rank_mean_mismatch':
        mean = x1+.15*x2
    elif case == 'null':
        mean = .6*x1+.2*x2
    noise = rng.normal(scale=.5, size=(total, n))
    if case == 'rank_mean_mismatch':
        skew = np.where(rng.random((total, n)) < .9, .6, -5.4)
        noise += skew*(x2+2)
    if case == 'null' and variant == 'overlap':
        innovations = rng.normal(scale=.5, size=(total+5, n))
        noise = sum(innovations[h:h+total] for h in (2, 3, 4, 5))/2
    y = mean+noise
    teacher = (.3*x1+.05*x2)
    if case == 'common_market':
        teacher = x1.copy()
    if case == 'teacher_projection':
        teacher = np.zeros_like(y)
    if case == 'null':
        teacher = mean.copy()
    rows = []
    for t in range(total):
        if t < ends[0]:
            segment = 'B'
            env, j = divmod(t, s['environment_dates'])
            date = (datetime.fromisoformat(cfg['environments'][env][0])+timedelta(days=j)).date().isoformat()
        elif t < ends[1]:
            segment, date = 'C', '2016-SYN-'+str(t-int(ends[0])).zfill(4)
        elif t < ends[2]:
            segment, date = 'REL', '2019-SYN-'+str(t-int(ends[1])).zfill(4)
        elif t < ends[3]:
            segment, date = 'WARM', '2020-SYN-'+str(t-int(ends[2])).zfill(4)
        else:
            segment, date = 'EVAL', '2021-SYN-'+str(t-int(ends[3])).zfill(4)
        truth = np.column_stack([y[t], standardized_rank(y[t], True)])
        taught = np.column_stack([teacher[t], standardized_rank(teacher[t], True)])
        mask = np.ones((n, 2), dtype=bool)
        # Same frozen parameter-fitting data across all three feedback delays.
        if segment == 'REL' and t+max(s['delays']) >= ends[2]:
            mask[:] = False
        origin_h = shared[t]
        h = shared[t]
        if case == 'teacher_projection':
            h = x1[t, :, None]
        if case == 'null' and variant == 'collinear':
            h = x1[t, :, None]
            origin_h = h
        rows.append({'t': t, 'date': date, 'segment': segment, 'ids': np.arange(n),
                     'h': h, 'origin_h': origin_h, 'native': np.zeros((n, 2)), 'T0': taught,
                     'teacher': taught.copy(), 'truth': truth, 'mask': mask,
                     'training_mask': mask.copy(), 'known_mean': mean[t], 'hidden_state': hidden[t]})
    return rows


def run_synthetic(cfg, manifest, run):
    start = time.monotonic()
    root = Path(run)/'evidence/synthetic'
    root.mkdir(parents=True, exist_ok=True)
    states, all_summary = {}, []
    for case in cfg['synthetic']['cases']:
        variants = cfg['synthetic']['null_variants'] if case == 'null' else ['default']
        for variant in variants:
            name = case+'_'+variant
            folder = root/name
            folder.mkdir(parents=True, exist_ok=True)
            try:
                rows = generate(case, variant, cfg)
                scales, parameters = fit_dictionary(rows, cfg, folder)
                for row in rows:
                    # Controlled fixed student representation: native and matched direct coincide.
                    row['native'] = row['D'].copy()
                paired_gain = []
                eta = cfg['training']['distillation_weight']
                for row in rows:
                    if row['segment'] != 'EVAL':
                        continue
                    for channel in range(2):
                        y = row['truth'][:, channel]
                        teacher = row['teacher'][:, channel]
                        for env in range(4):
                            corrected = teacher+row['raw_c'][:, channel, env]/eta
                            fk, fe = row['K'][:, channel], row['E'][:, channel, env]
                            paired_gain.append({'t': row['t'], 'channel': channel, 'environment': env,
                                                'teacher_gain': float(np.mean((y-teacher)**2-(y-corrected)**2)),
                                                'student_gain': float(np.mean((y-fk)**2-(y-fe)**2))})
                append_csv(folder/'teacher_student_same_origin.csv', paired_gain)
                save_bank(rows, folder/'frozen_bank')
                state_cache = {}
                for delay in cfg['synthetic']['delays']:
                    settings = copy.deepcopy(cfg)
                    settings['controller']['delay'] = delay
                    target = folder/('delay_'+str(delay))
                    pred, status = replay(rows, scales, settings, manifest, target, state_cache)
                    daily, summary = evaluate_predictions(pred, manifest, settings, target)
                    overlap_diagnostics(pred, status, manifest, settings, scales, target)
                    for method_index, method in enumerate(manifest['methods']):
                        eval_pred = [p for p in pred if p['row']['segment'] == 'EVAL']
                        losses = [float(np.mean((p['prediction'][method_index, :, 0]-p['row']['known_mean'])**2))
                                  for p in eval_pred if np.isfinite(p['prediction'][method_index, :, 0]).all()]
                        matched = next(r for r in summary if r['period'] == 'EVAL' and r['method'] == method)
                        all_summary.append({'case': case, 'variant': variant, 'delay': delay,
                                            **matched, 'known_mean_mse': mean_defined(losses),
                                            'mu_status': status[method]['channels'][0]['status'],
                                            'rank_status': status[method]['channels'][1]['status']})
                states[name] = {'status': 'completed', 'controller_unidentified_outputs_preserved': True,
                                'dictionary_seconds': parameters['elapsed_seconds']}
                print({'event': 'synthetic_complete', 'case': name}, flush=True)
            except Exception as exc:
                states[name] = {'status': 'failed', 'error': repr(exc), 'traceback': traceback.format_exc()}
                save_json(folder/'failure.json', states[name])
                print(traceback.format_exc(), flush=True)
    save_json(root/'status.json', states)
    save_json(root/'summary.json', all_summary)
    append_csv(root/'summary.csv', all_summary)
    save_json(root/'scope.json', {'classes': 8, 'panels': 10, 'delays': cfg['synthetic']['delays'],
              'neural_fits': 0, 'elapsed_seconds': time.monotonic()-start,
              'scope': 'controlled fixed function spaces and known synthetic conditional means; not neural-training evidence',
              'hidden_state_used_by_controller': False, 'adverse_accuracy_blocks_financial_run': False})
    return states
