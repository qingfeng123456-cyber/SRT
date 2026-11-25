"""Predict-only views and a single delayed feedback ledger shared by all controls."""
import json
import time
import traceback
from pathlib import Path
import numpy as np
from .linear import make_basis, moments
from .controllers import MomentControl, Hedge, solve_coeff
from .state import fit_state, GainFilter
from .io_utils import save_json, clean, append_csv


STATE_METHODS = {'R0C1': 'R0', 'SRT': 'R1', 'RES-STATE': 'RES',
                 'SRT-NOPROP': 'R1', 'SRT-NOSIGNED': 'R1', 'SRT-OLDGEOM': 'R1'}
MOMENT_METHODS = {'R0C0': 'R0', 'R1C0': 'R1', 'R1-ROLL': 'R1', 'RES-RLS': 'H', 'GATE': 'GATE'}
VIEW_KEYS = ('t', 'D', 'K', 'E', 'res_E', 'raw_c', 'native', 'h', 'h_mean', 'h_scale')


def replay(rows, scales, cfg, manifest, folder, state_cache=None):
    started = time.monotonic()
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'predictions').mkdir(exist_ok=True)
    (folder/'matured_moments').mkdir(exist_ok=True)
    methods = manifest['methods']
    records = [r for r in rows if r['segment'] in ('REL', 'CHECK', 'WARM', '2020H2', '2021', 'EVAL')]
    records.sort(key=lambda r: r['t'])
    rel = [r for r in records if r['segment'] == 'REL']
    deployment = [r for r in records if r['segment'] != 'REL']
    if not rel or not deployment:
        raise ValueError('Need frozen REL and subsequent deployment records')
    start_t = deployment[0]['t']
    delay = cfg['controller']['delay']
    if max(r['t']+delay for r in rel if r['mask'].any()) > start_t:
        raise ValueError('Controller fit labels not matured at deployment start')
    observations = {}
    for row in records:
        item = {}
        for basis in ('R0', 'R1', 'RES', 'GATE', 'H'):
            value = make_basis(row, basis)
            for channel in range(2):
                o = moments(row, value, scales[basis], channel,
                            cfg['controller']['minimum_feedback_stocks'])
                if o is not None:
                    o['release'] = row['t']+delay
                    item[(basis, channel)] = o
        observations[row['t']] = item
    state_models, state_filters, state_diagnostics = {}, {}, {}
    for basis in ('R0', 'R1', 'RES'):
        for channel in range(2):
            key = (basis, channel)
            training = [observations[r['t']][key] for r in rel if key in observations[r['t']]]
            ts = time.monotonic()
            reused = state_cache is not None and key in state_cache
            print({'event': 'gain_state_start', 'basis': basis, 'channel': channel,
                   'dates': len(training), 'reused': reused}, flush=True)
            try:
                if state_cache is not None and key in state_cache:
                    entry = state_cache[key]
                    if 'failure' in entry:
                        raise ValueError('Shared state fit previously failed: '+entry['failure'])
                    model = entry['model']
                else:
                    model = fit_state(training, cfg)
                    if state_cache is not None:
                        state_cache[key] = {'model': model}
                state_models[key] = model
                state_filters[key] = GainFilter(model)
                state_diagnostics[str(key)] = {'status': model['identification'],
                                               'reused_frozen_fit': reused, 'state_fit_calls': 0 if reused else 1,
                                               'elapsed_seconds': time.monotonic()-ts}
                save_json(folder/'state_models'/(basis+'_'+str(channel)+'.json'), model)
            except Exception as exc:
                if state_cache is not None and key not in state_cache:
                    state_cache[key] = {'failure': repr(exc)}
                state_diagnostics[str(key)] = {'status': 'failed', 'error': repr(exc),
                                               'reused_frozen_fit': reused, 'state_fit_calls': 0 if reused else 1,
                                               'traceback': traceback.format_exc(),
                                               'elapsed_seconds': time.monotonic()-ts}
    save_json(folder/'state_fit_status.json', state_diagnostics)
    controls = {(name, c): MomentControl(scales[basis].shape[1], cfg, name == 'R1-ROLL')
                for name, basis in MOMENT_METHODS.items() for c in range(2)}
    hedges = [Hedge(cfg), Hedge(cfg)]
    statuses = {name: {'channels': [{'status': 'completed'}, {'status': 'completed'}]} for name in methods}
    for name, basis in STATE_METHODS.items():
        for channel in range(2):
            model = state_models.get((basis, channel))
            statuses[name]['channels'][channel] = ({'status': 'failed', 'reason': 'state_fit_failed'}
                if model is None else {'status': 'completed' if model['identification'] == 'identified'
                                      else 'completed_unidentified',
                                      'reasons': model['identification_reasons']})
    costs = {name: 0. for name in methods}
    feedback_pointer, latest = 0, {}
    predictions, ledger = [], []
    coefficients_path = folder/'coefficients.jsonl'
    with coefficients_path.open('w', encoding='utf-8') as coeff_stream:
        for position, row in enumerate(deployment):
            t = row['t']
            # Release each stored origin once; no recomputation using current heads.
            while feedback_pointer < len(records) and records[feedback_pointer]['t']+delay <= t:
                old = records[feedback_pointer]
                released = old['t']+delay
                observed = observations[old['t']]
                payload = {}
                for (basis, channel), o in observed.items():
                    latest[(basis, channel)] = o
                    payload[basis+'_'+str(channel)+'_a'] = o['a']
                    payload[basis+'_'+str(channel)+'_B'] = o['B']
                    if (basis, channel) in state_filters:
                        state_filters[(basis, channel)].update(o)
                for (name, channel), control in controls.items():
                    key = (MOMENT_METHODS[name], channel)
                    if key in observed:
                        control.update(observed[key], released)
                for channel in range(2):
                    mask = old['mask'][:, channel]
                    if mask.sum() >= cfg['controller']['minimum_feedback_stocks']:
                        experts = np.column_stack([old['D'][:, channel], old['K'][:, channel], old['E'][:, channel]])
                        losses = np.mean((experts[mask]-old['truth'][mask, channel, None])**2, axis=0)
                        hedges[channel].update(losses)
                np.savez_compressed(folder/'matured_moments'/(str(old['t'])+'.npz'),
                                    origin=old['t'], released_at=released, **payload)
                ledger.append({'origin_t': old['t'], 'released_at': released,
                               'first_processed_at': t, 'lag': delay,
                               'mu_labels': int(old['mask'][:, 0].sum()),
                               'rank_labels': int(old['mask'][:, 1].sum())})
                feedback_pointer += 1
            view = {key: row[key] for key in VIEW_KEYS}
            bases = {key: make_basis(view, key)/scales[key][None, :, :]
                     for key in ('R0', 'R1', 'RES', 'GATE', 'H')}
            output = np.full((len(methods), len(row['ids']), 2), np.nan)
            for method_index, name in enumerate(methods):
                for channel in range(2):
                    if statuses[name]['channels'][channel]['status'] == 'failed':
                        continue
                    begin = time.monotonic()
                    detail = {'t': t, 'method': name, 'channel': channel,
                              'latest_released_origin': records[feedback_pointer-1]['t'] if feedback_pointer else None}
                    try:
                        if name in ('D-ALL', 'D', 'K'):
                            key = 'native' if name == 'D-ALL' else name
                            forecast = view[key][:, channel]
                        elif name == 'OCT-FIX':
                            forecast = view['E'][:, channel].mean(1)
                        elif name == 'HEDGE':
                            experts = np.column_stack([view['D'][:, channel], view['K'][:, channel], view['E'][:, channel]])
                            forecast = experts@hedges[channel].weights
                            detail['expert_weights'] = hedges[channel].weights
                        elif name in MOMENT_METHODS:
                            basis = MOMENT_METHODS[name]
                            B, a = controls[(name, channel)].statistics(t)
                            cap = float(scales['GATE'][channel, 0]) if name == 'GATE' else None
                            alpha, numerical = solve_coeff(B, a, cfg, nonnegative=name == 'GATE', radius=cap)
                            forecast = view['D'][:, channel]+bases[basis][:, channel]@alpha
                            detail.update(numerical)
                            detail.update({'coefficient': alpha, 'basis': basis})
                        else:
                            basis = STATE_METHODS[name]
                            mean, variance, pi = state_filters[(basis, channel)].predict(t, name != 'SRT-NOPROP')
                            d = bases[basis][:, channel]
                            B = d.T@d/len(d)
                            if name == 'SRT-OLDGEOM' and (basis, channel) in latest:
                                B = latest[(basis, channel)]['B']
                            alpha, numerical = solve_coeff(B, B@mean, cfg, variance,
                                                           nonnegative=name == 'SRT-NOSIGNED')
                            forecast = view['D'][:, channel]+d@alpha
                            detail.update(numerical)
                            detail.update({'coefficient': alpha, 'state_posterior': pi, 'bbar': mean,
                                           'distance_from_bbar': float(np.linalg.norm(alpha-mean)),
                                           'filtered_origin': state_filters[(basis, channel)].last,
                                           'basis': basis})
                        if not np.isfinite(forecast).all():
                            raise FloatingPointError('Nonfinite forecast')
                        if channel == 1:
                            forecast = forecast-forecast.mean()
                        output[method_index, :, channel] = forecast
                        detail['output_shift_rms'] = float(np.sqrt(np.mean((forecast-view['D'][:, channel])**2)))
                    except Exception as exc:
                        statuses[name]['channels'][channel] = {'status': 'failed', 't': t,
                                                               'error': repr(exc), 'traceback': traceback.format_exc()}
                        save_json(folder/'failures'/(name+'_'+str(channel)+'.json'), statuses[name]['channels'][channel])
                    finally:
                        costs[name] += time.monotonic()-begin
                    coeff_stream.write(json.dumps(clean(detail), ensure_ascii=False, allow_nan=False)+'\n')
            np.savez_compressed(folder/'predictions'/(str(t)+'.npz'), t=t, ids=row['ids'], predictions=output)
            predictions.append({'row': row, 'prediction': output})
            if position % 100 == 0:
                print({'event': 'shared_control_replay', 'dates': position+1,
                       'date': row['date'], 'feedback_origins': feedback_pointer}, flush=True)
    append_csv(folder/'feedback_ledger.csv', ledger)
    save_json(folder/'method_status.json', statuses)
    save_json(folder/'costs.json', {'elapsed_seconds': time.monotonic()-started,
              'prediction_seconds_by_method': costs, 'state_fits': state_diagnostics,
              'matured_origins': feedback_pointer, 'prediction_origins': len(predictions),
              'cost_scope': 'prediction dispatch excludes shared moment preparation and feedback; total includes them'})
    return predictions, statuses
