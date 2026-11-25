"""Retrospective evaluation after prediction commitment; fixed exploratory families."""
import json
from pathlib import Path
import numpy as np
from .io_utils import save_json, append_csv
from .metrics import rank_ic, mean_defined, paired_test, block_indices, holm


def evaluate_predictions(predictions, manifest, cfg, folder, sigma=1.):
    folder = Path(folder)
    daily = []
    for packet in predictions:
        row, output = packet['row'], packet['prediction']
        for index, method in enumerate(manifest['methods']):
            mu, rank = output[index, :, 0], output[index, :, 1]
            mu_mask, rank_mask = row['mask'][:, 0], row['mask'][:, 1]
            mu_ok = mu_mask.any() and np.isfinite(mu).all()
            rank_ok = (int(rank_mask.sum()) >= cfg['evaluation']['minimum_rank_stocks']
                       and np.isfinite(rank).all())
            daily.append({'method': method, 't': row['t'], 'date': row['date'], 'period': row['segment'],
                          'mu_mse': float(np.mean((mu[mu_mask]-row['truth'][mu_mask, 0])**2)*sigma*sigma) if mu_ok else None,
                          'rank_target_mse': float(np.mean((rank[rank_mask]-row['truth'][rank_mask, 1])**2)) if rank_ok else None,
                          'rank_ic': rank_ic(rank[rank_mask], row['truth'][rank_mask, 0], True) if rank_ok else None,
                          'midrank_ic': rank_ic(rank[rank_mask], row['truth'][rank_mask, 0], False) if rank_ok else None,
                          'rank_ic_mu': rank_ic(mu[rank_mask], row['truth'][rank_mask, 0], True) if rank_ok and mu_ok else None,
                          'origin_stocks': len(row['ids']), 'mu_labels': int(mu_mask.sum()),
                          'rank_labels': int(rank_mask.sum()), 'label_maturity_t': row['t']+cfg['controller']['delay']})
    append_csv(folder/'daily_metrics.csv', daily)
    summary = []
    for period in sorted({r['period'] for r in daily}):
        for method in manifest['methods']:
            subset = [r for r in daily if r['period'] == period and r['method'] == method]
            summary.append({'period': period, 'method': method, 'origins': len(subset),
                            **{key: mean_defined([r[key] for r in subset])
                               for key in ('mu_mse', 'rank_target_mse', 'rank_ic', 'midrank_ic', 'rank_ic_mu')}})
    append_csv(folder/'summary.csv', summary)
    save_json(folder/'summary.json', summary)
    return daily, summary


def infer(daily, statuses, manifest, cfg, folder):
    lookup = {(r['period'], r['method'], r['t']): r for r in daily}
    results = {}
    rng = np.random.default_rng(cfg['random_state'])
    for family in ('rank_family', 'mse_family'):
        out = []
        for spec in manifest[family]:
            endpoint = spec['endpoint']
            channel = 1 if endpoint == 'rank_ic' else 0
            terms = spec.get('terms', [[spec.get('candidate'), 1], [spec.get('control'), -1]])
            needed = [name for name, _ in terms]
            ready = all(statuses[name]['channels'][channel]['status'] == 'completed' for name in needed)
            available_times = sorted({r['t'] for r in daily if r['period'] == spec['period']})
            times = list(range(available_times[0], available_times[-1]+1)) if available_times else []
            values = []
            if ready:
                for t in times:
                    data = [lookup.get((spec['period'], name, t), {}).get(endpoint) for name, _ in terms]
                    if all(v is not None and np.isfinite(v) for v in data):
                        value = sum(weight*v for (_, weight), v in zip(terms, data))
                        values.append(-value if endpoint == 'mu_mse' else value)
                    else:
                        values.append(np.nan)
            indices = block_indices(len(values), cfg['inference'], rng) if len(values) else np.empty((0, 0), int)
            result = paired_test(values, indices, cfg['inference'])
            result.update(spec)
            if not ready:
                result['reason'] = 'failed_or_unidentified_method; registered family slot retained'
            out.append(result)
        holm(out)
        save_json(Path(folder)/(family+'.json'), out)
        results[family] = out
    save_json(Path(folder)/'claim_status.json', {'scope': 'exploratory already-seen development',
              'rank_family_size': len(results['rank_family']), 'mse_family_size': len(results['mse_family']),
              'automatic_winner_selection': False, 'requires_human_result_to_claim_review': True,
              'note': 'Any significant comparison is a comparison-level result, not automatic paper support.'})
    return results


def overlap_diagnostics(predictions, statuses, manifest, cfg, scales, folder):
    from .linear import make_basis, moments
    folder = Path(folder)
    rows = []
    controller_root = folder if (folder/'coefficients.jsonl').exists() else folder.parent/'controllers'
    coefficient_lookup = {}
    if (controller_root/'coefficients.jsonl').exists():
        with (controller_root/'coefficients.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                item = json.loads(line)
                if item['method'] == 'SRT' and 'state_posterior' in item:
                    coefficient_lookup[(item['t'], item['channel'])] = item
    models = {}
    for channel in range(2):
        path = controller_root/'state_models'/('R1_'+str(channel)+'.json')
        if path.exists():
            models[channel] = json.loads(path.read_text(encoding='utf-8'))
    for offset in range(5):
        subset = [p for p in predictions if p['row']['t'] % 5 == offset]
        for channel in range(2):
            gain_values, times, residuals, posterior = [], [], [], []
            for packet in subset:
                row = packet['row']
                if row['segment'] not in ('2020H2', '2021', 'EVAL'):
                    continue
                obs = moments(row, make_basis(row, 'R1'), scales['R1'], channel,
                              cfg['controller']['minimum_feedback_stocks'])
                if obs is not None:
                    gain_values.append(obs['a'])
                    times.append(row['t'])
                    saved = coefficient_lookup.get((row['t'], channel))
                    if saved and channel in models:
                        pi = np.asarray(saved['state_posterior'])
                        predicted_b = pi@np.asarray(models[channel]['b'])
                        residuals.append(obs['a']-obs['B']@predicted_b)
                        posterior.append(pi)
            if gain_values:
                a = np.stack(gain_values)
                autocorrelations = []
                for j in range(a.shape[1]):
                    ok = len(a) > 2 and a[:-1, j].std() > 1e-12 and a[1:, j].std() > 1e-12
                    autocorrelations.append(float(np.corrcoef(a[:-1, j], a[1:, j])[0, 1]) if ok else None)
                residual_ac = []
                if len(residuals) > 2:
                    r = np.stack(residuals)
                    for j in range(r.shape[1]):
                        ok = r[:-1, j].std() > 1e-12 and r[1:, j].std() > 1e-12
                        residual_ac.append(float(np.corrcoef(r[:-1, j], r[1:, j])[0, 1]) if ok else None)
                rows.append({'offset': offset, 'channel': channel, 'dates': len(times),
                             'alignment_lag1_correlation': autocorrelations,
                             'emission_residual_lag1_correlation': residual_ac,
                             'predicted_state_occupancy': np.sum(posterior, axis=0) if posterior else None,
                             'scope': 'all five fixed non-overlapping groups; descriptive only'})
    save_json(folder/'nonoverlap_alignment.json', rows)


def economic_diagnostics(data, predictions, statuses, manifest, cfg, folder):
    from .portfolio import portfolio
    folder = Path(folder)
    # Preserve the same four sleeves continuously across the reporting boundary.
    data.bounds['DEVELOPMENT'] = (data.bounds['2020H2'][0], data.bounds['2021'][1])
    output = {}
    for j, method in enumerate(manifest['methods']):
        relevant = [p for p in predictions if p['row']['segment'] in ('2020H2', '2021')]
        if not relevant or not all(np.isfinite(p['prediction'][j, :, 1]).all() for p in relevant):
            output[method] = {'status': 'unavailable', 'method_status': statuses[method]}
            continue
        scores = {p['row']['t']: (p['row']['ids'], p['prediction'][j, :, 1]) for p in relevant}
        _, result = portfolio(data, scores, 'DEVELOPMENT', cfg, folder/method)
        output[method] = {**result, 'method_status': statuses[method], 'continuous_reporting': True}
    save_json(folder/'economic_summary.json', output)


def figures(daily, summary, folder, teacher_audit=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    captions = []
    for period in ('2020H2', '2021', 'EVAL'):
        subset = [r for r in summary if r['period'] == period]
        if not subset:
            continue
        for endpoint, title in [('mu_mse', 'Return MSE'), ('rank_ic', 'RankIC strict'),
                                ('rank_target_mse', 'Rank-target MSE')]:
            names = [r['method'] for r in subset]
            values = [r[endpoint] if r[endpoint] is not None else np.nan for r in subset]
            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.bar(np.arange(len(names)), values)
            ax.set_xticks(np.arange(len(names)), names, rotation=65, ha='right', fontsize=8)
            ax.set_title(period+' / '+title+' / exploratory')
            ax.set_ylabel(title)
            fig.tight_layout()
            name = period+'_'+endpoint+'.pdf'
            fig.savefig(folder/name)
            plt.close(fig)
            captions.append(name+': all fixed methods; missing results remain blank; consult method_status.json for unidentified models.')
        lookup = {(r['method'], r['t']): r for r in daily if r['period'] == period}
        fig, ax = plt.subplots(figsize=(9, 4))
        for method in ('SRT', 'R1C0', 'RES-RLS', 'OCT-FIX'):
            xs, values = [], []
            for t in sorted({r['t'] for r in daily if r['period'] == period}):
                a, b = lookup.get((method, t)), lookup.get(('D', t))
                if a and b and a['mu_mse'] is not None and b['mu_mse'] is not None:
                    xs.append(t)
                    values.append(b['mu_mse']-a['mu_mse'])
            if values:
                ax.plot(xs, np.cumsum(values), label=method)
        ax.axhline(0, color='black', linewidth=.6)
        ax.set_title(period+' / cumulative date-equal MSE improvement vs D')
        ax.legend()
        fig.tight_layout()
        name = period+'_cumulative_gain.pdf'
        fig.savefig(folder/name)
        plt.close(fig)
        captions.append(name+': descriptive cumulative squared-loss differences, not portfolio returns or a significance test.')
    if teacher_audit and Path(teacher_audit).exists():
        import csv
        with Path(teacher_audit).open(encoding='utf-8') as f:
            audit = list(csv.DictReader(f))
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for c, ax in enumerate(axes):
            selected = [r for r in audit if int(r['channel']) == c]
            ax.scatter([float(r['teacher_gain']) for r in selected],
                       [float(r['student_gain']) for r in selected], s=3, alpha=.2, rasterized=False)
            ax.axhline(0, color='black', linewidth=.6)
            ax.axvline(0, color='black', linewidth=.6)
            ax.set_xlabel('Teacher correction gain')
            ax.set_ylabel('Student response gain')
            ax.set_title('Return' if c == 0 else 'Rank target')
        fig.tight_layout()
        fig.savefig(folder/'teacher_student_gain.pdf')
        plt.close(fig)
        captions.append('teacher_student_gain.pdf: retrospective same-origin/mask diagnostics; future teacher was isolated from deployed decisions.')
    (folder/'CAPTIONS.md').write_text('\n\n'.join(captions)+'\n', encoding='utf-8')
