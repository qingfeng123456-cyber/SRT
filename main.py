"""Single AutoDL entry: checks, fixed synthetic mechanisms, three financial fits."""
import argparse
import gc
import importlib
import json
import os
import platform
import shutil
import sys
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from src.io_utils import save_json

ROOT = Path(__file__).resolve().parent


class Tee:
    def __init__(self, original, stream):
        self.original, self.stream = original, stream

    def write(self, value):
        self.original.write(value)
        self.stream.write(value)
        self.stream.flush()
        return len(value)

    def flush(self):
        self.original.flush()
        self.stream.flush()


def snapshot(run):
    target = run/'code'
    target.mkdir()
    for name in ('main.py', 'config.json', 'experiment_manifest.json', 'requirements.txt', 'README.md', 'LICENSE', 'CITATION.cff'):
        shutil.copyfile(ROOT/name, target/name)
    for name in ('src', 'tools', 'docs'):
        shutil.copytree(ROOT/name, target/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))


def archive(run):
    path = ROOT/(run.name+'-evidence.zip')
    with zipfile.ZipFile(path, 'x', zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for file in sorted(run.rglob('*')):
            if file.is_file() and '__pycache__' not in file.parts and '.tmp' not in file.name:
                z.write(file, (Path(run.name)/file.relative_to(run)).as_posix())
    return path


def environment(cfg):
    info = {'python': sys.version, 'platform': platform.platform(), 'runtime_location': 'AutoDL'}
    for name in ('numpy', 'scipy', 'torch', 'matplotlib'):
        info[name] = importlib.import_module(name).__version__
    import torch
    if not cfg['device'].startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('Select an AutoDL CUDA PyTorch image; dependencies are not installed automatically.')
    torch.cuda.set_device(cfg['device'])
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    info.update({'gpu': torch.cuda.get_device_name(cfg['device']), 'cuda': torch.version.cuda,
                 'gpu_memory_bytes': torch.cuda.get_device_properties(cfg['device']).total_memory})
    return info


def financial(cfg, manifest, run, state):
    from src.data import MarketData
    from src.neural import fit
    from src.bank import feature_rows, fit_dictionary, save_bank, audit_teacher_student
    from src.replay import replay
    from src.analysis import evaluate_predictions, infer, overlap_diagnostics, economic_diagnostics, figures
    from src.runtime_checks import check_clock
    data = MarketData(ROOT, cfg, run/'evidence/data')
    check_clock(data, cfg, run/'evidence/data_clock_checks.json')
    models = {}
    for name in ('T0', 'TF', 'D-ALL'):
        state['fits'][name] = {'status': 'running'}
        save_json(run/'run_state.json', state)
        try:
            model, result = fit(name, data, cfg, run)
            models[name], state['fits'][name] = model, result
        except Exception as exc:
            state['fits'][name] = {'status': 'failed', 'error': repr(exc), 'traceback': traceback.format_exc()}
            save_json(run/'evidence/failures'/(name+'.json'), state['fits'][name])
            print(traceback.format_exc(), flush=True)
        save_json(run/'run_state.json', state)
    if len(models) != 3:
        raise RuntimeError('Shared financial dependency failed; no replacement or retry')
    start = time.monotonic()
    rows = feature_rows(data, models, cfg)
    save_json(run/'evidence/feature_cache_cost.json', {'seconds': time.monotonic()-start, 'dates': len(rows)})
    scales, dictionary = fit_dictionary(rows, cfg, run/'evidence/dictionary')
    save_bank(rows, run/'evidence/frozen_bank')
    predictions, statuses = replay(rows, scales, cfg, manifest, run/'evidence/controllers')
    # All prediction files and controller states are committed before this audit.
    audit_start = time.monotonic()
    audit = audit_teacher_student(rows, models, data, cfg, run/'evidence/audit')
    save_json(run/'evidence/audit/scope.json', {**audit, 'seconds': time.monotonic()-audit_start,
              'prediction_commitment_precedes_audit': True})
    for key in list(models):
        del models[key]
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    daily, summary = evaluate_predictions(predictions, manifest, cfg, run/'evidence/analysis', data.final_sigma)
    infer(daily, statuses, manifest, cfg, run/'evidence/analysis')
    overlap_diagnostics(predictions, statuses, manifest, cfg, scales, run/'evidence/analysis')
    economic_diagnostics(data, predictions, statuses, manifest, cfg, run/'evidence/economic')
    figures(daily, summary, run/'evidence/figures', run/'evidence/audit/teacher_student_same_origin.csv')
    state['method_status'] = statuses
    state['dictionary'] = {'seconds': dictionary['elapsed_seconds'], 'head_dates': dictionary['head_dates']}
    return all(c['status'] == 'completed' for s in statuses.values() for c in s['channels'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(ROOT/'config.json'))
    parser.add_argument('--sanity-only', action='store_true', help='AutoDL-only checks and all fixed synthetic mechanisms')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    manifest = json.loads((ROOT/'experiment_manifest.json').read_text(encoding='utf-8'))
    if cfg['confirmation']['enabled'] or any(cfg[k] for k in ('automatic_dependency_installation', 'automatic_retry', 'automatic_sweep')):
        raise ValueError('Confirmation, automatic installation/retry/sweep are disabled')
    if cfg['training']['future_k'] != [1, 3] or cfg['controller']['delay'] != 5:
        raise ValueError('Registered financial protocol requires k1/k3 and five-date release')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    run = ROOT/'log'/('RUN-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'-SRT')
    run.mkdir(parents=True)
    snapshot(run)
    save_json(run/'effective_config.json', cfg)
    save_json(run/'execution_manifest.json', manifest)
    save_json(run/'evidence_export.json', {'raw_dataset': 'present once in upload; omitted from return',
              'preserved': ['all methods and failures', 'neural checkpoints', 'dictionary parameters',
                            'frozen bank', 'matured moments', 'predictions', 'coefficient/state logs',
                            'retrospective audit', 'fixed statistical families', 'costs', 'vector figures'],
              'local_runtime_validation': 'not_run'})
    state = {'stage': 'environment', 'fits': {name: {'status': 'pending'} for name in ('T0', 'TF', 'D-ALL')},
             'status': 'running', 'runtime_location': 'AutoDL'}
    original_out, original_err = sys.stdout, sys.stderr
    stream = (run/'console.log').open('w', encoding='utf-8', buffering=1)
    sys.stdout, sys.stderr = Tee(original_out, stream), Tee(original_err, stream)
    started = time.monotonic()
    exit_code = 0
    try:
        save_json(run/'environment.json', environment(cfg))
        from src.runtime_checks import run_checks
        state['stage'] = 'runtime_checks'
        save_json(run/'run_state.json', state)
        run_checks(cfg, run/'evidence/runtime_checks.json')
        from src.synthetic import run_synthetic
        state['stage'] = 'synthetic'
        save_json(run/'run_state.json', state)
        synthetic = run_synthetic(cfg, manifest, run)
        if any(s['status'] != 'completed' for s in synthetic.values()):
            raise RuntimeError('Synthetic pipeline failed; financial training blocked. Adverse accuracy alone does not block.')
        if args.sanity_only:
            state['status'] = 'sanity_only_completed'
        else:
            state['stage'] = 'financial'
            save_json(run/'run_state.json', state)
            complete = financial(cfg, manifest, run, state)
            state['status'] = 'completed' if complete else 'completed_with_failed_or_unidentified_methods'
        state['stage'] = 'finished'
    except (Exception, KeyboardInterrupt) as exc:
        state['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        state['error'] = repr(exc)
        exit_code = 1
        save_json(run/'evidence/failures/run.json', {**state, 'traceback': traceback.format_exc()})
        print(traceback.format_exc(), flush=True)
    finally:
        state['elapsed_seconds'] = time.monotonic()-started
        state['finished_at'] = datetime.now(timezone.utc).isoformat()
        save_json(run/'run_state.json', state)
        save_json(run/'run_status.json', state)
        print({'event': 'finished', 'status': state['status'], 'seconds': state['elapsed_seconds']}, flush=True)
        sys.stdout.flush()
        sys.stdout, sys.stderr = original_out, original_err
        stream.close()
        returned = archive(run)
        print('Download evidence archive:', returned, flush=True)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
