"""Three fixed financial neural fits; all runtime work is AutoDL-only."""
import gc
import hashlib
import time
from pathlib import Path
import numpy as np
import torch
from .models import ForecastNet, mse
from .io_utils import append_csv, save_json


def tensor(x, cfg, boolean=False):
    return torch.as_tensor(x, dtype=torch.bool if boolean else torch.float32,
                           device=cfg['device'])


def atomic_torch(path, packet):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.pt')
    torch.save(packet, tmp)
    tmp.replace(path)


def loss_for(model, name, data, item, cfg):
    segment, t = item
    z = data.label(t, segment)
    x = tensor(data.window(t), cfg)
    y, v = tensor(z['y'], cfg), tensor(z['v'], cfg)
    mask, rm = tensor(z['mask'], cfg, True), tensor(z['rank_mask'], cfg, True)
    parts = []
    if name == 'TF':
        for k in cfg['training']['future_k']:
            future, prefix, available = data.future(t, k)
            output = model(x, tensor(future, cfg), k)
            ok = tensor(available, cfg, True)
            point = mse(output[:, 0], y-tensor(prefix, cfg), mask & ok)
            rank = mse(output[:, 1]-output[:, 1].mean(), v, rm & ok)
            if point is not None:
                parts.append(point/data.variance)
            if rank is not None:
                parts.append(rank)
    else:
        output = model(x)
        point = mse(output[:, 0], y, mask)
        # Center on the origin universe, before any future missingness mask.
        rank = mse(output[:, 1]-output[:, 1].mean(), v, rm)
        if point is not None:
            parts.append(point/data.variance)
        if rank is not None:
            parts.append(rank)
    return torch.stack(parts).mean() if parts else None


@torch.no_grad()
def validate(model, name, data, items, cfg):
    model.eval()
    values = []
    for item in items:
        loss = loss_for(model, name, data, item, cfg)
        if loss is not None:
            values.append(float(loss))
    if not values or not np.isfinite(values).all():
        raise FloatingPointError('Invalid selection objective '+name)
    return float(np.mean(values))


def fit(name, data, cfg, run):
    start = time.monotonic()
    folder = Path(run)/'fits'/name
    folder.mkdir(parents=True, exist_ok=True)
    train = data.items(['A'], 'fit') if name in ('T0', 'TF') else data.items(['A', 'B', 'C'])
    select = data.items(['A'], 'select') if name in ('T0', 'TF') else data.items(['SELECT'])
    if len(train) < cfg['data']['minimum_train_dates'] or len(select) < cfg['data']['minimum_validation_dates']:
        raise ValueError('Insufficient training or selection dates '+name)
    assert max(t+5 for _, t in train) < min(t for _, t in select)
    derived = int(hashlib.sha256((name+str(cfg['random_state'])).encode()).hexdigest()[:8], 16)
    torch.manual_seed(derived)
    model = ForecastNet(data.n_features, cfg, data.final_sigma).to(cfg['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['learning_rate'],
                                 weight_decay=cfg['training']['weight_decay'])
    torch.cuda.reset_peak_memory_stats(cfg['device'])
    best, patience, steps = np.inf, 0, 0
    checkpoint = folder/'best.pt'
    for epoch in range(cfg['training']['epochs']):
        order = np.random.default_rng(derived+epoch).permutation(len(train))
        model.train()
        losses = []
        batch = cfg['training']['dates_per_update']
        for begin in range(0, len(order), batch):
            optimizer.zero_grad(set_to_none=True)
            parts = [loss_for(model, name, data, train[j], cfg) for j in order[begin:begin+batch]]
            parts = [value for value in parts if value is not None]
            if not parts:
                continue
            loss = torch.stack(parts).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite loss '+name)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['clip_norm'])
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError('Nonfinite gradient '+name)
            optimizer.step()
            steps += 1
            losses.append(float(loss.detach()))
            if steps % cfg['training']['progress_every_updates'] == 0:
                print({'event': 'train', 'fit': name, 'epoch': epoch+1, 'updates': steps,
                       'loss': losses[-1]}, flush=True)
        score = validate(model, name, data, select, cfg)
        improved = score < best
        if improved:
            best, patience = score, 0
            atomic_torch(checkpoint, {'model': model.state_dict(), 'epoch': epoch+1,
                                     'selection_objective': best})
        else:
            patience += 1
        append_csv(folder/'epochs.csv', [{'epoch': epoch+1, 'optimizer_updates': steps,
                                        'train_loss': float(np.mean(losses)) if losses else None,
                                        'selection': score, 'selected': improved}])
        print({'event': 'epoch', 'fit': name, 'epoch': epoch+1, 'selection': score}, flush=True)
        if patience >= cfg['training']['patience']:
            break
    torch.cuda.synchronize(cfg['device'])
    packet = torch.load(checkpoint, map_location=cfg['device'], weights_only=False)
    model.load_state_dict(packet['model'])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    result = {'status': 'completed', 'fit': name, 'checkpoint': str(checkpoint),
              'train_dates': len(train), 'selection_dates': len(select),
              'latest_training_target': str(data.days[max(t+5 for _, t in train)]),
              'last_selection_target': str(data.days[max(t+5 for _, t in select)]),
              'optimizer_updates': steps, 'parameters': sum(p.numel() for p in model.parameters()),
              'elapsed_seconds': time.monotonic()-start,
              'peak_cuda_bytes': torch.cuda.max_memory_allocated(cfg['device']), 'runtime_location': 'AutoDL'}
    save_json(folder/'result.json', result)
    del optimizer
    gc.collect()
    return model, result
