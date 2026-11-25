"""Numerical and clock checks invoked only by the AutoDL entry point."""
import itertools
import numpy as np
from .io_utils import save_json


def run_checks(cfg, path):
    from .controllers import solve_coeff, MomentControl
    from .state import GainFilter, forward_backward
    from .replay import VIEW_KEYS
    checks = []
    B, a = np.eye(2), np.asarray([1., -.5])
    alpha, _ = solve_coeff(B, a, cfg)
    expected = np.sign(a)*np.maximum(np.abs(a)-cfg['controller']['tau']/2, 0)/(1+cfg['controller']['rho'])
    assert np.allclose(alpha, expected, atol=1e-5)
    checks.append('bounded_convex_solution_matches_diagonal_closed_form')
    alpha, _ = solve_coeff(B, np.full(2, cfg['controller']['tau']/4), cfg)
    assert np.allclose(alpha, 0, atol=1e-12)
    checks.append('l1_true_zero_action')
    x = np.asarray(list(itertools.product([-1., 1.], repeat=2)))
    y = x.sum(1)
    c = -.2*x[:, 0]+x[:, 1]
    fk, fe = .5*x[:, 0], .4*x[:, 0]
    actual = np.asarray([np.mean(y*y), np.mean((y-c)**2), np.mean((y-fk)**2), np.mean((y-fe)**2)])
    assert np.allclose(actual, [2., 1.44, 1.25, 1.36], atol=1e-12)
    checks.append('teacher_student_projection_counterexample')
    from .linear import fit_ridge, predict_ridge
    target = np.column_stack([y, y/np.sqrt(2)])
    correction = np.column_stack([c, c/np.sqrt(2)])
    toy = {'h': x[:, :1], 'truth': target, 'correction': correction,
           'mask': np.ones((4, 2), dtype=bool)}
    head = fit_ridge([toy], 'h', lambda r: np.stack([.5*r['truth'], .5*(r['truth']+r['correction'])], axis=2),
                     lambda r: r['mask'], .0001, (np.zeros(1), np.ones(1)))
    projected = predict_ridge(toy['h'], head)
    assert np.allclose(projected[:, 0, 0], fk, atol=1e-3)
    assert np.allclose(projected[:, 0, 1], fe, atol=1e-3)
    assert np.allclose(projected[:, 1].mean(0), 0, atol=1e-10)
    checks.append('actual_shared_ridge_operator_and_rank_centering')
    from .metrics import standardized_rank
    assert np.allclose(standardized_rank(y), standardized_rank(y+3.2))
    checks.append('common_shift_leaves_cross_sectional_rank_unchanged')
    from .metrics import paired_test
    dated = np.arange(30, dtype=float)
    dated[8] = np.nan
    result = paired_test(dated, np.tile(np.arange(30), (10, 1)), cfg['inference'])
    assert result['dates'] == 29 and result['calendar_origins'] == 30
    assert np.isclose(result['mean_improvement'], np.nanmean(dated))
    checks.append('paired_inference_keeps_internal_calendar_gap')
    P = np.full((3, 3), .05)
    np.fill_diagonal(P, .9)
    model = {'initial': np.asarray([.7, .2, .1]), 'P': P,
             'b': np.asarray([[-1., 0.], [0., 0.], [1., 0.]]),
             'omega': np.eye(2), 'first_t': 1}
    state = GainFilter(model)
    state.update({'t': 1, 'a': np.asarray([.2, 0.]), 'B': np.eye(2)})
    saved_pi, saved_origin = state.pi.copy(), state.last
    _, _, now = state.predict(6)
    assert np.allclose(now, saved_pi@np.linalg.matrix_power(P, 5))
    assert np.allclose(state.pi, saved_pi) and state.last == saved_origin
    state.update({'t': 2, 'a': np.asarray([.1, 0.]), 'B': np.eye(2)})
    assert state.last == 2
    checks.append('five_date_state_propagation_does_not_mutate_origin_filter')
    try:
        state.update({'t': 2, 'a': np.zeros(2), 'B': np.eye(2)})
        raise AssertionError('Duplicate feedback accepted')
    except ValueError:
        pass
    checks.append('duplicate_feedback_rejected')
    e = np.log(np.asarray([[.2, .3, .5], [.5, .4, .1], [.3, .3, .4]]))
    ll, gamma, _, filtered = forward_backward(e, P, model['initial'])
    enumeration = sum(model['initial'][z[0]]*np.exp(e[0, z[0]])*
                      P[z[0], z[1]]*np.exp(e[1, z[1]])*P[z[1], z[2]]*np.exp(e[2, z[2]])
                      for z in itertools.product(range(3), repeat=3))
    assert np.allclose(np.exp(ll), enumeration) and np.allclose(gamma.sum(1), 1)
    assert np.allclose(filtered.sum(1), 1)
    checks.append('forward_backward_matches_enumeration')
    assert not {'truth', 'mask', 'teacher', 'hidden_state', 'known_mean'}.intersection(VIEW_KEYS)
    checks.append('deploy_view_excludes_future_label_and_audit_fields')
    control = MomentControl(2, cfg)
    control.update({'a': np.ones(2), 'B': np.eye(2), 't': 1}, 6)
    before = control.statistics(7)
    assert np.allclose(before[0], np.eye(2)) and np.allclose(before[1], np.ones(2))
    checks.append('equal_date_forgetting_moments')
    import torch
    from .models import ForecastNet
    network = ForecastNet(6, cfg, 1.).to(cfg['device'])
    with torch.no_grad():
        output = network(torch.zeros((4, 8, 6), device=cfg['device']))
    assert tuple(output.shape) == (4, 2) and bool(torch.isfinite(output).all())
    checks.append('shared_network_forward_on_AutoDL')
    save_json(path, {'status': 'passed_on_AutoDL', 'checks': checks,
                     'runtime_location': 'AutoDL', 'local_execution': False})


def check_clock(data, cfg, path):
    assert data.bounds['C'][1] <= data.bounds['SELECT'][0]
    assert data.bounds['SELECT'][1] <= data.bounds['REL'][0]
    assert data.bounds['REL'][1] <= data.bounds['CHECK'][0]
    assert max(data.days) <= '2021-12-31'
    try:
        data.future(data.bounds['REL'][0], 1)
        raise AssertionError('Deployment future teacher interface accepted')
    except ValueError:
        pass
    sample = data.items(['A', 'B', 'C', 'SELECT', 'REL'])
    for segment, t in sample[::max(1, len(sample)//20)]:
        z = data.label(t, segment)
        ids = data.cohorts[t]
        valid = z['mask']
        raw = 100*np.log(data.close[t+5, ids[valid]]/data.close[t+1, ids[valid]])
        assert np.allclose(z['y'][valid], raw, atol=1e-5)
        assert t+5 < data.bounds[segment][1]
    save_json(path, {'status': 'passed_on_AutoDL', 'target': cfg['target'],
                     'future_interface_restricted': True, '2022_opened': False})
