"""Observed-design three-state gain model. Parameters freeze before deployment."""
import numpy as np
from scipy.special import logsumexp


def regular_cov(cov, floor):
    values, vectors = np.linalg.eigh((cov+cov.T)/2)
    return (vectors*np.maximum(values, floor))@vectors.T


def emission(a, B, b, omega):
    difference = a[None, :]-np.einsum('ij,kj->ki', B, b)
    sign, logdet = np.linalg.slogdet(omega)
    if sign <= 0:
        raise FloatingPointError('Invalid emission covariance')
    return -.5*(len(a)*np.log(2*np.pi)+logdet+
                np.einsum('ki,ij,kj->k', difference, np.linalg.inv(omega), difference))


def forward_backward(log_e, P, initial):
    n, k = log_e.shape
    logp = np.log(np.maximum(P, 1e-300))
    f = np.empty((n, k))
    f[0] = np.log(np.maximum(initial, 1e-300))+log_e[0]
    for j in range(1, n):
        f[j] = log_e[j]+logsumexp(f[j-1, :, None]+logp, axis=0)
    likelihood = logsumexp(f[-1])
    back = np.zeros((n, k))
    for j in range(n-2, -1, -1):
        back[j] = logsumexp(logp+log_e[j+1][None, :]+back[j+1][None, :], axis=1)
    gamma = np.exp(f+back-likelihood)
    gamma /= gamma.sum(1, keepdims=True)
    xi = np.zeros((k, k))
    for j in range(n-1):
        value = f[j, :, None]+logp+log_e[j+1][None, :]+back[j+1][None, :]-likelihood
        v = np.exp(value-logsumexp(value))
        xi += v
    filtered = np.exp(f-logsumexp(f, axis=1, keepdims=True))
    return float(likelihood), gamma, xi, filtered


def fit_state(observations, cfg):
    """Missing-date emissions are neutral; P always advances by market date."""
    settings = cfg['state_model']
    k = cfg['controller']['states']
    if len(observations) < settings['minimum_dates']:
        raise ValueError('Insufficient dates for registered three-state fit')
    lookup = {o['t']: o for o in observations}
    times = np.arange(min(lookup), max(lookup)+1)
    p = len(observations[0]['a'])
    a = np.stack([o['a'] for o in observations])
    Bs = np.stack([o['B'] for o in observations])
    score = a[:, 0]/np.maximum(Bs[:, 0, 0], .01)
    order = np.argsort(score, kind='stable')
    groups = np.array_split(order, k)
    b = np.zeros((k, p))
    for state, group in enumerate(groups):
        mat = sum(Bs[j].T@Bs[j] for j in group)/len(group)
        rhs = sum(Bs[j].T@a[j] for j in group)/len(group)
        b[state] = np.linalg.solve(mat+settings['parameter_ridge']*np.eye(p), rhs)
    omega = regular_cov(np.cov(a, rowvar=False)+settings['covariance_floor']*np.eye(p),
                        settings['covariance_floor'])
    stay = settings['initial_stay_probability']
    P = np.full((k, k), (1-stay)/(k-1))
    np.fill_diagonal(P, stay)
    initial = np.full(k, 1/k)
    history, converged = [], False
    for iteration in range(settings['max_iterations']):
        log_e = np.zeros((len(times), k))
        for j, t in enumerate(times):
            if int(t) in lookup:
                o = lookup[int(t)]
                log_e[j] = emission(o['a'], o['B'], b, omega)
        ll, gamma, xi, filtered = forward_backward(log_e, P, initial)
        if not np.isfinite(ll):
            raise FloatingPointError('Nonfinite state likelihood')
        history.append(ll)
        if iteration % 10 == 0:
            print({'event': 'state_fit_iteration', 'iteration': iteration+1,
                   'dates': len(observations), 'log_likelihood': ll}, flush=True)
        if len(history) > 2 and abs(history[-1]-history[-2]) <= settings['tolerance']*(1+abs(history[-2])):
            converged = True
            break
        initial = np.maximum(gamma[0], 1e-8)
        initial /= initial.sum()
        P = xi+settings['transition_pseudocount']
        P /= P.sum(1, keepdims=True)
        inverse = np.linalg.inv(omega)
        for state in range(k):
            mat, rhs, total = np.zeros((p, p)), np.zeros(p), 0.
            for j, t in enumerate(times):
                if int(t) not in lookup:
                    continue
                o, w = lookup[int(t)], gamma[j, state]
                mat += w*(o['B'].T@inverse@o['B'])
                rhs += w*(o['B'].T@inverse@o['a'])
                total += w
            b[state] = np.linalg.solve(mat/max(total, 1)+settings['parameter_ridge']*np.eye(p),
                                        rhs/max(total, 1))
        covariance, total = np.zeros((p, p)), 0.
        for j, t in enumerate(times):
            if int(t) not in lookup:
                continue
            o = lookup[int(t)]
            for state in range(k):
                r = o['a']-o['B']@b[state]
                covariance += gamma[j, state]*np.outer(r, r)
                total += gamma[j, state]
        omega = regular_cov(covariance/max(total, 1), settings['covariance_floor'])
    # Recompute final posteriors for the returned parameters, including iteration-limit cases.
    log_e = np.zeros((len(times), k))
    for j, t in enumerate(times):
        if int(t) in lookup:
            o = lookup[int(t)]
            log_e[j] = emission(o['a'], o['B'], b, omega)
    ll, gamma, _, filtered = forward_backward(log_e, P, initial)
    observed_indices = [j for j, t in enumerate(times) if int(t) in lookup]
    occupancy = gamma[observed_indices].sum(0)
    average_B = Bs.mean(0)
    distances = [float(np.sqrt(max((b[i]-b[j])@average_B@(b[i]-b[j]), 0)))
                 for i in range(k) for j in range(i)]
    design_condition = float(np.linalg.cond(average_B+settings['covariance_floor']*np.eye(p)))
    covariance_condition = float(np.linalg.cond(omega))
    reasons = []
    if not converged:
        reasons.append('iteration_limit')
    if occupancy.min() < settings['minimum_occupancy']:
        reasons.append('low_state_occupancy')
    if min(distances) < settings['minimum_state_separation']:
        reasons.append('poor_state_separation')
    if design_condition > settings['maximum_design_condition']:
        reasons.append('ill_conditioned_design')
    if covariance_condition > settings['maximum_covariance_condition']:
        reasons.append('ill_conditioned_emission')
    result = {'b': b, 'P': P, 'omega': omega, 'initial': initial,
              'first_t': int(times[0]), 'last_t': int(times[-1]), 'likelihood': ll,
              'likelihood_history': history, 'converged': converged,
              'occupancy': occupancy, 'state_distances': distances,
              'design_condition': design_condition, 'covariance_condition': covariance_condition,
              'effective_response_rank': int(np.linalg.matrix_rank(average_B, tol=1e-6)),
              'identification': 'identified' if not reasons else 'unidentified',
              'identification_reasons': reasons,
              'filtered_fit': filtered, 'fit_times': times,
              'assumptions': ['constant working covariance', 'stable coefficients within state',
                              'state disagreement is not full parameter uncertainty']}
    if not all(np.isfinite(result[key]).all() for key in ('b', 'P', 'omega')):
        raise FloatingPointError('Invalid state parameters')
    return result


class GainFilter:
    def __init__(self, model):
        self.model = model
        self.pi = model['initial'].copy()
        self.last = model['first_t']
        self.has_observation = False

    def update(self, observation):
        t = observation['t']
        if t < self.last or (t == self.last and self.has_observation):
            raise ValueError('Duplicate or backwards mature-origin update')
        prior = self.pi@np.linalg.matrix_power(self.model['P'], t-self.last)
        logp = np.log(np.maximum(prior, 1e-300))+emission(
            observation['a'], observation['B'], self.model['b'], self.model['omega'])
        self.pi = np.exp(logp-logsumexp(logp))
        self.last = t
        self.has_observation = True

    def predict(self, t, propagate=True):
        if t < self.last:
            raise ValueError('State forecast precedes filtered origin')
        pi = self.pi@np.linalg.matrix_power(self.model['P'], t-self.last) if propagate else self.pi.copy()
        mean = pi@self.model['b']
        diff = self.model['b']-mean
        variance = np.einsum('k,ki,kj->ij', pi, diff, diff)
        return mean, variance, pi
