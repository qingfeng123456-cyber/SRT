"""Same-information online controls and bounded signed convex decisions."""
from collections import deque
import numpy as np


def l1_ball(value, radius):
    if np.abs(value).sum() <= radius:
        return value
    u = np.sort(np.abs(value))[::-1]
    sums = np.cumsum(u)
    ids = np.flatnonzero(u-(sums-radius)/np.arange(1, len(u)+1) > 0)
    j = int(ids[-1])
    threshold = (sums[j]-radius)/(j+1)
    return np.sign(value)*np.maximum(np.abs(value)-threshold, 0)


def solve_coeff(B, a, cfg, variance=None, nonnegative=False, radius=None):
    c = cfg['controller']
    Q = (B+B.T)/2+c['rho']*np.eye(len(a))
    if variance is not None:
        Q += c['kappa']*(B@variance@B)
    L = 2*float(np.linalg.eigvalsh(Q).max())
    if not np.isfinite(L) or L <= 0:
        raise FloatingPointError('Invalid decision curvature')
    cap = c['radius'] if radius is None else radius

    def prox(z):
        if nonnegative:
            z = np.maximum(z-c['tau']/L, 0)
        else:
            z = np.sign(z)*np.maximum(np.abs(z)-c['tau']/L, 0)
        return l1_ball(z, cap)

    x, z, momentum = np.zeros(len(a)), np.zeros(len(a)), 1.
    for iteration in range(c['solver_steps']):
        new = prox(z-(2*(Q@z-a))/L)
        next_momentum = (1+np.sqrt(1+4*momentum*momentum))/2
        z = new+(momentum-1)/next_momentum*(new-x)
        change = np.linalg.norm(new-x)
        x, momentum = new, next_momentum
        if change <= c['solver_tolerance']*(1+np.linalg.norm(x)):
            residual = np.linalg.norm(x-prox(x-2*(Q@x-a)/L))*L
            if residual <= c['kkt_tolerance']*(1+np.linalg.norm(a)):
                break
    fixed_point = prox(x-2*(Q@x-a)/L)
    gap = float(np.linalg.norm(x-fixed_point)*L)
    if gap > c['kkt_tolerance']*(1+np.linalg.norm(a)) or not np.isfinite(x).all():
        raise FloatingPointError('Coefficient solver failed fixed-point check: '+str(gap))
    return x, {'iterations': iteration+1, 'fixed_point_gap': gap,
               'l1_active': bool(np.abs(x).sum() >= cap-1e-6),
               'zero_action': bool(np.abs(x).sum() <= 1e-10)}


class MomentControl:
    def __init__(self, dimension, cfg, rolling=False):
        self.cfg, self.rolling = cfg, rolling
        self.a, self.B = np.zeros(dimension), np.zeros((dimension, dimension))
        self.weight, self.last = 0., None
        self.records = deque()

    def advance(self, t):
        if self.last is not None and t < self.last:
            raise ValueError('Backwards feedback clock')
        if not self.rolling:
            decay = 1. if self.last is None else self.cfg['controller']['forgetting']**(t-self.last)
            self.a *= decay
            self.B *= decay
            self.weight *= decay
        self.last = t

    def update(self, observation, released_at):
        self.advance(released_at)
        if self.rolling:
            self.records.append(observation)
        else:
            self.a += observation['a']
            self.B += observation['B']
            self.weight += 1

    def statistics(self, t):
        self.advance(t)
        if self.rolling:
            while self.records and self.records[0]['t'] < t-self.cfg['controller']['delay']-self.cfg['controller']['rolling_dates']+1:
                self.records.popleft()
            if not self.records:
                return self.B.copy(), self.a.copy()
            return (np.mean([o['B'] for o in self.records], axis=0),
                    np.mean([o['a'] for o in self.records], axis=0))
        return self.B/max(self.weight, 1e-12), self.a/max(self.weight, 1e-12)


class Hedge:
    def __init__(self, cfg):
        self.cfg = cfg
        self.weights = np.full(6, 1/6)

    def update(self, losses):
        c = self.cfg['controller']
        logits = np.log(np.maximum(self.weights, 1e-300))-c['hedge_rate']*(losses-losses.min())
        logits -= logits.max()
        value = np.exp(logits)
        value /= value.sum()
        self.weights = (1-c['fixed_share'])*value+c['fixed_share']/len(value)
