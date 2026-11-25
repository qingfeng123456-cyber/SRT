"""Past-only daily features and fixed-at-origin historical CSI300 cohorts.

Runtime preprocessing executes only on AutoDL. No future missingness selection.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from .io_utils import save_json, append_csv


def rolling(x, window, minimum=None):
    mask = np.isfinite(x)
    x0 = np.where(mask, x, 0.0)
    def sums(a):
        cumulative = np.concatenate([np.zeros_like(a[:1]), np.cumsum(a, axis=0)], axis=0)
        ends = np.arange(1, len(a)+1)
        return cumulative[ends]-cumulative[np.maximum(ends-window, 0)]
    count = sums(mask.astype(float))
    mean = sums(x0)/np.maximum(count, 1)
    variance = np.maximum(0, sums(x0*x0)/np.maximum(count, 1)-mean*mean)
    valid = count >= (window if minimum is None else minimum)
    return np.where(valid, mean, np.nan), np.where(valid, np.sqrt(variance), np.nan), count


def logratio(a, b, scale=1.0):
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    result = np.full(np.broadcast_shapes(a.shape, b.shape), np.nan)
    np.subtract(np.log(np.where(ok, a, 1)), np.log(np.where(ok, b, 1)), out=result, where=ok)
    return result*scale


def lag(x, steps):
    out = np.full_like(x, np.nan)
    out[steps:] = x[:-steps]
    return out


class MarketData:
    def __init__(self, root, cfg, audit_dir):
        self.cfg = cfg
        self.history = cfg['data']['history_days']
        raw = Path(root)/cfg['raw_data']
        manifest = json.loads((raw.parent/'source_manifest.json').read_text(encoding='utf-8'))
        if manifest['cutoff'] != '2021-12-31':
            raise ValueError('Closed data cutoff required')
        for relative, item in manifest['files'].items():
            path = raw.parent/relative
            if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('Data checksum mismatch: '+relative)
        self.days = np.asarray((raw/'calendars/day.txt').read_text().splitlines())
        if max(self.days) > '2021-12-31' or list(self.days) != sorted(set(self.days)):
            raise ValueError('Invalid calendar or 2022 opened')
        rows = [line.split() for line in (raw/'instruments/csi300.txt').read_text().splitlines() if line.strip()]
        self.codes = np.asarray(sorted({r[0].upper() for r in rows}))
        lookup = {c:i for i,c in enumerate(self.codes)}
        shape = (len(self.days), len(self.codes))
        self.members = np.zeros(shape, bool)
        for code, begin, end in rows:
            if end > '2021-12-31':
                raise ValueError('Unclosed membership interval')
            self.members[np.searchsorted(self.days, begin):np.searchsorted(self.days, end, side='right'), lookup[code.upper()]] = True
        fields = {}
        missing = set(manifest['missing_source_files'])
        for field in cfg['data']['fields']:
            matrix = np.full(shape, np.nan)
            for j, code in enumerate(self.codes):
                relative = Path('qlib_cn/features')/str(code).lower()/(field+'.day.bin')
                path = raw.parent/relative
                if not path.exists():
                    if relative.as_posix() not in missing:
                        raise FileNotFoundError(path)
                    continue
                payload = np.fromfile(path, dtype='<f4')
                start = int(payload[0])
                stop = start+len(payload)-1
                if start < 0 or start != float(payload[0]) or stop > len(self.days):
                    raise ValueError('Invalid Qlib payload '+str(path))
                matrix[start:stop,j] = payload[1:]
            fields[field] = matrix
        self.close = fields['close']
        self.volume = fields['volume']
        self.close_ok = np.isfinite(self.close) & (self.close > 0)
        r = logratio(self.close, lag(self.close, 1), 100)
        self.daily = r
        _, _, history_count = rolling(r, self.history, 1)
        self.state_valid = self.close_ok & (history_count >= cfg['data']['minimum_history'])
        amounts = np.where(fields['amount'] > 0, fields['amount'], np.nan)
        liquidity, _, liquidity_count = rolling(amounts, cfg['data']['liquidity_window'], 1)
        self.cohorts = []
        for t in range(len(self.days)):
            ready = (self.members[t] & self.state_valid[t]
                & (liquidity_count[t] >= cfg['data']['minimum_liquidity_observations']))
            ids = np.flatnonzero(ready)
            if len(ids) > cfg['data']['maximum_stocks']:
                ids = ids[np.argsort(-liquidity[t,ids], kind='stable')[:cfg['data']['maximum_stocks']]]
            if t < self.history:
                ids = np.array([], dtype=np.int64)
            self.cohorts.append(np.sort(ids))
        market = np.full(len(self.days), np.nan)
        for t in range(1, len(self.days)):
            ok = self.members[t-1] & np.isfinite(r[t])
            if ok.any():
                market[t] = r[t,ok].mean()
        values, names = [], []
        def add(name, v):
            names.append(name)
            values.append(np.broadcast_to(v, shape))
        add('log_return_1_div5', r/5)
        for field in ('open','high','low','vwap'):
            add(field+'_to_close_logpct_div5', logratio(fields[field], self.close, 20))
        add('high_low_logpct_div5', logratio(fields['high'], fields['low'], 20))
        for field in ('amount','volume'):
            positive = np.where(fields[field] > 0, fields[field], np.nan)
            mean, _, _ = rolling(positive, 20, 16)
            add(field+'_to_past20_log_div2', logratio(positive, mean, .5))
            for w in (5,20):
                add(field+'_lag'+str(w)+'_log_div2', logratio(positive, lag(positive,w), .5))
        add('market_return_1_div5', market[:,None]/5)
        add('excess_return_1_div5', (r-market[:,None])/5)
        for w in (5,20,60):
            add('momentum_'+str(w)+'_div10', logratio(self.close, lag(self.close,w), 10))
            _, std, _ = rolling(r,w,max(3,int(.8*w)))
            add('volatility_'+str(w)+'_div5', std/5)
            mean, _, _ = rolling(np.where(self.close_ok,self.close,np.nan),w,max(3,int(.8*w)))
            add('close_to_mean_'+str(w)+'_logpct_div10', logratio(self.close,mean,10))
        for w in (5,20):
            mean, _, _ = rolling(market[:,None],w,max(3,int(.8*w)))
            add('market_momentum_'+str(w)+'_div10', mean*w/10)
        _, std, _ = rolling(market[:,None],20,16)
        add('market_volatility_20_div5', std/5)
        values = np.stack(values,axis=-1)
        observed = np.isfinite(values)
        self.feature_names = names+[name+'_observed' for name in names]
        self.n_values = len(names)
        self.n_features = len(self.feature_names)
        clip = cfg['data']['feature_clip']
        self.features = np.concatenate([np.clip(np.where(observed,values,0),-clip,clip), observed],axis=-1).astype(np.float32)
        self.bounds = {}
        previous_stop = -1
        for segment,(begin,end) in sorted(cfg['segments'].items(), key=lambda pair: pair[1][0]):
            start,stop = int(np.searchsorted(self.days,begin)),int(np.searchsorted(self.days,end,side='right'))
            if start < previous_stop or not start < stop or end > '2021-12-31':
                raise ValueError('Segments must be chronological, disjoint and closed')
            self.bounds[segment] = (start,stop)
            previous_stop = stop
        self.internal = {}
        fraction = cfg['training']['internal_fit_fraction']
        for segment in ('A', 'B'):
            begin, end = self.bounds[segment]
            cut = begin + int((end-begin)*fraction)
            allowed = self.indices(segment, mature=True)
            self.internal[segment] = {
                'fit': [t for t in allowed if t+5 < cut],
                'select': [t for t in allowed if t >= cut], 'cut': cut}
        self.label_cache = {}
        means, seconds = [], []
        for t in self.internal['A']['fit']:
            z = self.targets(t, 'A')
            if z['mask'].any():
                values = z['y'][z['mask']].astype(float)
                means.append(float(values.mean()))
                seconds.append(float((values**2).mean()))
        self.variance = float(np.mean(seconds)-np.mean(means)**2)
        if not np.isfinite(self.variance) or self.variance <= 1e-12:
            raise ValueError('A-fit target variance is invalid')
        self.final_sigma = float(np.sqrt(self.variance))
        self.daily_sigma = self.final_sigma/2
        self.write_audit(audit_dir, manifest)

    def indices(self, segment, mature=False):
        start,stop = self.bounds[segment]
        return [t for t in range(start,stop-5 if mature else stop)
            if len(self.cohorts[t]) >= self.cfg['data']['minimum_stocks']]

    def window(self, t, ids=None):
        ids = self.cohorts[t] if ids is None else np.asarray(ids)
        if t-self.history+1 < 0:
            raise ValueError('History underflow')
        return np.transpose(self.features[t-self.history+1:t+1,ids],(1,0,2)).copy()

    def targets(self,t,segment):
        ids = self.cohorts[t]
        n = len(ids)
        _,stop = self.bounds[segment]
        if segment in ('CHECK', 'WARM', '2020H2', '2021'):
            stop = len(self.days)
        mature = t+5 < stop
        y,daily = np.zeros(n),np.zeros((n,5))
        mask,dm,sm = np.zeros(n,bool),np.zeros((n,5),bool),np.zeros((n,5),bool)
        if mature:
            daily0 = self.daily[t+1:t+6,ids].T
            dm = np.isfinite(daily0)
            daily = np.where(dm,daily0,0)
            sm = self.state_valid[t+1:t+6,ids].T.copy()
            raw_y = logratio(self.close[t+5,ids],self.close[t+1,ids],100)
            mask = np.isfinite(raw_y)
            y = np.where(mask,raw_y,0)
        cumulative = np.zeros((n,5),dtype=np.float32)
        cumulative_mask = np.zeros((n,5),dtype=bool)
        # Same five-origin purge for every supervised objective. No boundary
        # crossing auxiliary target is smuggled into the training loss.
        if mature:
            for h in range(1,6):
                value = logratio(self.close[t+h,ids],self.close[t,ids],100)
                cumulative_mask[:,h-1] = np.isfinite(value)
                cumulative[:,h-1] = np.where(np.isfinite(value),value,0)
        return {'y':y.astype(np.float32),'mask':mask,'daily':daily.astype(np.float32),
            'daily_mask':dm,'state_mask':sm,'mature':mature,
            'cumulative':cumulative,'cumulative_mask':cumulative_mask}

    def fixed_tools(self,t,ids):
        columns=[self.feature_names.index(name) for name in ('log_return_1_div5', 'momentum_5_div10', 'momentum_20_div10', 'volatility_20_div5')]
        return self.features[t,np.asarray(ids)][:,columns].copy()

    def write_audit(self,folder,manifest):
        rows=[]
        folder=Path(folder)
        folder.mkdir(parents=True,exist_ok=True)
        with (folder/'cohorts.jsonl').open('w',encoding='utf-8') as stream:
            for segment,(start,stop) in self.bounds.items():
                for t in range(start,stop):
                    ids=self.cohorts[t]
                    target=self.targets(t,segment)
                    rows.append({'segment':segment,'date':str(self.days[t]),'cohort':len(ids),
                        'observed_targets':int(target['mask'].sum()),'mature':target['mature']})
                    stream.write(json.dumps({'segment':segment,'date':str(self.days[t]),
                        'codes':self.codes[ids].tolist()})+'\n')
        availability = folder/'availability.csv'
        if availability.exists():
            availability.unlink()
        append_csv(availability,rows)
        save_json(folder/'data_audit.json',{'features':self.feature_names,'feature_dimensions':self.n_features,
            'normalization':'fixed past-only feature divisors/clipping; losses scaled by A-fit target variance; masks explicit',
            'A_fit_daily_sigma_proxy':self.daily_sigma,'A_fit_final_sigma':self.final_sigma, 'A_fit_variance':self.variance,
            'cohort_rule':'all past-eligible historical CSI300, capped at300 by past amount only',
            'forecast_clock':'after close t; entry close t+1; exit close t+5',
            'label_rule':'per-stock masks, no complete-date deletion; five-origin segment purge',
            'source_limits':[manifest['price_convention'],manifest['membership_limit']],
            'confirmation_2022':'physically excluded','local_runtime_validation':'not_run'})

    def label(self, t, segment):
        from .metrics import standardized_rank
        key = (segment, int(t))
        if key not in self.label_cache:
            z = self.targets(t, segment)
            valid = z['mask']
            v = np.zeros(len(valid), dtype=np.float32)
            ranked = standardized_rank(z['y'][valid])
            rank_mask = np.zeros_like(valid)
            if ranked is not None:
                v[valid] = ranked
                rank_mask[valid] = True
            z['v'], z['rank_mask'] = v, rank_mask
            self.label_cache[key] = z
        return self.label_cache[key]

    def items(self, segments, part=None):
        out = []
        for segment in segments:
            dates = self.internal[segment][part] if part else self.indices(segment, mature=True)
            out.extend((segment, int(t)) for t in dates)
        return out

    def pretrain_items(self):
        begin, end = self.bounds['A'][0], self.bounds['B'][1]
        cut = begin + int((end-begin)*self.cfg['training']['internal_fit_fraction'])
        items = self.items(['A', 'B'])
        return ([x for x in items if x[1]+5 < cut], [x for x in items if x[1] >= cut])

    def future(self, t, k):
        if not any(self.bounds[s][0] <= t and t+5 < self.bounds[s][1] for s in ('A', 'B', 'C')):
            raise ValueError('Future teacher inputs are restricted to mature A/B/C origins')
        return self._future_slots(t, k)

    def audit_future(self, t, k, released_at):
        """Retrospective audit only; never called by controllers or forecasting."""
        if t+5 > released_at or t+5 >= len(self.days):
            raise ValueError('Audit target has not completely matured')
        return self._future_slots(t, k)

    def _future_slots(self, t, k):
        ids = self.cohorts[t]
        if k not in (1, 3):
            raise ValueError('Registered future k only')
        # Original X is retained by the model. These are separate training-only slots.
        slots = np.zeros((len(ids), 3, self.n_features+1), dtype=np.float32)
        slots[:, :k, :-1] = self.features[t+1:t+k+1, ids].transpose(1, 0, 2)
        slots[:, :k, -1] = 1
        prefix = np.zeros(len(ids), dtype=np.float32)
        available = np.ones(len(ids), dtype=bool)
        if k > 1:
            increments = self.daily[t+2:t+k+1, ids].T
            available = np.isfinite(increments).all(1)
            prefix = np.where(available, np.nan_to_num(increments).sum(1), 0).astype(np.float32)
        return slots, prefix, available
