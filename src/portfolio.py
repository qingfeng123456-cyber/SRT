"""Four capital sleeves; reused close-price development diagnostic."""
import numpy as np
from .io_utils import append_csv, save_json


def portfolio(data,predictions,segment,cfg,folder):
    """Four capital sleeves; close(t+1) entry, close(t+5) scheduled exit.

    Prices missing at sale defer the sale; absent volume also blocks execution.
    Missing marks are carried, tracked and disclosed, never converted to cash.
    """
    buy_fee=cfg['evaluation']['buy_cost_bps']/10000
    sell_fee=cfg['evaluation']['sell_cost_bps']/10000
    sleeves=[{'cash':.25,'lots':[]} for _ in range(4)]
    start,stop=data.bounds[segment]
    previous_nav=1.0
    rows=[]
    for d in range(start,stop):
        traded,fees,stale_exposure=0.0,0.0,0.0
        for sleeve in sleeves:
            retained=[]
            for lot in sleeve['lots']:
                code=lot['id']
                mark=data.close[d,code]
                observed=np.isfinite(mark) and mark>0
                tradable=observed and np.isfinite(data.volume[d,code]) and data.volume[d,code]>0
                if observed:
                    lot['mark']=float(mark)
                else:
                    stale_exposure+=lot['quantity']*lot['mark']
                if d>=lot['exit'] and tradable:
                    value=lot['quantity']*mark
                    sleeve['cash']+=value*(1-sell_fee)
                    traded+=value
                    fees+=value*sell_fee
                else:
                    retained.append(lot)
            sleeve['lots']=retained
        origin=d-1
        if origin in predictions and origin+5<stop:
            ids,scores=predictions[origin]
            count=max(1,int(np.ceil(len(ids)*cfg['evaluation']['top_fraction'])))
            selected=np.argsort(-scores,kind='stable')[:count]
            sleeve=sleeves[origin%4]
            allocation=sleeve['cash']/count
            for j in selected:
                code=int(ids[j])
                price=data.close[d,code]
                if not (np.isfinite(price) and price>0 and np.isfinite(data.volume[d,code]) and data.volume[d,code]>0):
                    continue
                notional=allocation/(1+buy_fee)
                sleeve['cash']-=allocation
                sleeve['lots'].append({'id':code,'quantity':notional/price,'mark':float(price),'exit':origin+5})
                traded+=notional
                fees+=notional*buy_fee
        nav=sum(s['cash']+sum(l['quantity']*l['mark'] for l in s['lots']) for s in sleeves)
        rows.append({'date':str(data.days[d]),'nav':nav,'net_return':nav/previous_nav-1,
            'turnover':traded/previous_nav,'fees':fees,'stale_mark_exposure':stale_exposure/nav,
            'overdue_lots':sum(l['exit']<=d for s in sleeves for l in s['lots'])})
        previous_nav=nav
    append_csv(folder/(segment+'_portfolio_daily.csv'),rows)
    nav=np.asarray([1.0]+[r['nav'] for r in rows])
    returns=np.asarray([r['net_return'] for r in rows])
    unresolved=sum(l['quantity']*l['mark'] for s in sleeves for l in s['lots'])
    result={'net_total_return':float(nav[-1]-1),
        'annualized_return':float(nav[-1]**(252/max(len(rows),1))-1),
        'sharpe_zero_cash_rate':float(np.sqrt(252)*returns.mean()/returns.std(ddof=1)) if len(returns)>1 and returns.std()>0 else None,
        'maximum_drawdown':float(np.min(nav/np.maximum.accumulate(nav)-1)),
        'mean_daily_turnover':float(np.mean([r['turnover'] for r in rows])),
        'stale_mark_dates':sum(r['stale_mark_exposure']>0 for r in rows),
        'ending_unresolved_position_value':unresolved,'ending_nav':float(nav[-1]),
        'scope':'close-price, positive-volume execution proxy; no limit-lock/auction-fill simulation; provider price convention',
        'missing_rule':cfg['evaluation']['missing_price_policy']}
    save_json(folder/(segment+'_portfolio_summary.json'),result)
    return rows,result
