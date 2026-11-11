"""Paired summaries for the common publication experiment."""
from collections import Counter
import json
from pathlib import Path
from statistics import mean


HERE = Path(__file__).resolve().parent


def main():
    data = json.loads((HERE / 'issued_indexed_results.json').read_text())
    rows = data['rows']
    methods = sorted({row['method'] for row in rows})
    lookup = {(r['date'], r['query'], r['method'], r['clock']): r for r in rows}
    contexts = sorted({(r['date'], r['query']) for r in rows})
    if len(contexts) != 4 or len(methods) != 5 or len(lookup) != len(rows) or len(rows) != 60:
        raise ValueError('The expected paired design is four cases, five planners and three clocks')
    summary = {}
    for method in methods:
        clocks = {}
        for clock in ('logical', 'background', 'blocking'):
            group = [lookup[day, query, method, clock] for day, query in contexts]
            clocks[clock] = dict(loss_lower=mean(r['loss_lower'] for r in group),
                loss_upper=mean(r['loss_upper'] for r in group),
                statuses=dict(Counter(r['status'] for r in group)),
                published=sum(r['published'] for r in group), rejected=sum(r['rejected'] for r in group),
                incumbent_moves=sum(r['incumbent_moves'] for r in group))
        pairs = []
        for day, query in contexts:
            a, b = (lookup[day, query, method, clock] for clock in ('background', 'blocking'))
            pairs.append(dict(date=day, query=query, difference_lower=a['loss_lower']-b['loss_upper'],
                difference_upper=a['loss_upper']-b['loss_lower'], publish_seconds=a['publish_clock']-29100,
                incumbent_boarded=[r['ridden_connection'] for r in a['rides'] if r['worker_running']]))
        summary[method] = dict(clocks=clocks, pairs=pairs,
            mean_plan_wall_seconds=mean(lookup[day, query, method, 'logical']['wall_plan_seconds']
                                       for day, query in contexts),
            mean_plan_cpu_seconds=mean(lookup[day, query, method, 'logical']['cpu_plan_seconds']
                                      for day, query in contexts))
    (HERE / 'issued_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    for method, value in summary.items():
        print(method, round(value['mean_plan_wall_seconds'], 3),
            [(clock, row['loss_lower'], row['loss_upper']) for clock, row in value['clocks'].items()])


if __name__ == '__main__':
    main()
