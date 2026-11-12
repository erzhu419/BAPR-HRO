"""Summarize the saved benchmark and plot its charged CPU/quality points."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
LABELS = dict(rolling_nominal='Nominal', single_rolling='Single target',
    combined_rolling='Combined', pair_rolling='Two targets',
    pair_rollout='Rollout', pair_stopping='Stopping')


def main():
    source = json.loads((HERE/'benchmark.json').read_text())
    rows = source['measurements']
    model = [row for row in rows if row['scenario'] == 'model_world0']
    cells = [dict(root_index=row['root_index'], scenario=row['scenario'], policy=row['policy'],
        travel_loss=row['quality']['loss'], status=row['quality']['status'],
        cold_cpu=row['cold']['total_process_cpu'], warm_cpu=row['warm']['total_process_cpu'],
        cold_wall=row['cold']['total_wall'], warm_wall=row['warm']['total_wall'],
        identical_trace_and_outcome=row['identical_public_trace_and_outcome']) for row in rows]
    unsupported = []
    for index in source['root_indices']:
        affected = [row for row in cells if row['root_index'] == index and row['travel_loss'] is None]
        if affected:
            unsupported.append(dict(root_index=index, scenarios=sorted({r['scenario'] for r in affected}),
                policies=[r['policy'] for r in affected], statuses=sorted({r['status'] for r in affected}),
                travel_loss=None))
    summary = dict(complete=source['complete'], cells=len(cells),
        identical_replays=sum(row['identical_trace_and_outcome'] for row in cells),
        environment=source['environment'], data_io_wall=source['data_io_wall'],
        cold_model_world_cpu_range=[min(r['cold']['total_process_cpu'] for r in model),
                                   max(r['cold']['total_process_cpu'] for r in model)],
        warm_model_world_cpu_range=[min(r['warm']['total_process_cpu'] for r in model),
                                   max(r['warm']['total_process_cpu'] for r in model)],
        model_world_losses={str(index):sorted({r['quality']['loss'] for r in model if r['root_index']==index})
                            for index in source['root_indices']},
        unsupported=unsupported, measurements=cells,
        interpretation='Warm is repeated replay of the identical public history, not acceleration on a new journey. '
                       'Wall times include concurrent-host contention; compare total process CPU. '
                       'These two roots provide no new algorithm quality gain.')
    (HERE/'benchmark_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')

    colors = dict(zip(LABELS, plt.get_cmap('tab10').colors))
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.9), sharex=True)
    for axis, index in zip(axes, source['root_indices']):
        selected = [row for row in model if row['root_index'] == index]
        loss = selected[0]['quality']['loss']
        for row in selected:
            color = colors[row['policy']]
            axis.scatter(row['cold']['total_process_cpu'], row['quality']['loss'],
                         color=color, marker='o', s=66, alpha=.85, zorder=3)
            axis.scatter(row['warm']['total_process_cpu'], row['quality']['loss'],
                         edgecolors=[color], facecolors='none', marker='D', s=60, linewidths=1.7, zorder=3)
        axis.set_xscale('log')
        axis.set_xlim(summary['warm_model_world_cpu_range'][0]*.65,
                      summary['cold_model_world_cpu_range'][1]*1.4)
        axis.set_ylim(loss-20, loss+20)
        axis.set_yticks([loss-20, loss, loss+20])
        axis.set_title(f'Root {index} / model world 0')
        axis.set_xlabel('Total process CPU (seconds; log scale)')
        axis.grid(True, alpha=.23)
        axis.text(.5, .90, f'All six policies: {loss:g} seconds', transform=axis.transAxes,
                  ha='center', fontsize=10)
    axes[0].set_ylabel('Truncated travel loss (seconds)')
    phase = [Line2D([],[],color='black',marker='o',linestyle='none',label='Fresh model (cold)'),
             Line2D([],[],color='black',marker='D',markerfacecolor='none',linestyle='none',label='Identical history (warm)')]
    axes[0].legend(handles=phase, loc='lower left', fontsize=8)
    handles = [Line2D([],[],color=colors[key],marker='o',linestyle='none',label=label)
               for key,label in LABELS.items()]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.115),ncol=6,frameon=False,fontsize=9)
    fig.suptitle('Frozen controls: equal travel quality, charged CPU cost',fontsize=13)
    fig.text(.5,.070,'Actual replay: root 0 unsupported for all six policies; root 1 loss 894 s for all six (not plotted).',
             ha='center',fontsize=9)
    fig.text(.5,.025,'Warm repeats the identical public history. No acceleration on new journeys is established.',
             ha='center',fontsize=9)
    fig.subplots_adjust(left=.09,right=.98,top=.85,bottom=.28,wspace=.23)
    fig.savefig(HERE/'quality_cpu.png',dpi=180)
    plt.close(fig)
    print(json.dumps({key:summary[key] for key in ('cells','identical_replays',
        'cold_model_world_cpu_range','warm_model_world_cpu_range','model_world_losses')}))


if __name__ == '__main__':
    main()
