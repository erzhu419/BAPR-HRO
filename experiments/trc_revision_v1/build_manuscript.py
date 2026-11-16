"""Build the TR-C evaluation manuscript from retained and development receipts."""
from pathlib import Path
import json
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT / 'experiments'
PAPER = ROOT / 'paper'
LABELS = {
    'cheap_statmax_ideal': 'Analytical, station/ideal',
    'full128_statmax_ideal': 'Rollout128, station/ideal',
    'cheap_plat_async': 'Analytical, pair/measured',
    'cheap_plat_ideal': 'Analytical, pair/ideal',
    'snapshot_csa_plat_async': 'Snapshot scan, pair/measured',
    'recursive_plat_async': 'Recursive, pair/measured',
    'recursive_plat_ideal': 'Recursive, pair/ideal',
    'screen64_plat_async': 'Screen64, pair/measured',
    'full128_plat_ideal': 'Rollout128, pair/ideal',
    'full128_plat_async': 'Rollout128, pair/measured',
}


def read(path):
    return json.loads(path.read_text())


def number(value, digits=2, signed=False):
    return '--' if value is None else format(value, ('+' if signed else '') + f'.{digits}f')


def interval(values, digits=2):
    return '$[' + ','.join(number(v, digits=digits, signed=True) for v in values) + ']$'


def table(caption, columns, rows, widths=None):
    spec = widths or ('l' + 'r' * (len(columns)-1))
    body = ['\\begin{table}[ht]', '\\centering\\small\\setlength{\\tabcolsep}{4pt}', '\\caption{' + caption + '}',
            '\\begin{tabular}{' + spec + '}', '\\toprule',
            ' & '.join(columns) + r' \\', '\\midrule']
    body.extend(' & '.join(map(str, row)) + r' \\' for row in rows)
    return '\n'.join(body + ['\\bottomrule', '\\end{tabular}', '\\end{table}'])


def oriented_comparison(comparisons, baseline, subject):
    key = baseline+'__vs__'+subject
    if key in comparisons:
        return comparisons[key]
    original = comparisons[subject+'__vs__'+baseline]
    value = dict(original,baseline=baseline,subject=subject,
        wins_count=original['losses_count'],losses_count=original['wins_count'],
        complete_mean_saving_seconds=-original['complete_mean_saving_seconds'],
        all_roots_missingness_saving_bounds=[-v for v in original['all_roots_missingness_saving_bounds'][::-1]])
    value['date_cluster_bootstrap'] = dict(original['date_cluster_bootstrap'],
        ci95_saving_seconds=[-v for v in original['date_cluster_bootstrap']['ci95_saving_seconds'][::-1]])
    return value


def revision_results():
    """The complete corrected summary is the only source for new results."""
    path = HERE / 'summary.json'
    if not path.exists():
        return (r'The corrected development replay is in progress; no new quality '
                r'or timing result is reported in this draft.',
                'The corrected development comparison is reported separately from the retained historical results.')
    result = read(path)
    aggregate = result['aggregate']
    quality_rows = [[LABELS[label], value['complete_roots'],
        number(value['complete_mean_loss_seconds']),
        interval(value['all_root_loss_bounds']), number(value['mean_cpu_seconds']),
        number(value['p99_cpu_seconds'])] for label,value in aggregate.items()]
    quality = table('Corrected development replay: truncated travel loss and total journey process CPU. Pair denotes the planned-platform transfer contract; measured denotes computation-aware activation. All-root bounds retain unresolved journeys.',
        ['Policy and contract', 'Known', 'Loss (s)', 'All-60 bound (s)', 'CPU (s)', 'CPU $p_{99}$'],
        quality_rows, 'p{.34\\textwidth}rrrrr')
    comparisons = result['comparisons']
    requested = [
        ('Platform vs. station, analytical ideal', 'cheap_statmax_ideal', 'cheap_plat_ideal'),
        ('Measured vs. ideal, analytical pair', 'cheap_plat_ideal', 'cheap_plat_async'),
        ('Recursive vs. analytical, pair ideal', 'cheap_plat_ideal', 'recursive_plat_ideal'),
        ('Measured vs. ideal, recursive pair', 'recursive_plat_ideal', 'recursive_plat_async'),
        ('Rollout vs. analytical, pair ideal', 'cheap_plat_ideal', 'full128_plat_ideal'),
        ('Measured vs. ideal, rollout pair', 'full128_plat_ideal', 'full128_plat_async'),
        ('Recursive vs. analytical, pair measured', 'cheap_plat_async', 'recursive_plat_async'),
        ('Screen vs. analytical, pair measured', 'cheap_plat_async', 'screen64_plat_async'),
        ('Rollout vs. analytical, pair measured', 'cheap_plat_async', 'full128_plat_async'),
    ]
    rows = []
    for name,base,subject in requested:
        c = oriented_comparison(comparisons,base,subject)
        counts = '/'.join(str(c[k]) for k in ('wins_count','losses_count','ties_count'))
        counts += '/' + str(len(c['unresolved_roots']))
        rows.append([name, counts, number(c['complete_mean_saving_seconds'],signed=True),
            interval(c['date_cluster_bootstrap']['ci95_saving_seconds']),
            interval(c['all_roots_missingness_saving_bounds'])])
    paired = table('Matched corrected comparisons. Positive saving favors the first named policy; W/L/T/U counts wins, losses, ties and unknown pairs. Date intervals are descriptive on five exposed dates.',
        ['Comparison', 'W/L/T/U', 'Saving (s)', 'Date interval (s)', 'All-60 bound (s)'], rows,
        'p{.34\\textwidth}rrrr')
    execution = result['execution']
    audit = read(HERE/'execution_audit.json')['checked']
    root12 = {r['arm']:r for r in read(HERE/'root12_execution_diagnostic.json')}
    root12_publications = {key:next(p for p in row['publications']
        if p['kind']=='onboard_publication') for key,row in root12.items()}
    scoring_cases = read(HERE/'scoring_decision_cases.json')['full_ideal_scoring_losses']
    budget_rows = []
    for label,stats in execution.items():
        budgets = stats['query_budget_receipts']
        if not budgets and not stats['base_exhausted_query_allowance']:
            continue
        bank_counts = '/'.join(str(stats[key]) for key in
            ('fully_computed_banks','fully_resolved_banks','gates'))
        budget_rows.append([LABELS[label], bank_counts,
            sum(b['interrupted'] for b in budgets), number(max((b['requested_cpu'] for b in budgets),default=None),3),
            number(max((b['actual_cpu'] for b in budgets),default=None),3),
            number(max((b['overshoot_seconds'] for b in budgets),default=None),3)])
    budget = table('Sampled-evaluation allowance after deducting analytical-base CPU, including posterior preparation. C/R/N counts fully computed, fully resolved, and requested banks. A computed world can have unknown outcomes. Max CPU and overrun concern the sampled call.',
        ['Policy', 'C/R/N', 'Stopped', 'Max budget', 'CPU max.', 'Overrun max.'],
        budget_rows, 'p{.30\\textwidth}rrrrr')
    sensitivity_rows = []
    for arm,by_tau in result['sensitivity'].items():
        for tau,value in by_tau.items():
            sensitivity_rows.append([LABELS[arm],tau,value['complete_roots'],
                number(value['complete_mean_loss']),interval(value['all_root_loss_bounds'])])
    sensitivity = table('Same-platform allowance sensitivity in the corrected analytical policies. The 60 s and 120 s settings have no rollout comparator; 90 s repeats the primary setting.',
        ['Policy','Assumed time (s)','Known','Loss (s)','All-60 bound (s)'],
        sensitivity_rows,'p{.36\\textwidth}rrrr')
    base,full = aggregate['cheap_plat_async'], aggregate['full128_plat_async']
    contrast = comparisons['cheap_plat_async__vs__full128_plat_async']
    ideal_contrast = oriented_comparison(comparisons,'cheap_plat_ideal','full128_plat_ideal')
    clock_contrast = oriented_comparison(comparisons,'full128_plat_ideal','full128_plat_async')
    abstract = ('The corrected development replay retains all 60 queries: sampled '
        f'continuation increases complete-pair mean loss by {number(-contrast["complete_mean_saving_seconds"])} s '
        'relative to the analytical rule; the all-query loss-increase bound is '
        f'{interval([-v for v in contrast["all_roots_missingness_saving_bounds"][::-1]])} s. '
        f'Mean journey process CPU is {number(base["mean_cpu_seconds"])} s for the analytical rule '
        f'and {number(full["mean_cpu_seconds"])} s for sampled continuation.')
    roots_unknown = ', '.join(map(str,result['equality']['roots_with_any_unknown'])) or 'none'
    body = [
        f'The corrected study contains {result["core_receipts"]} primary journey replays '
        f'and {result["sensitivity_additional_receipts"]} additional transfer-time sensitivity replays '
        f'on the same {result["roots"]} selected queries. Model fits, menus, external '
        'messages, and declared query allowances remain fixed. The common platform '
        'command policy changes as described above; these are development results.',
        quality, paired,
        f'Whole-journey mean CPU is {number(base["mean_cpu_seconds"])} s for the analytical '
        f'pair/measured arm and {number(full["mean_cpu_seconds"])} s for rollout. '
        f'Their {contrast["complete_pairs"]} complete pairs contain '
        f'{contrast["wins_count"]} rollout gains, {contrast["losses_count"]} harms and '
        f'{contrast["ties_count"]} ties. Across all ten arms, unresolved query indices are '
        f'{roots_unknown}; every such query is retained in the bounds.',
        'The matched decomposition separates the sampled policy from its measured clock. '
        f'At zero latency, rollout has {number(-ideal_contrast["complete_mean_saving_seconds"])} s '
        'larger complete-pair mean loss than the analytical rule. Applying its measured '
        f'computation clock adds {number(-clock_contrast["complete_mean_saving_seconds"])} s; '
        'the all-root loss-increase bound for this clock contrast is '
        f'{interval([-v for v in clock_contrast["all_roots_missingness_saving_bounds"][::-1]])} s. '
        'The recursive rule matches the analytical loss on every resolved pair while '
        f'using {number(aggregate["recursive_plat_async"]["mean_cpu_seconds"])} s mean journey CPU.',
        'The three zero-latency harms (queries '
        + ', '.join(str(case['root_index']) for case in scoring_cases)
        + ') repeat the same exit pattern: a point-estimate comparison changes '
        'continuation to alighting. Each bank has 128 resolved pairs, but all three '
        'approximate 95\\% intervals for the action difference contain zero, indicating '
        'weak action separation in those saved banks.',
        'Query 12 illustrates why computation records matter for attribution. Its '
        f'{number(root12_publications["screen64_plat_async"]["closes_at"]-root12_publications["screen64_plat_async"]["opens_at"],0)} s '
        'recorded alighting window contains the recursive arm\'s '
        f'{number(root12_publications["recursive_plat_async"]["base_cpu_seconds"],3)} s base computation, '
        'whereas the same recursive base in the screen arm takes '
        f'{number(root12_publications["screen64_plat_async"]["base_cpu_seconds"],3)} s. '
        'The latter misses the cutoff before any sampled call. The screen arm\'s '
        'single harmful query therefore reflects this measured-runtime difference '
        'crossing a decision boundary. On the same query, rollout takes '
        f'{number(root12_publications["full128_plat_async"]["background_cpu_seconds"],3)} s '
        'after its base calculation, exceeding the window during sampled evaluation.',
        budget,
        'The analytical base exhausted the query allowance before sampling in '
        + ', '.join(LABELS[label]+': '+str(stats['base_exhausted_query_allowance'])
            for label,stats in execution.items() if stats['base_exhausted_query_allowance'])
        + ' decisions. These cases issue no sampled query.',
        f'Across its {execution["screen64_plat_async"]["gates"]} sampled queries, '
        'the screen retains the base proposal in every case.',
        'The normal screen requires known outcomes for every requested pair and at least '
        'two pairs. Interrupted computations and incompatible physical projections remain '
        'unresolved. Query '
        'allowances and whole-journey costs are different quantities: the latter also '
        'include other decisions and inherited menu construction.',
        f'Saved-receipt checks cover {audit["core_receipts"]} primary replays and '
        f'{audit["platform_command_precedes_departure"]} command-controlled boardings. '
        f'For discretionary alighting, {audit["alight_consumption_equals_platform_entry"]} '
        'subsequent platform entries match the recorded action-consumption clock; '
        f'{audit["alight_with_no_later_platform_record"]} alightings have no later platform record '
        'and remain outside that entry-clock check. This verifies the available replay records.',
        sensitivity,
    ]
    calibration_path = HERE/'calibration_summary.json'
    if calibration_path.exists():
        calibration = read(calibration_path)['contracts']
        rows = []
        for key in ('00','01','10','11'):
            row = ['$P_{'+key+'}$']
            for name in ('station_max','platform_pair'):
                for weight in ('equal_incident_then_equal_date','equal_event_then_equal_date'):
                    row.append(interval(calibration[name]['weightings'][weight]['bootstrap']['marginal95'][key],digits=4))
            rows.append(row)
        body += [table('Conditional first-onboard snapshot calibration: descriptive date-bootstrap intervals for observed minus predicted joint probability. Readiness uses arrival plus transfer time. I weights incidents equally within a date; E weights events equally within a date. Dates have equal weight.',
            ['Class','Station I','Station E','Pair I','Pair E'],rows,'lrrrr')]
        for name,label in [('station_max','station'),('platform_pair','platform-pair')]:
            c = calibration[name]
            statuses = ', '.join(k.replace('_',' ')+' '+str(v) for k,v in c['initial_state_statuses'].items())
            body.append(f'The {label} calibration contains {c["registered_roots"]} roots '
                f'({statuses}) and {c["registered_events"]} opportunities, of which '
                f'{c["reached_events"]} are reached. Roots without opportunities remain '
                'in the availability denominator.')
    build_revision_figure(result)
    body += [r'\begin{figure}[ht]\centering',
        r'\includegraphics[width=\textwidth]{trc_revision_quality.pdf}',
        r'\caption{Corrected development comparison under the platform-pair, measured-clock contract. Left: complete-case loss versus whole-journey process CPU; vertical segments retain all-root missing-outcome bounds. Right: every root in two paired comparisons; crosses at the lower edge mark unresolved pairs. Positive saving favors the challenger.}\end{figure}']
    rendered = '\n\n'.join(body)
    (HERE/'development_results.tex').write_text(rendered+'\n')
    (HERE/'manuscript_results.json').write_text(json.dumps(dict(abstract_sentence=abstract,
        source='summary.json and calibration_summary.json'),indent=2)+'\n')
    return rendered, abstract


def build_revision_figure(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,(ax,bx) = plt.subplots(1,2,figsize=(11,4),layout='constrained')
    entries = [('cheap_plat_async','Analytical','C0'),('recursive_plat_async','Recursive','C1'),
        ('snapshot_csa_plat_async','Snapshot','C2'),('screen64_plat_async','Screen64','C3'),
        ('full128_plat_async','Rollout128','C4')]
    for label,name,color in entries:
        a = result['aggregate'][label]
        x,y = a['mean_cpu_seconds'],a['complete_mean_loss_seconds']
        ax.plot([x,x],a['all_root_loss_bounds'],color=color,alpha=.5,lw=2)
        ax.scatter(x,y,color=color,s=35,label=name,zorder=3)
    ax.set(xscale='log',xlabel='Mean journey process CPU (s, log)',ylabel='Mean truncated loss (s)')
    ax.legend(fontsize=8,frameon=False);ax.grid(alpha=.2)
    all_gains=[]
    for label,_,_ in entries:
        if label=='cheap_plat_async':continue
        c=result['comparisons']['cheap_plat_async__vs__'+label]
        all_gains += [row['saving_seconds'] for row in c['wins']+c['losses']]
    low=min([0]+all_gains);high=max([0]+all_gains);bottom=low-.12*max(1,high-low)
    for subject,name,color,offset in [('full128_plat_async','Rollout128','C4',-.16),
            ('recursive_plat_async','Recursive','C1',.16)]:
        c=result['comparisons']['cheap_plat_async__vs__'+subject]
        known={i:0 for i in range(c['total_roots']) if i not in c['unresolved_roots']}
        known.update({row['root_index']:row['saving_seconds'] for row in c['wins']+c['losses']})
        bx.scatter([i+offset for i in known],list(known.values()),s=14,color=color,label=name)
        bx.scatter([i+offset for i in c['unresolved_roots']],
            [bottom]*len(c['unresolved_roots']),marker='x',color=color,s=28)
    bx.axhline(0,color='0.5',lw=.7);bx.set(xlabel='Registered query index',
        ylabel='Saving over analytical (s)',xlim=(-1,60))
    bx.legend(fontsize=8,frameon=False);bx.grid(alpha=.2)
    fig.savefig(PAPER/'figures/trc_revision_quality.pdf')
    fig.savefig(PAPER/'figures/trc_revision_quality.png',dpi=180)
    plt.close(fig)


def build():
    ci = read(EXP/'control_identity_v1/summary.json')
    dg = read(EXP/'decision_gap_v1/summary.json')
    v50 = read(EXP/'reconstruction_v50/summary.json')
    ng = read(EXP/'natural_gap_v2/summary_compact.json')
    bg = read(EXP/'bounded_gate_5d_v1/summary.json')
    off = read(EXP/'g2_reset_v1/OFFLINE_AUDIT_SUMMARY.json')
    cal = read(EXP/'bounded_gate_5d_v1/calibration_split_summary.json')
    fit = read(EXP/'control_identity_v1/repaired_fit.json')
    old = ci['models']['old_fit']['aggregate']['new_cheap']['complete_mean_loss']
    new = ci['models']['repaired_fit']['aggregate']['new_cheap']['complete_mean_loss']
    ngc = ng['comparisons']['new_cheap__new_full128']
    bgc = bg['comparisons']['cheap_statmax_ideal__vs__cheap_plat_async']
    dc = off['finding_1_codex_calibration_split_verdict']['decision_calibration_v2_split']['conditional_summary']
    dctotal = dc['reached_resolved_opportunities'] + dc['reached_unresolved_opportunities']
    ngtotal = off['finding_1_codex_calibration_split_verdict']['natural_gap_v2_conditional_onboard']['total_registered_opportunities']
    total = dctotal + ngtotal + cal['metric_c1_initial_availability']['registered_opportunities']
    ngcal = off['finding_1_codex_calibration_split_verdict']['natural_gap_v2_conditional_onboard']
    resolved = dc['reached_resolved_opportunities'] + ngcal['conditional_reached_opportunities'] + cal['metric_c2_conditional_onboard_calibration']['reached_resolved_opportunities']
    cohorts = table('Retained data panels and the role of the corrected replay.',
        ['Service dates (2023)', 'Days', 'Queries', 'Role'], [
        ['October 25--28, 30--31',6,72,'Historical development'],
        ['March 9--22',14,168,'Conditional calibration'],
        ['March 23--April 5',14,168,'Station-level control'],
        ['April 6--10',5,60,'Historical comparison'],
        ['April 6--10 (same queries)',5,60,'Corrected development replay']],
        'lrrp{.32\\textwidth}')
    narrow = v50['aggregate']['narrow_common']['mean_loss_bounds'][0]
    wide = v50['aggregate']['wide_common']['mean_loss_bounds'][0]
    dgloss = dg['aggregate']['new_cheap']['complete_mean_loss']
    developments = table('October development comparisons. The cross-stage row is not an isolated factor.',
        ['Comparison', 'Before (s)', 'After (s)', 'Saving (s)'], [
        ['Local departure cap, common observations',number(narrow),number(wide),number(narrow-wide)],
        ['Prefix closure, common observations',number(wide),number(dgloss),number(wide-dgloss)],
        ['Cross-stage source/implementation change',number(dgloss),number(old),number(dgloss-old)],
        ['Old vs. repaired likelihood, same corrected stream',number(old),number(new),number(old-new)]],
        'p{.49\\textwidth}rrr')
    contrasts = [
        ('Platform-pair analytical vs. station-level', 'cheap_statmax_ideal__vs__cheap_plat_async'),
        ('Recursive vs. analytical, platform-pair', 'cheap_plat_async__vs__recursive_plat_async'),
        ('Gated vs. analytical, platform-pair', 'cheap_plat_async__vs__bounded_cert_plat_async'),
        ('Full rollout vs. analytical, platform-pair', 'cheap_plat_async__vs__full128_plat_async')]
    rows = []
    for name,key in contrasts:
        c = bg['comparisons'][key]
        rows.append([name, c['complete_pairs'], number(c['complete_mean_saving_seconds'],signed=True),
                     interval(c['date_cluster_bootstrap']['ci95_saving_seconds']),
                     interval(c['all_roots_missingness_saving_bounds'])])
    historical = table('Retained five-day comparisons before the execution revision. Positive saving favors the first named policy.',
        ['Comparison','Pairs','Saving (s)','Date interval (s)','All-60 bound (s)'], rows,
        'p{.32\\textwidth}rrrr')
    revised, abstract = revision_results()
    contract = (HERE/'execution_contract.tex').read_text()
    weighted = read(HERE/'historical_weighting_audit.json')['weightings']
    residuals = [weighted[key]['date_cluster_inference']['mean_residual']['10']
        for key in ('equal_incident_then_equal_date','equal_event_then_equal_date')]
    runtime = read(HERE/'runtime_context.json')
    development_history = ''
    if (HERE/'summary.json').exists():
        ledger = read(HERE/'summary.json')
        valid = ledger['valid_controller_calls']
        additional = ledger['total_completed_controller_calls_including_invalid_attempts']-valid
        development_history = (
            f'The retained record contains {valid} final-controller replay receipts '
            f'and {additional} additional completed development-attempt receipts. '
            'Some interrupted attempts ended before a complete receipt was written; '
            'their total count is not fully recoverable. These attempts are archived '
            'separately from the final paired panel.')
    values = dict(COHORT_TABLE=cohorts, DEVELOPMENT_TABLE=developments,
        HISTORICAL_CONTRAST_TABLE=historical, BG_GAIN=number(bgc['complete_mean_saving_seconds'],signed=True),
        BG_CI=interval(bgc['date_cluster_bootstrap']['ci95_saving_seconds']),
        DEP_BIAS=number(fit['pooled']['departure']['bias_seconds']),
        ARR_BIAS=number(fit['pooled']['arrival']['bias_seconds']),
        EVENT_TOTAL=f'{total:,}', EVENT_RESOLVED=f'{resolved:,}', DC_UNKNOWN=dc['reached_unresolved_opportunities'],
        DG_LOSS=number(dgloss), CI_OLD_LOSS=number(old), CI_NEW_LOSS=number(new),
        NG_GAIN=number(ngc['complete_pair_mean_gain'],signed=True),
        NG_BOUND=interval(ngc['all_root_mean_gain_bounds']),
        NG_CHEAP_CPU=number(ng['aggregate']['new_cheap']['mean_cpu']),
        NG_FULL_CPU=number(ng['aggregate']['new_full128']['mean_cpu']),
        REC_LOSS=number(-bg['comparisons']['cheap_plat_async__vs__recursive_plat_async']['complete_mean_saving_seconds']),
        GATE_LOSS=number(-bg['comparisons']['cheap_plat_async__vs__bounded_cert_plat_async']['complete_mean_saving_seconds']),
        CALIBRATION_WEIGHTING='Recomputing the retained March conditional $P_{10}$ residual '
            'from the same saved opportunities gives '+number(residuals[0],5)
            +' with incident weights and '+number(residuals[1],5)
            +' with event weights. Both are positive; the difference is a weighting effect. '
            'The original calibration gate remains unresolved.',
        REVISION_CONTRACT=contract, REVISION_RESULTS=revised, REVISION_ABSTRACT=abstract,
        RUNTIME_CONTEXT='The corrected replays ran on '+runtime['processor_model'].replace('(R)','')
            +f' with {runtime["logical_cpus"]} logical CPUs and {runtime["workers"]} parallel workers. '
            f'The software environment used Python {runtime["python"]}, NumPy {runtime["numpy"]}, '
            f'and SciPy {runtime["scipy"]}; OpenBLAS, OpenMP, and MKL thread settings were one. '
            'Each journey uses its own process CPU as replay latency; wall time is recorded separately.',
        DEVELOPMENT_HISTORY=development_history,
        BIBLIOGRAPHY=(HERE/'references.tex').read_text())
    text = (HERE/'manuscript.tex.in').read_text()
    for key,value in values.items():
        text = text.replace('@'+key+'@',str(value))
    unresolved = re.findall(r'@[A-Z_]+@',text)
    if unresolved:
        raise ValueError(f'Unresolved manuscript variables: {unresolved}')
    (PAPER/'paper_reconstruction.tex').write_text(text)
    trace = dict(generator='experiments/trc_revision_v1/build_manuscript.py',
        source_template='manuscript.tex.in',old_manuscript_archive='history/manuscript_before_revision.tar.gz',
        input_receipts=['control_identity_v1/summary.json','decision_gap_v1/summary.json',
            'reconstruction_v50/summary.json','natural_gap_v2/summary_compact.json',
            'bounded_gate_5d_v1/summary.json','g2_reset_v1/OFFLINE_AUDIT_SUMMARY.json',
            'bounded_gate_5d_v1/calibration_split_summary.json','control_identity_v1/repaired_fit.json'],
        revision_receipts=['trc_revision_v1/summary.json','trc_revision_v1/calibration_summary.json',
            'trc_revision_v1/historical_weighting_audit.json','trc_revision_v1/runtime_context.json',
            'trc_revision_v1/execution_audit.json','trc_revision_v1/root12_execution_diagnostic.json',
            'trc_revision_v1/scoring_decision_cases.json'],
        current_figure='paper/figures/trc_revision_quality.pdf',
        development_result=(HERE/'summary.json').exists(),source='Existing historical files are read only.',
        scientific_scope='Retrospective transport evaluation; no new optimality or finite-sample certification claim.')
    (HERE/'manuscript_sources.json').write_text(json.dumps(trace,indent=2)+'\n')
    index = '''# Current manuscript and evidence

The current paper is the TR-C development revision, not a new optimality or action-certification claim. Historical receipts remain unchanged. Corrected April replays are development evidence; the original 72 roots receive no additional budget.

- [Current manuscript](../paper/paper_reconstruction.tex) and [PDF](../paper/paper_reconstruction.pdf).
- [Sole current generator](../experiments/trc_revision_v1/build_manuscript.py), [template](../experiments/trc_revision_v1/manuscript.tex.in), [references](../experiments/trc_revision_v1/references.tex), and [build instructions](../experiments/trc_revision_v1/MANUSCRIPT.md).
- [Frozen development protocol](../experiments/trc_revision_v1/protocol.json), [execution contract](../experiments/trc_revision_v1/execution_contract.tex), [executor](../experiments/trc_revision_v1/executor.py), [interruptible rollout](../experiments/trc_revision_v1/rollout.py), and [calibration](../experiments/trc_revision_v1/calibration.py).
- [Current result summary](../experiments/trc_revision_v1/summary.json), [calibration summary](../experiments/trc_revision_v1/calibration_summary.json), and [manuscript source receipts](../experiments/trc_revision_v1/manuscript_sources.json).
- [Execution-record checks](../experiments/trc_revision_v1/execution_audit.json) and [saved scoring and runtime cases](../experiments/trc_revision_v1/scoring_decision_cases.json) give the local attribution evidence.
- Historical evidence remains in [control identity](../experiments/control_identity_v1/summary.json), [natural events](../experiments/natural_gap_v2/summary_compact.json), and [original five-day comparison](../experiments/bounded_gate_5d_v1/summary.json). Their contracts and dates differ; cross-stage gains are not additive causal effects.
- [Previous index](../experiments/trc_revision_v1/history/EVIDENCE_TRACEABILITY_INDEX_before_revision.md) and [previous manuscript and generators](../experiments/trc_revision_v1/history/manuscript_before_revision.tar.gz) are preserved for historical reconstruction only.

Run `python3 -m experiments.trc_revision_v1.build_manuscript` from the repository root. The old five-day manuscript generator is historical and does not generate the current paper.
'''
    (ROOT/'md/EVIDENCE_TRACEABILITY_INDEX.md').write_text(index)
    print('Generated paper/paper_reconstruction.tex from trc_revision_v1 sources')


if __name__=='__main__':
    build()
