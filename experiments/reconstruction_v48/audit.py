"""Independent V48 scheduling and saved-certificate checks.

The full-scan influence reference is used on small DAGs, never as a natural
performance competitor. Frozen Monte Carlo estimates are not population truth.
"""
from collections import deque
from math import isfinite

import numpy as np

from experiments.reconstruction_v47 import audit as previous


def full_scan_weights(root, *, is_block):
    """Sum every physical parent path; stop at an opaque bulk block.

    Zero-weight bulk children are work dependencies, not physical successors.
    They remain the bulk block's responsibility and are not probability-pruned.
    """
    nodes, edges, incoming = {}, {}, {}
    stack = [root]
    while stack:
        node = stack.pop()
        if node in nodes:
            continue
        nodes[node] = node
        incoming.setdefault(node, 0)
        children = [] if is_block(node) or not node.expanded else [
            (np.longdouble(mass), child) for mass, child in node.children if mass > 0]
        edges[node] = children
        for _, child in children:
            incoming[child] = incoming.get(child, 0)+1
            stack.append(child)
    queue = deque(node for node in nodes if incoming[node] == 0)
    weights = {node: np.longdouble(0) for node in nodes}
    weights[root] = np.longdouble(1)
    processed = 0
    while queue:
        node = queue.popleft()
        processed += 1
        for mass, child in edges[node]:
            weights[child] += weights[node]*mass
            incoming[child] -= 1
            if incoming[child] == 0:
                queue.append(child)
    if processed != len(nodes):
        raise ValueError('The physical computation graph is not acyclic')
    return weights


def full_scan_priority(root, *, is_block):
    """Independent exposed-frontier priorities, including arithmetic nodes."""
    weights = full_scan_weights(root, is_block=is_block)
    memo = {}
    result = {}
    for node, weight in weights.items():
        block = is_block(node)
        if block and node.drained:
            continue
        if block or (not node.expanded and node.build is not None):
            value = node.value(memo)
            width = max(0., float(value[1]-value[0]))
            if width > 0:
                result[node] = weight*np.longdouble(width)
    return result


def first_certificate(row):
    for index, point in enumerate(row['checkpoints']):
        values(point, clock=row['clock'], deadline=row['deadline'])
        if point['certified_action'] is not None:
            return index, point
    return None, None


def values(point, *, clock, deadline):
    """Rounded zero unresolved mass is not proof that dependencies are done."""
    actions = point['actions']
    for arm in previous.ARMS:
        value = actions[arm]
        lo, hi, mass = (value[k] for k in ('lower', 'upper', 'unresolved_probability'))
        if not all(isfinite(v) for v in (lo, hi, mass)):
            raise ValueError(f'Nonfinite bounded value: {arm}')
        if lo < clock-previous.TOL or hi > deadline+previous.TOL or lo > hi+previous.TOL:
            raise ValueError(f'Invalid physical interval: {arm}')
        if mass < -previous.TOL or mass > 1+previous.TOL:
            raise ValueError(f'Invalid unresolved probability: {arm}')
        pending = value.get('pending_dependencies', 0)
        mean = value['mean']
        if mean is not None and (mass != 0 or pending > 0 or abs(mean-lo) > previous.TOL
                                 or abs(hi-lo) > previous.TOL):
            raise ValueError(f'Inexact value advertised as exact: {arm}')
        if mass == 0 and mean is None and pending <= 0:
            raise ValueError(f'Zero mass without exact value or retained unfinished dependencies: {arm}')
        if mass == 0 and mean is not None and abs(hi-lo) > previous.TOL:
            raise ValueError(f'Complete mass without a point value: {arm}')
    a, c = (actions[arm] for arm in previous.ARMS)
    gap = [a['lower']-c['upper'], a['upper']-c['lower']]
    if any(abs(x-y) > previous.TOL for x, y in zip(gap, point['difference_interval'], strict=True)):
        raise ValueError('Action gap does not match the two arm intervals')
    if point['certified_action'] != previous.certified(gap):
        raise ValueError('Incorrect deterministic action certification')
    return gap


def curve(checkpoints, *, clock, deadline, max_compiles=2000):
    last = None
    for point in checkpoints:
        values(point, clock=clock, deadline=deadline)
        count = point['counters']['charged_compiles']
        if not isinstance(count, int) or not 0 <= count <= max_compiles:
            raise ValueError('Compilation quota exceeded')
        if last is not None:
            if count < last['counters']['charged_compiles']:
                raise ValueError('Cumulative work decreased')
            for arm in previous.ARMS:
                old, new = last['actions'][arm], point['actions'][arm]
                if (new['lower'] < old['lower']-previous.TOL or new['upper'] > old['upper']+previous.TOL
                        or new['unresolved_probability'] > old['unresolved_probability']+previous.TOL):
                    raise ValueError(f'Incremental interval widened: {arm}')
            for field in ('process_cpu', 'wall'):
                if point['cost'][field] < last['cost'][field]-previous.TOL:
                    raise ValueError('Cumulative query cost decreased')
        last = point
    if last is None:
        raise ValueError('Missing checkpoint curve')
    return dict(checkpoints=len(checkpoints), nested=True,
                charged_compiles=last['counters']['charged_compiles'],
                certified_action=last['certified_action'])


def bounded_receipt(row, *, max_compiles=2000):
    """Validate all saved ranges, final state, phases and actual stopping."""
    result = curve(row['checkpoints'], clock=row['clock'],
                   deadline=row['deadline'], max_compiles=max_compiles)
    values(row, clock=row['clock'], deadline=row['deadline'])
    previous.matching_values(row['checkpoints'][-1], row)
    count = row['counters']['charged_compiles']
    if sum(phase['charged_compiles'] for phase in row['budget_allocation']) != count:
        raise ValueError('Phase work does not sum to the query budget')
    last = 0
    for phase in row['budget_allocation']:
        last += phase['charged_compiles']
        if phase['cumulative_compiles'] != last:
            raise ValueError('Phase cumulative work does not match spent work')
    first_index, first = first_certificate(row)
    if 'first_certificate' in row:
        saved = row['first_certificate']
        if (saved is None) != (first is None):
            raise ValueError('First-certificate metadata does not match the saved curve')
        if saved is not None:
            previous.matching_values(saved, first)
            if saved['charged_compiles'] != first['counters']['charged_compiles']:
                raise ValueError('First-certificate metadata uses the wrong work count')
    if row.get('stop_reason') == 'certificate' and row['certified_action'] is None:
        raise ValueError('Certificate stop has no valid action certificate')
    result.update(first_certificate_checkpoint=first_index,
                  first_certificate_count=None if first is None else first['counters']['charged_compiles'],
                  first_certificate_action=None if first is None else first['certified_action'])
    return result


def stopped_prefix(full, stopped):
    """Early stopping must return the first checked certificate, not a midpoint."""
    full_index, first_full = first_certificate(full)
    stopped_index, first_stop = first_certificate(stopped)
    if first_full is None:
        if first_stop is not None:
            raise ValueError('Early run certified but the matched full prefix did not')
        previous.matching_values(full, stopped)
        if full['counters']['charged_compiles'] != stopped['counters']['charged_compiles']:
            raise ValueError('Uncertified full and stop runs used different work')
        return dict(certified=False, prefix_equal=True)
    if first_stop is None:
        raise ValueError('Early-stop run missed a saved full-run certificate')
    previous.matching_values(first_full, first_stop)
    previous.matching_values(first_stop, stopped)
    count = first_full['counters']['charged_compiles']
    if (first_stop['counters']['charged_compiles'] != count
            or stopped['counters']['charged_compiles'] != count):
        raise ValueError('Early-stop run performed paid work after its certificate')
    # A phase-end snapshot can duplicate an immediately preceding budget one.
    # Every preceding saved scientific state must still be the same prefix.
    full_prefix = full['checkpoints'][:full_index+1]
    stop_prefix = stopped['checkpoints'][:stopped_index+1]
    if len(full_prefix) != len(stop_prefix):
        raise ValueError('Full and stop runs changed certificate checking cadence')
    for a, b in zip(full_prefix, stop_prefix, strict=True):
        previous.matching_values(a, b)
        if a['counters']['charged_compiles'] != b['counters']['charged_compiles']:
            raise ValueError('Full and stop runs changed their paid prefix')
    return dict(certified=True, prefix_equal=True, stopping_count=count,
                certified_action=stopped['certified_action'])


def selected_action(row, v45_action):
    """Uncertified results preserve the frozen V45 action."""
    expected = row['certified_action'] or v45_action
    if row['selected_action'] != expected:
        raise ValueError('Certificate or frozen V45 fallback action changed')
    return expected
