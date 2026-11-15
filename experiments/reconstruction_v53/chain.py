"""Exact integration of a same-trip suffix with no remaining choice.

The root law already includes received evidence. Its posterior (Doob)
transitions incorporate future emissions, so emissions are not multiplied
again. Intermediate observations can be summed out because the first retained
exit is absorbing. No observation is removed before a subsequent decision.
"""
import numpy as np

from experiments.reconstruction_v41.physical_posterior import posterior_transition


def terminal_ride_value(root_law, history, ride, position, clock, deadline):
    """Return absolute expected truncated arrival, or None if not applicable.

    Only the suffix up to its first retained exit is considered; that exit
    must be None. Each executed departure/arrival uses the same cancellation,
    time-order and deadline checks as V52._ride. Hidden chain slots transmit
    probability but acquire no extra service restriction. The caller charges
    this finite matrix calculation to its original query budget.
    """
    path=[]
    for cid in ride.path[position:]:
        path.append(cid)
        if cid in ride.exits:
            if ride.exits[cid] is not None:
                return None
            break
    else:
        return None
    positions=[root_law.model._positions[cid,kind]
               for cid in path for kind in ('departure','arrival')]
    trip=positions[0][0]
    if any(other!=trip for other,slot in positions):
        return None
    slots=[slot for other,slot in positions]
    if any(right<=left for left,right in zip(slots,slots[1:])):
        return None
    if clock>=deadline:
        return float(deadline)
    law=root_law.law(trip,history)
    first=slots[0]
    mass=law.marginal(first).copy()
    times=law.times[first]
    mass*=~law.canceled[first] & (times>=clock) & (times<deadline)
    for left,right in zip(slots,slots[1:]):
        kernel=posterior_transition(law,left)
        for hidden in range(left+1,right):
            kernel=kernel @ posterior_transition(law,hidden)
        target=law.times[right]
        legal=(~law.canceled[right] & (target<deadline))[None,:]
        legal=legal & (target[None,:]>=law.times[left][:,None])
        mass=mass @ (kernel*legal)
    # All mass removed by a failed executed event pays H. This equivalent
    # expression avoids separately subtracting nearly equal success masses.
    value=float(deadline+np.dot(mass,law.times[slots[-1]]-deadline))
    return min(float(deadline),max(min(float(clock),float(deadline)),value))
