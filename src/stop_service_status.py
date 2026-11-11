"""Use received stop restrictions when authorizing an alighting plan."""


def stop_status_at(receipts, time, stop_sequence):
    """Latest explicit stop status from one dated trip's received updates.

    Absent stop rows supply no new evidence. An explicit later status can
    supersede an earlier skip. The caller separately maintains trip identity.
    """
    latest_time, latest_status = None, None
    for update in receipts:
        received = update['receipt_seconds']
        if received > time or (latest_time is not None and received < latest_time):
            continue
        for stop in update['stops']:
            if stop['sequence'] == stop_sequence:
                latest_time, latest_status = received, stop['status']
    return latest_status


def available_alighting_leg(state, cid, stop_sequence, receipts):
    """An issued stay-to-stop certificate requires a served target endpoint.

    SKIPPED affects this alighting claim. The vehicle can continue through the
    station. Neither the whole trip nor its earlier legs are canceled here.
    """
    return cid not in state.unavailable_ids and stop_status_at(receipts, state.time, stop_sequence) != 'SKIPPED'
