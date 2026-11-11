"""Paired outcomes and probability checks using only already public facts."""
import json
from pathlib import Path
import pickle
import re
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent


def known_probability_checks(row, graph):
    """Known shared-trip outcomes imply a point-valued first-attempt outcome.

    Message receipt uses poll order, not event timestamps. Vehicle delay after
    boarding is recovered from its public boarding time, never the hidden world.
    Unknown vehicle outcomes are left unlabelled rather than treated as truth.
    """
    messages = {m["event_id"]: m for m in row["messages"]}
    boarded = {}
    for event in row["events"]:
        if event["event_type"] == "board":
            cid = int(re.search(r"connection=(\d+)", event["details"])[1])
            boarded[cid] = event["time"]
    known, checks = {}, []
    poll_index, trip, last_arrival = 0, None, 490
    cap = row["spec"][5]

    def trip_key(c):
        return c.trip_id or f"connection:{c.id}"

    for prediction in row["predictions"]:
        current_poll = prediction["private_model_query"]
        for poll in row["polls"][poll_index:current_poll + 1]:
            for event_id in poll["event_ids"]:
                message = messages[event_id]
                c = graph.connections[int(event_id.split(":", 1)[0])]
                known[trip_key(c)] = (message["is_canceled"],
                                      message["actual_time"] - message["scheduled_time"])
        poll_index = current_poll + 1
        cid = prediction["connection"]
        c = graph.connections[cid]
        outcome = known.get(trip_key(c))
        probability = prediction.get("first_attempt_failure_probability")
        if outcome is not None and probability is not None:
            canceled, delay = outcome
            ready = max(prediction["time"], last_arrival + (
                0 if trip is None or trip == c.trip_id else graph.stops[c.dep_stop].min_transfer_time))
            departure = c.dep_time + delay
            catches = (not canceled and ready < 490 + cap
                       and ready <= departure <= min(490 + cap, ready + 12))
            truth = float(not catches)
            checks.append({"time": prediction["time"], "connection": cid,
                           "p_hat": probability, "p_from_public_facts": truth,
                           "absolute_error": abs(probability - truth),
                           "actual_kind": prediction.get("actual_kind")})
        kind = prediction.get("actual_kind")
        if kind == "board":
            known[trip_key(c)] = (False, boarded[cid] - c.dep_time)
            trip, last_arrival = trip_key(c), prediction["action_end"]
        elif kind == "cancel":
            known[trip_key(c)] = (True, 0)
    return checks


def readout(path, graph):
    data = json.loads(path.read_text())
    rows = data["methods"]
    by_case = {}
    for row in rows:
        by_case.setdefault(row["index"], {})[row["variant"]] = row
    paired = []
    for index, methods in sorted(by_case.items()):
        for factor, first, second in [
                ("rescue_given_conditioned", "conditioned_cap", "conditioned_rescue"),
                ("rescue_given_marginal", "marginal_cap", "marginal_rescue"),
                ("conditioning_given_rescue", "marginal_rescue", "conditioned_rescue"),
                ("conditioning_given_cap", "marginal_cap", "conditioned_cap")]:
            if first in methods and second in methods:
                paired.append({"index": index, "factor": factor,
                               "date": methods[first]["spec"][0],
                               "stress": methods[first]["spec"][1],
                               "delta_loss": methods[second]["loss"] - methods[first]["loss"]})
    checks = {}
    for row in rows:
        points = known_probability_checks(row, graph)
        if points:
            checks.setdefault(row["variant"], []).extend(
                dict(point, index=row["index"]) for point in points)
    check_summary = {name: {"n_known_selected_states": len(points),
                            "mean_absolute_error": float(np.mean([p["absolute_error"] for p in points])),
                            "max_absolute_error": max(p["absolute_error"] for p in points)}
                     for name, points in checks.items()}
    for name, points in checks.items():
        if name.startswith("conditioned") and any(p["absolute_error"] > 1e-12 for p in points):
            raise AssertionError(f"A known physical fact was not respected by {name}")
    legacy_mismatches = [{"index": r["index"], "old": r["legacy_archived_loss"], "new": r["loss"]}
                         for r in rows if r["variant"] == "legacy" and r["legacy_archived_loss"] is not None
                         and r["loss"] != r["legacy_archived_loss"]]
    if legacy_mismatches:
        raise AssertionError(f"Legacy behavior changed: {legacy_mismatches}")
    return {"source": str(path), "journeys": len(rows), "cases": len(by_case),
            "external_world_units": len({(r["spec"][0], r["spec"][1], r["spec"][4]) for r in rows}),
            "methods": data["summary"], "paired_factor_differences": paired,
            "known_fact_probability_checks": check_summary,
            "known_fact_points": checks, "legacy_reproduction_mismatches": legacy_mismatches}


def main():
    with (ROOT / "data/zurich_wide.pkl").open("rb") as handle:
        graph = pickle.load(handle)
    result = {"scope": "descriptive paired generated-world results; no population significance claim",
              "known_fact_check": "selected states differ by policy; this is physical consistency, not a common-state calibration comparison",
              "primary": readout(HERE / "matched_results.json", graph),
              "independent_generated": readout(HERE / "independent_results.json", graph)}
    (HERE / "readout.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for panel in ("primary", "independent_generated"):
        print(panel, result[panel]["journeys"], result[panel]["known_fact_probability_checks"])


if __name__ == "__main__":
    main()
