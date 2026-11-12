"""Audit frozen V13 platform-policy receipts without resampling or rerunning them."""
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean


HERE = Path(__file__).resolve().parent
EXP = HERE.parent
METHODS = ("primary", "fixed_timeout", "public_failure", "first_visible")
COMPARISONS = (("fixed_timeout", "primary"), ("public_failure", "fixed_timeout"),
               ("first_visible", "public_failure"), ("first_visible", "fixed_timeout"))


def read(path):
    return json.loads(path.read_text())


def identity(row):
    return row["date"], row["query"], row["decision"]


def ride(row):
    if "cid" in row:
        return row
    return dict(cid=row["ridden_connection"], departure=row["time"], arrival=row["next_time"],
                origin=row["stop"], destination=row["next_stop"])


def summarize_arm(row, goal, deadline, main, rescue, prefix=()):
    assert row["published"] and not row["rejected"]
    assert row["status"] in ("arrived", "timeout")
    assert row["publish_at"] == row["root_clock"] + 1
    route = [ride(r) for r in row["rides"]]
    if row["status"] == "arrived":
        finish = route[-1]["arrival"]
        if "destination" in route[-1]:
            assert route[-1]["destination"] == goal
    else:
        finish = deadline
    loss = finish - row["root_clock"]
    assert row["loss_lower"] == row["loss_upper"] == loss
    assert all(a["arrival"] <= b["departure"] for a, b in zip(route, route[1:]))
    assert all(r["departure"] <= r["arrival"] <= deadline for r in route)
    events = [dict(reason=reason, target_cid=target, clock=clock,
                   next_cid=step["target"][0] if step["target"] else None,
                   next_ready=step["ready"], next_expiry=step["expiry"],
                   original_rescue_window=step["reference_window"])
              for step in row["trigger_trace"] for reason, target, clock in step["events"]]
    origin_class = "no_ride"
    if tuple(r["cid"] for r in route[:len(prefix)]) != tuple(prefix):
        origin_class = "different_prefix"
    elif len(route) > len(prefix):
        origin_class = {main: "A", rescue: "B"}.get(route[len(prefix)]["cid"], "other_connection")
    return dict(status=row["status"], loss=loss, first_ride=origin_class, route=route, triggers=events)


def compare(worlds, candidate, reference):
    gain = [w["arms"][reference]["loss"] - w["arms"][candidate]["loss"] for w in worlds]
    path_differs = [w["arms"][candidate]["route"] != w["arms"][reference]["route"] for w in worlds]
    reasons = Counter()
    changed = []
    for world, delta, path in zip(worlds, gain, path_differs, strict=True):
        if delta or path:
            candidate_arm = world["arms"][candidate]
            relevant = candidate_arm["triggers"]
            tag = relevant[0]["reason"] if relevant else "no_candidate_trigger"
            if delta > 0:
                reasons[tag] += 1
            changed.append(dict(world=world["world"], improvement=delta, route_differs=path,
                                first_ride=candidate_arm["first_ride"], first_trigger=tag))
    return dict(mean_improvement_seconds=mean(gain), loss_improved=sum(x > 0 for x in gain),
                loss_worse=sum(x < 0 for x in gain), loss_tied=sum(x == 0 for x in gain),
                route_differs=sum(path_differs), first_trigger_in_improved=sorted(reasons.items()),
                disagreements=changed)


def summarize(worlds):
    arms = {}
    for method in METHODS:
        rows = [w["arms"][method] for w in worlds]
        arms[method] = dict(mean_loss=mean(r["loss"] for r in rows),
                            arrived=sum(r["status"] == "arrived" for r in rows),
                            first_ride=dict(Counter(r["first_ride"] for r in rows)))
    return dict(n_worlds=len(worlds), arms=arms,
                comparisons={f"{a}_vs_{b}": compare(worlds, a, b) for a, b in COMPARISONS})


def main():
    platform = read(EXP / "reconstruction_v13/platform_results.json")
    facts = read(EXP / "reconstruction_v13/factored_results.json")
    inventory = {identity(r): r for r in read(EXP / "reconstruction_v12/transfer_inventory.json")["rows"]}
    plans = {identity(r): r["fixed_primary_tail_rescue"]["plan"]
             for r in read(EXP / "reconstruction_v10/tail_rescue_results.json")["rows"]}
    setups = {identity(r): r for r in facts["setups"]}
    frozen = defaultdict(dict)
    for row in platform["rows"]:
        key = identity(row), row["panel"], row["source"], row["world"]
        assert row["method"] not in frozen[key]
        frozen[key][row["method"]] = row
    roots, all_posterior = [], []
    for case in sorted(inventory):
        record, plan = inventory[case], plans[case]
        prefix = record["next_transfer_boundary"]["prefix"]
        assert plan["primary"][:len(prefix)] == prefix
        main, rescue = plan["primary"][len(prefix)], plan["rescue"][0]
        deadline, goal = record["root"]["deadline"], record["root"]["destination"]
        metadata = record["connections"]
        root = dict(date=case[0], query=case[1], decision=case[2],
                    A=metadata.get(str(main), {"cid": main}), B=metadata.get(str(rescue), {"cid": rescue}),
                    model_trip_A=setups[case]["hidden_trips"][0], model_trip_B=setups[case]["hidden_trips"][1],
                    deadline=deadline)
        for panel, source, numbers in (("waiting", "posterior", range(64)), ("waiting", "actual", (-1,)),
                                        ("onboard_bridge", "actual", (-1,))):
            worlds = []
            for number in numbers:
                rows = frozen.pop((case, panel, source, number))
                assert set(rows) == set(METHODS)
                clocks = {r["root_clock"] for r in rows.values()}
                assert len(clocks) == 1
                world = dict(world=number, root_clock=clocks.pop(), date=case[0], decision=case[2],
                             arms={m: summarize_arm(rows[m], goal, deadline, main, rescue,
                                                   prefix if panel == "onboard_bridge" else ()) for m in METHODS})
                worlds.append(world)
            root[f"{panel}_{source}"] = dict(summary=summarize(worlds), worlds=worlds)
            if source == "posterior":
                assert not any(w["arms"][m]["first_ride"] == "other_connection" for w in worlds for m in METHODS)
                all_posterior.extend(worlds)
        roots.append(root)
    assert not frozen and len(all_posterior) == 192

    factored = defaultdict(list)
    for row in facts["rows"]:
        if row["arm"] != "product":
            continue
        assert row["status"] == "complete"
        assert mean(row["policies"]["first_visible"]["losses"]) == row["policies"]["first_visible"]["mean_loss"]
        factored[identity(row)].append(dict(bank=row["bank"], relaxed=row["value"],
            clairvoyant=row["clairvoyant_root_mean"], first_visible=row["policies"]["first_visible"]["mean_loss"],
            first_visible_minus_relaxed=row["policies"]["first_visible"]["mean_loss"] - row["value"]))
    result = dict(source_commit="28d97a37988ca014619cc76eefe28d18185caa2e",
        scope="Receipt audit of the 192-world V13 platform panel and six actual bridge/waiting inputs. "
              "The separate 24-bank quantities use bank-matched policies and relaxation only.",
        receipt_verification="Loss recomputed from final ride time or declared timeout deadline. "
                             "Route disagreement compares the recorded connection and clock sequences.",
        restrictions="A/B label the issued main and rescue connection at the waiting root. "
                     "No other first connection was taken in the posterior panel. These receipts "
                     "cannot say what an expanded causal candidate rule would do.",
        posterior_summary=summarize(all_posterior), roots=roots,
        bank_gap_receipts=[dict(date=k[0], query=k[1], decision=k[2], banks=v) for k, v in sorted(factored.items())])
    (HERE / "trace_results.json").write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
    print("Audited", len(all_posterior), "posterior and", 2 * len(roots), "actual input groups.")


if __name__ == "__main__":
    main()
