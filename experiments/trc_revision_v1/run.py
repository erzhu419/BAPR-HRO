"""Development replay of all 60 exposed BG5 roots under the corrected executor."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from os import getpid
from pathlib import Path
from time import perf_counter, process_time

from experiments.bounded_gate_5d_v1.pair_contract import (
    DayPlatformIndex,
    PairMatchedPlanner,
    PairRecursivePlanner,
    decorate_menu_with_mct,
)
from experiments.trc_revision_v1.executor import BoundedArmExecutor
from experiments.decision_gap_v1.panel import read, write
from experiments.decision_gap_v1.prepare import observations
from experiments.reconstruction_v34.discover import context
from experiments.reconstruction_v34.execute import ArchivedWorld
from experiments.reconstruction_v34.run import relevant_frames, setup
from experiments.reconstruction_v38.run import ordered, restore_node, save
from experiments.reconstruction_v41.physical_posterior import PhysicalHistory
from experiments.reconstruction_v50.run import actions

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESIDENT = None


def check_inputs(directory, source):
    directory,source = Path(directory),Path(source)
    contract,protocol = read(directory/'contract.json'),read(directory/'protocol.json')
    original,manifest = read(source/'contract.json'),read(source/'date_manifest.json')
    selection = read(source/'selection.json')['selected']
    if protocol['status']!='frozen_for_development_replay':
        raise ValueError('Freeze the development protocol before running')
    if contract['arms']!=protocol['arms'] or contract['roots']!=60 or len(selection)!=60:
        raise ValueError('The declared root/arm population changed')
    labels = {arm['label'] for arm in contract['arms']}
    if any(a not in labels or b not in labels for a,b in protocol['comparison']['causal_contrasts']):
        raise ValueError('A requested contrast has no registered controller')
    if contract['root_manifest']!=original['root_manifest'] or contract['control_dates']!=manifest['selected_dates']:
        raise ValueError('The original exposed root panel changed')
    if contract['controller_hyperparameters']!=original['controller_hyperparameters']:
        raise ValueError('Original world and CPU budgets changed')
    if contract['original_72_additional_budget']!=0:
        raise ValueError('Original72 budget must remain zero')
    return contract


def initialize(directory, source):
    global RESIDENT
    directory,source = Path(directory),Path(source)
    tick, cpu = perf_counter(), process_time()
    contract = read(directory / "contract.json")
    manifest = read(source / "date_manifest.json")
    static_by_day = {row["day"]: row["static_version"] for row in manifest["days"]}
    messages, available = {}, {}
    for day in contract["control_dates"]:
        path = source / "messages" / f"{day}.json.gz"
        src = read(path)
        messages[day] = src
        summ = src["summary"]
        available[day] = (
            summ.get("source_parse_complete", False)
            and summ.get("identity_mapping_complete", False)
            and summ.get("sampled_snapshots", 0) > 0
        )
    RESIDENT = dict(
        directory=directory,
        source=source,
        contract=contract,
        static_by_day=static_by_day,
        data=read(source / "data.json.gz"),
        messages={"dates": messages},
        fit=read(source / "physical_fit.json"),
        emission=read(source / "repaired_fit.json"),
        coverage={r["root_index"]: r for r in read(source / "endpoint_coverage.json")["roots"]},
        source_available=available,
        graphs={},
        platform_indexes={},
    )
    RESIDENT["load_cost"] = dict(process_cpu=process_time() - cpu, wall=perf_counter() - tick)
    write(directory/'worker_preparation'/f'{getpid()}_load.json',dict(
        process_id=getpid(),kind='source_load',cost=RESIDENT['load_cost']))


def _day_resources(day):
    resident = RESIDENT
    source = resident["source"]
    preparing = day not in resident['graphs'] or day not in resident['platform_indexes']
    tick,cpu = perf_counter(),process_time()
    if day not in resident["graphs"]:
        resident["graphs"][day] = context(resident["data"], day)
    if day not in resident["platform_indexes"]:
        graph, _, stop_ids, _, metadata, _ = resident["graphs"][day]
        stop_mcts = {
            str(bpuic): int(graph.stops[idx].min_transfer_time)
            for bpuic, idx in stop_ids.items()
        }
        version = resident["static_by_day"][day]
        static_root = source / "inputs" / f"static_{version}" / "data"
        resident["platform_indexes"][day] = DayPlatformIndex(
            metadata,
            regional_dir=static_root / "zurich_gtfs",
            static_dir=static_root / "swiss_gtfs",
            stop_mcts=stop_mcts,
        )
    if preparing:
        write(resident['directory']/'worker_preparation'/f'{getpid()}_{day}.json',dict(
            process_id=getpid(),kind='shared_day_graph_and_platform_index',day=day,
            cost=dict(process_cpu=process_time()-cpu,wall=perf_counter()-tick)))
    return resident["graphs"][day], resident["platform_indexes"][day]


def execute_single_arm(index, arm_cfg, menu, graph_ctx, plat_index, resident, *, tau_same=None):
    graph, _, _, mapping, metadata, _ = graph_ctx
    root, contract = menu["root"], resident["contract"]
    hyper = contract["controller_hyperparameters"]
    tau_val = (
        float(contract["mct_contracts"]["platform_pair"]["tau_same_platform_default_seconds"])
        if tau_same is None
        else float(tau_same)
    )

    tick, cpu = perf_counter(), process_time()
    model, parameters = setup(graph, resident["fit"])
    raw_node = restore_node(menu["expanded_node"])
    node = decorate_menu_with_mct(
        raw_node,
        graph,
        mct_contract=arm_cfg["mct_contract"],
        platform_index=plat_index if arm_cfg["mct_contract"] == "platform_pair" else None,
        tau_same_platform=tau_val,
    )
    sight = observations(menu["new_observation_contract"])
    frames = relevant_frames(
        resident["messages"]["dates"][root["day"]]["frames"], node.trips(model), mapping
    )
    history = PhysicalHistory(
        model, parameters, fit=resident["emission"], frames=frames, source_to_local=mapping
    )
    if arm_cfg["controller"] in ("recursive", "bounded_cert"):
        max_w = hyper[arm_cfg["controller"]]["exact_combined_max_width"]
        planner = PairRecursivePlanner(
            graph, model, parameters, root["deadline"], exact_combined_max_width=max_w
        )
    else:
        planner = PairMatchedPlanner(graph, model, parameters, root["deadline"])

    executor = BoundedArmExecutor(
        graph,
        planner,
        root["deadline"],
        arm_config=arm_cfg,
        hyperparams=hyper,
        root_index=index,
        cpu_start=cpu,
        observation_contract=sight,
        clock_rules=contract["clock_contracts"]["async"],
    )
    outcome = executor.run(node, root["start"], history, ArchivedWorld(metadata))
    cost = dict(
        process_cpu=process_time() - cpu,
        wall=perf_counter() - tick,
        shared_menu=menu["cost"],
    )
    cost["process_cpu_including_menu"] = cost["process_cpu"] + menu["cost"]["process_cpu"]
    cost["wall_including_menu"] = cost["wall"] + menu["cost"]["wall"]

    return dict(
        root_index=index,
        root_id=root["root_id"],
        day=root["day"],
        origin=root["origin_name"],
        goal=root["goal_name"],
        label=arm_cfg["label"],
        arm=arm_cfg,
        tau_same_platform_seconds=tau_val if arm_cfg["mct_contract"] == "platform_pair" else None,
        loss_cap=root["deadline"] - root["start"],
        executed=True,
        development_only=True,
        source_stage="bounded_gate_5d_v1",
        outcome=outcome,
        actions=actions(outcome),
        states=executor.states,
        gates=executor.gates,
        publications=executor.publications,
        decision_costs=executor.decision_costs,
        control_stats=executor.control_stats,
        planner_stats=planner.stats,
        cost=cost,
        coverage=dict(source_available=True, full_menu=resident["coverage"][index]),
    )


def run_root(index):
    resident = RESIDENT
    directory = resident["directory"]
    contract = resident["contract"]
    menu = read(resident["source"] / "menus" / f"{index:03}.json.gz")
    day = menu["root"]["day"]
    if not resident["source_available"][day]:
        raise ValueError(f"Frozen message source incomplete: {day}; retain this failure without date substitution")
    graph_ctx, plat_index = _day_resources(day)
    arms_by_label = {a["label"]: a for a in contract["arms"]}
    labels = tuple(a["label"] for a in contract["arms"])

    completed = []
    for label in ordered(labels, index):
        out_path = directory / "roots" / f"{index:03}_{label}.json.gz"
        if out_path.exists():
            continue
        print(f'start root={index:03} arm={label}',flush=True)
        row = execute_single_arm(
            index, arms_by_label[label], menu, graph_ctx, plat_index, resident
        )
        save(out_path, row)
        completed.append(
            dict(
                label=label,
                status=row["outcome"]["status"],
                loss=row["outcome"]["loss"],
                cpu=round(row["cost"]["process_cpu_including_menu"], 4),
                gates=len(row["gates"]),
            )
        )
        print(
            f"root={index:03} arm={label} status={row['outcome']['status']} "
            f"loss={row['outcome']['loss']} cpu={row['cost']['process_cpu_including_menu']:.3f}s",
            flush=True,
        )

    # Lightweight tau_same sensitivity [60, 90, 120] for analytical platform_pair controllers
    sens_path = directory / "sensitivity_roots" / f"{index:03}.json"
    if not sens_path.exists():
        sens_rows = {}
        taus = contract["mct_contracts"]["platform_pair"]["tau_same_platform_sensitivity_seconds"]
        for base_label in ("cheap_plat_async", "recursive_plat_async"):
            sens_rows[base_label] = {}
            for tau in taus:
                if float(tau) == 90.0:
                    saved = read(directory / "roots" / f"{index:03}_{base_label}.json.gz")
                    sens_rows[base_label][str(int(tau))] = dict(
                        loss=saved["outcome"]["loss"],
                        status=saved["outcome"]["status"],
                        cpu=saved["cost"]["process_cpu_including_menu"],
                    )
                else:
                    print(f'start root={index:03} arm={base_label} tau={tau}',flush=True)
                    res = execute_single_arm(
                        index,
                        arms_by_label[base_label],
                        menu,
                        graph_ctx,
                        plat_index,
                        resident,
                        tau_same=float(tau),
                    )
                    sens_rows[base_label][str(int(tau))] = dict(
                        loss=res["outcome"]["loss"],
                        status=res["outcome"]["status"],
                        cpu=res["cost"]["process_cpu_including_menu"],
                    )
        write(sens_path, dict(root_index=index, day=day, loss_cap=menu["root"]["deadline"] - menu["root"]["start"], sensitivity=sens_rows))

    write(
        directory / "preparation_roots" / f"{index:03}.json",
        dict(root_index=index, day=day, completed=completed),
    )
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=HERE)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--source-directory",type=Path,required=True)
    args = parser.parse_args()

    contract = check_inputs(args.directory,args.source_directory)
    labels = [a["label"] for a in contract["arms"]]
    pending = [
        i
        for i in range(contract["roots"])
        if any(not (args.directory / "roots" / f"{i:03}_{lbl}.json.gz").exists() for lbl in labels)
        or not (args.directory / "sensitivity_roots" / f"{i:03}.json").exists()
    ]
    tick = perf_counter()
    if pending:
        with ProcessPoolExecutor(
            max_workers=args.workers, initializer=initialize, initargs=(args.directory,args.source_directory)
        ) as pool:
            for fut in as_completed([pool.submit(run_root, i) for i in pending]):
                print("completed root", fut.result(), flush=True)
    write(
        args.directory / "replay_preparation.json",
        dict(
            workers=args.workers,
            wall=perf_counter() - tick,
            requested_roots=contract["roots"],
            executed_roots=pending,
            original_72_additional_budget=0,
            independent_confirmation=False,
            source_directory=str(args.source_directory),
        ),
    )


if __name__ == "__main__":
    main()
