"""Cross refresh/retain with first-failure cap/rescue on frozen generated worlds."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from statistics import mean, median
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.continuation_v1.run_matched import ARCHIVE, VERSION as V1_VERSION, run_one
from src.continuation_router import ContinuationRouter

VERSION = "continuation_v2_branch_complete"
PANELS = {
    "primary": (ARCHIVE, HERE / "matched_results_methods"),
    "independent_generated": (HERE / "independent_worlds", HERE / "independent_results_methods"),
}
CELLS = {
    f"{mode}_{score}": {"candidate_mode": mode, "conditioning": True, "rescue": score == "rescue"}
    for mode in ("refresh", "retain") for score in ("cap", "rescue")
}


def archived_references(directory, source, index, spec):
    references = {}
    for variant in ("legacy", "conditioned_cap", "conditioned_rescue"):
        path = directory / f"case_{index:03d}_{variant}.json"
        if not path.exists():
            continue
        row = json.loads(path.read_text())
        recorded = ROOT / row["archive_case"]
        if (recorded.resolve() != source.resolve() or row["spec"] != spec
                or row["variant"] != variant or row["implementation"] != V1_VERSION):
            raise ValueError(f"Reference receipt does not match this case: {path}")
        references[variant] = {"loss": row["loss"], "samples": row["samples"],
                               "receipt": str(path.relative_to(ROOT))}
    return references


def paired(deltas):
    return {"n": len(deltas), "mean_difference": mean(deltas) if deltas else None,
            "better": sum(d < 0 for d in deltas), "worse": sum(d > 0 for d in deltas)}


def summarize_group(rows):
    cells = {}
    by_case = {}
    for cell in CELLS:
        attempted = [row for row in rows if row["variant"] == cell]
        complete = [row for row in attempted if row["status"] == "complete"]
        for row in complete:
            by_case.setdefault(row["index"], {})[cell] = row
        reference_variant = "conditioned_rescue" if CELLS[cell]["rescue"] else "conditioned_cap"
        differences = {}
        for label, reference in (("legacy", "legacy"), ("v1_same_score", reference_variant)):
            differences[label] = paired([row["loss"] - row["references"][reference]["loss"]
                                         for row in complete if reference in row["references"]])
        cells[cell] = {
            "attempted": len(attempted), "complete": len(complete),
            "execution_errors": len(attempted) - len(complete),
            "arrival_rate": mean(row["arrived"] for row in complete) if complete else None,
            "mean_capped_loss": mean(row["loss"] for row in complete) if complete else None,
            "mean_total_ms": mean(row["total_ms"] for row in complete) if complete else None,
            "median_total_ms": median(row["total_ms"] for row in complete) if complete else None,
            "mean_rollout_ms": mean(row["rollout_ms"] for row in complete) if complete else None,
            "paired_minus_archived": differences,
        }
    full = [case for case in by_case.values() if case.keys() == CELLS.keys()]
    losses = [{cell: row["loss"] for cell, row in case.items()} for case in full]
    effects = {
        "retain_minus_refresh_cap": paired([r["retain_cap"] - r["refresh_cap"] for r in losses]),
        "retain_minus_refresh_rescue": paired([r["retain_rescue"] - r["refresh_rescue"] for r in losses]),
        "rescue_minus_cap_refresh": paired([r["refresh_rescue"] - r["refresh_cap"] for r in losses]),
        "rescue_minus_cap_retain": paired([r["retain_rescue"] - r["retain_cap"] for r in losses]),
        "interaction": paired([(r["retain_rescue"] - r["retain_cap"])
                               - (r["refresh_rescue"] - r["refresh_cap"]) for r in losses]),
    }
    candidate_pairs = {}
    for score in ("cap", "rescue"):
        pairs = [(case[f"retain_{score}"], case[f"refresh_{score}"]) for case in full]
        candidate_pairs[score] = {field: sum(
            [(d["stop"], d["time"], d[field]) for d in retain["decisions"]]
            != [(d["stop"], d["time"], d[field]) for d in refresh["decisions"]]
            for retain, refresh in pairs) for field in ("candidates", "selected")}
    return {"cells": cells, "complete_od_cases_for_factorial_effects": len(full), "effects": effects,
            "retain_refresh_pairs_with_different_sequences": candidate_pairs}


def summarize(rows):
    result = {}
    for panel in sorted({row["panel"] for row in rows}):
        group = [row for row in rows if row["panel"] == panel]
        result[panel] = {"all": summarize_group(group), "by_stress": {}, "by_date": {}}
        for stress in (False, True):
            part = [row for row in group if row["spec"][1] == stress]
            if part:
                result[panel]["by_stress"]["stress" if stress else "no_cancel"] = summarize_group(part)
        for date in sorted({row["spec"][0] for row in group}):
            part = [row for row in group if row["spec"][0] == date]
            result[panel]["by_date"][date] = summarize_group(part)
    return result


def write_json(path, data, indent=None):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=indent) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", nargs="+", choices=tuple(PANELS), default=list(PANELS))
    parser.add_argument("--primary-indices", nargs="+", type=int, default=list(range(24)))
    parser.add_argument("--independent-indices", nargs="+", type=int, default=list(range(6)))
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=HERE / "candidate_cross_results.json")
    args = parser.parse_args()
    if args.samples < 1 or args.workers < 1:
        parser.error("samples and workers must be positive")
    if args.output.exists() and json.loads(args.output.read_text()).get("implementation") != VERSION:
        parser.error("use a new output path for the candidate cross")

    indices = {"primary": args.primary_indices, "independent_generated": args.independent_indices}
    directory = args.output.with_name(args.output.stem + "_cases")
    rows, pending = [], []
    for panel in dict.fromkeys(args.panels):
        archive, reference_directory = PANELS[panel]
        panel_directory = directory / panel
        panel_directory.mkdir(parents=True, exist_ok=True)
        for index in dict.fromkeys(indices[panel]):
            source = (archive / f"case_{index:03d}.json").resolve()
            spec = json.loads(source.read_text())["spec"]
            references = archived_references(reference_directory, source, index, spec)
            for cell, options in CELLS.items():
                path = panel_directory / f"case_{index:03d}_{cell}.json"
                metadata = {"implementation": VERSION, "panel": panel, "index": index,
                            "variant": cell, "samples": args.samples, "spec": spec,
                            "archive_case": str(source), "router_options": options,
                            "references": references}
                if path.exists():
                    row = json.loads(path.read_text())
                    if any(row.get(key) != value for key, value in metadata.items()):
                        raise ValueError(f"Use a new output path for a changed case: {path}")
                    rows.append(row | {"receipt": str(path)})
                else:
                    pending.append((archive, path, metadata))
    total = len(rows) + len(pending)
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(run_one, metadata["index"], metadata["variant"], args.samples, archive,
                            router_cls=ContinuationRouter, router_options=metadata["router_options"],
                            implementation=VERSION): (path, metadata)
                for archive, path, metadata in pending
            }
            for future in as_completed(futures):
                path, metadata = futures[future]
                try:
                    row = future.result() | metadata | {"status": "complete"}
                except Exception as error:
                    row = metadata | {"status": "error", "error": f"{type(error).__name__}: {error}"}
                write_json(path, row)
                rows.append(row | {"receipt": str(path)})
                print(json.dumps({"completed": len(rows), "total": total, "panel": row["panel"],
                                  "case": row["index"], "cell": row["variant"],
                                  "status": row["status"], "loss": row.get("loss"),
                                  "error": row.get("error")}), flush=True)
    rows.sort(key=lambda row: (row["panel"], row["index"], row["variant"]))
    detail_fields = {"predictions", "polls", "decisions", "messages", "events", "query_ms"}
    errors = sum(row["status"] == "error" for row in rows)
    write_json(args.output, {"implementation": VERSION, "scope": "paired generated-world screening",
                             "config": {"panels": args.panels, "indices": indices, "samples": args.samples,
                                        "workers": args.workers, "cells": CELLS, "tuning": "none"},
                             "execution_errors": errors, "summary": summarize(rows),
                             "cases": [{k: v for k, v in row.items() if k not in detail_fields}
                                       for row in rows]}, indent=2)
    print(json.dumps({"output": str(args.output), "cases": len(rows), "execution_errors": errors}))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
