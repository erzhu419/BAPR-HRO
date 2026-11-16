"""Strict pre-outcome metadata prescreen, 5-day date/menu freeze, and source decoding."""

import argparse
import csv
from datetime import date, datetime, timedelta, timezone
import gzip
from io import BytesIO
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
from urllib.request import Request, urlopen
from zipfile import ZipFile
import zlib

from experiments.decision_calibration_v2.prepare_static import (
    dated_connections_from_static,
    extract_static,
)
from experiments.decision_calibration_v2.run_calibration import select_public
from experiments.decision_gap_v1.candidate_coverage import endpoint_coverage, expand_menu
from experiments.decision_gap_v1.panel import read, write
from experiments.decision_gap_v1.prepare import observations, trip_set
from experiments.decision_gap_v1.transfer_events import register_opportunities
from experiments.natural_gap_v2 import source_actuals
from experiments.reconstruction_data_audit.build_dated_replay import attach_outcomes
from experiments.reconstruction_v33.nodes import root_ride
from experiments.reconstruction_v34.discover import context, static_catalog
from experiments.reconstruction_v37.controller import node_structure
from experiments.reconstruction_v38.run import restore_node
from experiments.reconstruction_v50.execute import ObservationContract
from experiments.reconstruction_v50.menu import wide_root
from experiments.training_identity_v1.public_catalog import build_public_catalog
from experiments.validation_identity_v1.static_acquire import download_archive, save_gzip
from time import perf_counter, process_time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CANDIDATE_WINDOW_DAYS = 7  # 2023-04-06 .. 2023-04-12 (all under 2023-04-05 static release)


def fetch_static_zip_metadata(version):
    url = (
        "https://archive.opentransportdata.swiss/timetable_gtfs/"
        f"timetable-2023-gtfs2020/GTFS_FP2023_{version}_04-15.zip"
    )
    downloaded = 0

    def fetch_bytes(start, length):
        nonlocal downloaded
        rng = f"bytes={start}-{start + length - 1}" if start >= 0 else f"bytes=-{length}"
        with urlopen(Request(url, headers={"Range": rng}), timeout=60) as resp:
            if resp.status != 206:
                raise ValueError(f"Expected 206 Range response, got {resp.status}")
            raw = resp.read(length)
            downloaded += len(raw)
            total = int(resp.headers["Content-Range"].rsplit("/", 1)[1])
            return raw, total

    tail, size = fetch_bytes(-1, 65557)
    with ZipFile(BytesIO(tail)) as archive:
        member = archive.getinfo("feed_info.txt")
        header, _ = fetch_bytes(member.header_offset + size - len(tail), 30)
        fields = struct.unpack("<4s5H3L2H", header)
        raw, _ = fetch_bytes(
            member.header_offset + size - len(tail) + 30 + sum(fields[-2:]),
            member.compress_size,
        )
        text = zlib.decompress(raw, -15).decode("utf-8-sig")
    return dict(
        static_release=version,
        url=url,
        archive_bytes=size,
        feed_info=list(csv.DictReader(text.splitlines())),
        downloaded_header_bytes=downloaded,
        checked_at_utc=datetime.now(timezone.utc).isoformat(),
        actual_or_realtime_values_opened=False,
    )


def prescreen(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    protocol = read(directory / "protocol.json")
    ng2 = ROOT / "experiments/natural_gap_v2"

    # 1. Build 72-date combined exposure inventory
    prior_58 = read(ng2 / "exposure_inventory.json")["known_exposed_service_dates"]
    ng2_14 = read(ng2 / "exposure_record.json")["dates"]
    exposed = sorted(set(prior_58) | set(ng2_14))
    if len(exposed) != 72:
        raise ValueError(f"Expected 72 exposed dates, got {len(exposed)}")
    write(
        directory / "exposure_inventory.json",
        dict(
            recorded_at_utc=datetime.now(timezone.utc).isoformat(),
            known_exposed_service_dates=exposed,
            count=len(exposed),
            sources=protocol["frozen_principles"]["prior_exposure_sources"],
        ),
    )

    # 2. Copy frozen fits and RT capture rule
    shutil.copyfile(ng2 / "rt_metadata_rule.json", directory / "rt_metadata_rule.json")
    shutil.copyfile(ROOT / protocol["frozen_models"]["physical_fit"], directory / "physical_fit.json")
    shutil.copyfile(ROOT / protocol["frozen_models"]["emission_fit"], directory / "repaired_fit.json")
    shutil.copyfile(ng2 / "actual_metadata_2023-04.json", directory / "actual_metadata_2023-04.json")

    start_day = date.fromisoformat(protocol["date_selection"]["earliest_candidate_date"])
    candidate_dates = [
        (start_day + timedelta(days=i)).isoformat() for i in range(CANDIDATE_WINDOW_DAYS)
    ]
    if set(candidate_dates) & set(exposed):
        raise ValueError("Candidate window overlaps exposed dates")

    # 3. Verify static GTFS metadata for 2023-04-05 and acquire static schedule only
    static_meta_path = directory / "static_metadata_2023-04-05.json"
    if not static_meta_path.exists():
        write(static_meta_path, fetch_static_zip_metadata("2023-04-05"))
    static_meta = read(static_meta_path)

    static_manifest_path = directory / "static_source_manifest.json"
    if not static_manifest_path.exists():
        write(
            static_manifest_path,
            dict(
                frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                public_schedule_dates=candidate_dates,
                target_dates_frozen=False,
                source_version_by_date={d: "2023-04-05" for d in candidate_dates},
                versions={
                    "2023-04-05": {
                        k: static_meta[k] for k in ("url", "archive_bytes", "feed_info")
                    }
                },
                rule="Weekly Wednesday 04:15 GTFS release switched Wednesday 14:00; 2023-04-05 covers mornings 2023-04-06..2023-04-12.",
                actual_or_rt_values_read=False,
            ),
        )

    if not (directory / "static_acquisition.json").exists():
        manifest = read(static_manifest_path)
        meta = manifest["versions"]["2023-04-05"]
        archive = directory / "cache" / meta["url"].rsplit("/", 1)[1]
        transport = download_archive(meta["url"], meta["archive_bytes"], archive)
        runtime = directory / "inputs/static_2023-04-05"
        prep = extract_static(archive, runtime)
        if prep["feed_info"] != meta["feed_info"]:
            raise ValueError("Extracted static feed_info mismatch")
        stops, conns = dated_connections_from_static(candidate_dates, runtime)
        scheduled = dict(
            stops=stops,
            dates={
                d: dict(connections=rows, static_release="2023-04-05")
                for d, rows in conns.items()
            },
            source_version_by_date=manifest["source_version_by_date"],
            public_source_frozen_at_utc=manifest["frozen_at_utc"],
            actual_or_rt_values_read=False,
        )
        public_cat = build_public_catalog(runtime, candidate_dates)
        public_cat.update(static_release="2023-04-05", feed_info=meta["feed_info"])
        (directory / "source_catalogs").mkdir(parents=True, exist_ok=True)
        save_gzip(directory / "source_catalogs/2023-04-05.json.gz", public_cat)
        save_gzip(directory / "scheduled.json.gz", scheduled)
        save_gzip(
            directory / "active_public_endpoints.json.gz",
            dict(
                dates=public_cat["dates"],
                source_version_by_date=manifest["source_version_by_date"],
                public_source_frozen_at_utc=manifest["frozen_at_utc"],
                actual_or_rt_values_read=False,
            ),
        )
        write(
            directory / "static_acquisition.json",
            dict(
                completed_at_utc=datetime.now(timezone.utc).isoformat(),
                dates=sorted(scheduled["dates"]),
                transport=transport,
                scheduled_counts={d: len(r) for d, r in conns.items()},
                actual_or_rt_values_read=False,
            ),
        )

    # 4. Build actual_sources.json from April zip central directory headers only
    april_meta = read(directory / "actual_metadata_2023-04.json")
    actual_days = []
    for d in candidate_dates:
        matches = [m for m in april_meta["members"] if Path(m["name"]).name == f"{d}_istdaten.csv"]
        actual_days.append(
            dict(day=d, official_daily_member_count=len(matches), members=matches)
        )
    write(
        directory / "actual_sources.json",
        dict(
            checked_at_utc=datetime.now(timezone.utc).isoformat(),
            candidate_dates=candidate_dates,
            days=actual_days,
            target_values_read=False,
        ),
    )

    # 5. Fetch RT directory listing and prescreen capture clocks (tar headers only)
    rt_meta_path = directory / "rt_source_metadata.json"
    if not rt_meta_path.exists():
        with urlopen("https://mirror.traines.eu/swiss-gtfs-rt/", timeout=60) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        pattern = re.compile(
            r'<a href="((2023-[0-9]{2}-[0-9]{2})\.tar\.bz2)">.*?</a>\s+([0-9A-Za-z-]+ [0-9:]+)\s+([0-9]+)'
        )
        by_date = {}
        for fname, d_str, mod, sz in pattern.findall(html):
            by_date[d_str] = dict(
                date=d_str,
                filename=fname,
                url=f"https://mirror.traines.eu/swiss-gtfs-rt/{fname}",
                index_modified=mod,
                size_bytes=int(sz),
            )
        cand_rows = [by_date[d] for d in candidate_dates if d in by_date]
        write(
            rt_meta_path,
            dict(
                checked_at_utc=datetime.now(timezone.utc).isoformat(),
                excluded_exposed_dates=exposed,
                candidate_days=cand_rows,
                considered_dates=[],
                dates_frozen=False,
                protobuf_values_opened=False,
            ),
        )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.natural_gap_v2.rt_acquire",
            "--directory",
            str(directory),
            "--connections",
            "24",
        ],
        check=True,
    )
    source_actuals.cache(directory, candidate_dates)
    print("Prescreen complete (no protobuf or actual CSV decompressed).", flush=True)


def freeze_and_prepare(directory):
    directory = Path(directory)
    protocol = read(directory / "protocol.json")
    target_days = int(protocol["date_selection"]["target_days"])
    exposed = set(read(directory / "exposure_inventory.json")["known_exposed_service_dates"])
    rt = read(directory / "rt_source_metadata.json")
    static = read(directory / "static_source_manifest.json")
    actual = {row["day"]: row for row in read(directory / "actual_sources.json")["days"]}
    ready_static = set(read(directory / "static_acquisition.json")["dates"])

    if not (directory / "date_manifest.json").exists():
        considered, selected = [], []
        for src in sorted(rt["candidate_days"], key=lambda r: r["date"]):
            day = src["date"]
            reasons = []
            clocks = read(directory / "rt_clocks" / f"{day}.json")
            if day in exposed:
                reasons.append("already_exposed")
            if actual.get(day, {}).get("official_daily_member_count") != 1:
                reasons.append("official_actual_member_not_unique_or_absent")
            if day not in ready_static or day not in static["source_version_by_date"]:
                reasons.append("matching_public_static_unavailable")
            if not clocks.get("capture_coverage_complete", False):
                reasons.append("original_capture_window_incomplete")
            considered.append(
                dict(
                    day=day,
                    reasons=reasons,
                    captured_bins=clocks.get("matching_bins", []),
                    missing_bins=clocks.get("missing_bins", []),
                    target_values_opened=False,
                )
            )
            if not reasons:
                selected.append(
                    dict(
                        day=day,
                        static_version=static["source_version_by_date"][day],
                        rt=src,
                        actual_member=actual[day]["members"][0]["name"],
                        capture_metadata=f"rt_clocks/{day}.json",
                    )
                )
            if len(selected) == target_days:
                break
        write(
            directory / "date_consideration.json",
            dict(rows=considered, selected_count=len(selected), target_values_opened=False),
        )
        if len(selected) != target_days:
            raise ValueError(f"Expected {target_days} qualified dates, got {len(selected)}")
        write(
            directory / "date_manifest.json",
            dict(
                stage="bounded_gate_5d_v1",
                dates_frozen=True,
                frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                selected_dates=[r["day"] for r in selected],
                days=selected,
                target_values_opened_before_freeze=False,
                prior_exposure_checked=True,
                original_72_additional_budget=0,
            ),
        )
        print("Frozen 5 dates:", [r["day"] for r in selected], flush=True)

    manifest = read(directory / "date_manifest.json")
    dates = manifest["selected_dates"]
    if (directory / "selection.json").exists():
        print("Public menus already frozen.", flush=True)
        return

    stage_tick, stage_cpu = perf_counter(), process_time()
    v34_proto = read(ROOT / "experiments/reconstruction_v34/protocol.json")
    v34_proto["main_dates"] = dates
    source = read(directory / "scheduled.json.gz")
    data = dict(stops=source["stops"], dates={d: source["dates"][d] for d in dates})
    for d in dates:
        attach_outcomes(d, data["dates"][d]["connections"], [])
    write(directory / "scheduled_placeholder.json.gz", data)
    catalog = static_catalog(data, v34_proto)
    write(directory / "static_catalog.json.gz", catalog)
    selected = select_public(catalog, dates, v34_proto["structure"]["max_selected_per_day"])

    graphs, ledger, roots = {}, [], []
    for index, row in enumerate(selected["selected"]):
        day = row["day"]
        if day not in graphs:
            graphs[day] = context(data, day)[0]
        graph = graphs[day]
        tick, cpu = perf_counter(), process_time()
        root, caps = wide_root(graph, row, v34_proto["structure"])
        original = root_ride(graph, root)
        expanded, cov = expand_menu(graph, original, root["goal"])
        old_o, new_o = ObservationContract.from_menu(original), ObservationContract.from_menu(expanded)
        union = {stop: set(cids) for stop, cids in new_o.station_to_cids.items()}
        for stop, cids in old_o.station_to_cids.items():
            union.setdefault(stop, set()).update(cids)
        sight = ObservationContract(union)
        if trip_set(graph, original) != trip_set(graph, expanded):
            raise AssertionError("Prefix closure changed remote trip set")
        cost = dict(process_cpu=process_time() - cpu, wall=perf_counter() - tick)
        cid = next(c for c in original.path if original.exits.get(c) is not None)
        reg = register_opportunities(graph, original, cid, graph.connections[cid].arr_time, expanded)
        reg.update(root_index=index, day=day, clock_source="schedule placeholder")
        write(
            directory / "menus" / f"{index:03}.json.gz",
            dict(
                root_index=index,
                root=root,
                original_node=node_structure(original),
                expanded_node=node_structure(expanded),
                new_observation_contract=sight.describe(),
                cost=cost,
                ledger=cov,
                cap_ledger=caps,
                registration=reg,
                actual_values_used=False,
            ),
        )
        roots.append(dict(root_index=index, **{k: root[k] for k in ("root_id", "day", "start", "deadline")}))
        ledger.append(
            dict(
                root_index=index,
                day=day,
                opportunities=len(reg["rows"]),
                candidate_pairs=len(reg["candidate_pairs"]),
                no_opportunity=reg["no_opportunity"],
            )
        )
    stamp = datetime.now(timezone.utc).isoformat()
    selected.update(frozen_at_utc=stamp, registered=ledger, selection_uses_target_values=False)
    write(directory / "selection.json", selected)
    write(
        directory / "contract.json",
        dict(
            stage="bounded_gate_5d_v1",
            roots=len(roots),
            root_manifest=roots,
            control_dates=dates,
            arms=protocol["arms"],
            controller_hyperparameters=protocol["controller_hyperparameters"],
            clock_contracts=protocol["clock_contracts"],
            mct_contracts=protocol["mct_contracts"],
            frozen_at_utc=stamp,
            emission_snapshots={"repaired_fit": "repaired_fit.json"},
            physical_fit="physical_fit.json",
            original_72_additional_budget=0,
        ),
    )
    write(
        directory / "preparation_public.json",
        dict(
            process_cpu=process_time() - stage_cpu,
            wall=perf_counter() - stage_tick,
            roots=len(roots),
            target_values_read=False,
        ),
    )
    print(f"Frozen {len(roots)} public roots across {len(dates)} dates.", flush=True)


def attach_and_decode(directory, workers=5):
    directory = Path(directory)
    manifest = read(directory / "date_manifest.json")
    selection = read(directory / "selection.json")
    if not manifest["dates_frozen"] or not selection.get("frozen_at_utc"):
        raise ValueError("Freeze dates and menus before decoding actuals/RT")

    # 1. Decompress & filter ist-daten
    source_actuals.acquire(directory, manifest["selected_dates"])
    source_actuals.summarize(directory)

    # 2. Attach actual outcomes to scheduled connections
    if not (directory / "data.json.gz").exists():
        data = read(directory / "scheduled_placeholder.json.gz")
        summary, graphs, audits = {}, {}, []
        for day in data["dates"]:
            path = directory / "actual" / f"{day}.csv.gz"
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle, delimiter=";"))
            cov = attach_outcomes(day, data["dates"][day]["connections"], rows)
            summary[day] = dict(cov, actual_source_available=True, source=str(path))
            graphs[day] = context(data, day)
        for index, root in enumerate(selection["selected"]):
            menu = read(directory / "menus" / f"{index:03}.json.gz")
            graph, _, _, _, metadata, _ = graphs[root["day"]]
            audits.append(
                dict(
                    root_index=index,
                    day=root["day"],
                    **endpoint_coverage(
                        graph,
                        metadata,
                        restore_node(menu["expanded_node"]),
                        observations(menu["new_observation_contract"]),
                    ),
                )
            )
        write(directory / "data.json.gz", data)
        write(
            directory / "endpoint_coverage.json",
            dict(
                days=summary,
                roots=audits,
                selection_frozen_at_utc=selection["frozen_at_utc"],
                missing_roots_dropped=False,
            ),
        )

    # 3. Decode GTFS-RT messages with platform_stop_updates preserved
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.natural_gap_v2.rt_decode",
            "--directory",
            str(directory),
            "--workers",
            str(workers),
        ],
        check=True,
    )
    print("Attach and RT decode complete.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prescreen", "freeze", "decode"))
    parser.add_argument("--directory", type=Path, default=HERE)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    if args.stage == "prescreen":
        prescreen(args.directory)
    elif args.stage == "freeze":
        freeze_and_prepare(args.directory)
    else:
        attach_and_decode(args.directory, workers=args.workers)
