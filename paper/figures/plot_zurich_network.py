"""Plot the Zürich GTFS sub-network used in BAPR-HRO experiments,
overlaid on a real OpenStreetMap basemap (via contextily).

Mirrors offline-sumo's network_topology.py: each line gets a distinct
colour, disrupted-day routes are highlighted in bold red, key OD
endpoints are starred. The basemap is a real city map (CC-BY OSM).

Output: paper/fig_zurich_network.pdf and .png
"""

import os, sys, pickle
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import contextily as cx
import geopandas as gpd
from shapely.geometry import Point, LineString

ROOT = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(ROOT)
OUT = os.path.join(PAPER_DIR, "fig_zurich_network")

# --- Routes affected on Oct 29 disrupted day (100% cancel on these) ---------
DISRUPTED_ROUTES = {"2", "3", "4", "5", "6", "23"}
# Route 7 also had ~80% cancel + +48min mean (the focused-OD example)
HEAVILY_AFFECTED = {"7"}

# --- Tram lines we want to colour individually ----------------------------
TRAM_LINES = ["2", "3", "4", "5", "6", "7", "8", "9", "10",
              "11", "13", "14", "17", "23"]
TRAM_COLORS = {
    "2": "#d62728",   # disrupted (red family)
    "3": "#e7969c",
    "4": "#bb6a6c",
    "5": "#a55194",
    "6": "#ce6dbd",
    "7": "#ff7f0e",   # heavily-affected (orange)
    "8": "#2ca02c",
    "9": "#98df8a",
    "10": "#8c564b",
    "11": "#1f77b4",
    "13": "#aec7e8",
    "14": "#9467bd",
    "17": "#c5b0d5",
    "23": "#e377c2",   # disrupted
}

# --- Highlight ODs from the multi-OD experiment ----------------------------
# Source for all 17 viable ODs: Paradeplatz
ORIGIN_ID = 895684956  # Paradeplatz
# A few notable destinations
KEY_DESTS = {
    201257157: ("Sihlpost/HB",       "*", 280),  # focused-OD destination
    67001060:  ("Sternen Oerlikon",  "*", 280),  # focused-OD origin
}


def main():
    print("Loading graph...")
    with open(os.path.join(PAPER_DIR, "..", "data", "zurich_wide.pkl"), "rb") as f:
        g = pickle.load(f)
    print(f"  {len(g.stops)} stops, {len(g.connections)} connections, "
          f"{len(set(c.route for c in g.connections))} routes")

    # Build per-route undirected edge set (unique stop pairs)
    route_edges = defaultdict(set)
    for c in g.connections:
        a, b = c.dep_stop, c.arr_stop
        if a != b:
            route_edges[c.route].add((min(a, b), max(a, b)))

    # Reproject stops to Web Mercator for contextily
    stop_pts = {sid: Point(s.lon, s.lat) for sid, s in g.stops.items()}
    gdf_stops = gpd.GeoDataFrame(
        {"id": list(stop_pts.keys()), "name": [g.stops[i].name for i in stop_pts]},
        geometry=list(stop_pts.values()),
        crs="EPSG:4326",
    ).to_crs(epsg=3857)
    proj_xy = {row["id"]: (row.geometry.x, row.geometry.y)
               for _, row in gdf_stops.iterrows()}

    # Build per-route LineStrings
    route_lines = {}
    for route, edges in route_edges.items():
        segs = []
        for a, b in edges:
            if a not in proj_xy or b not in proj_xy:
                continue
            segs.append(LineString([proj_xy[a], proj_xy[b]]))
        if segs:
            route_lines[route] = segs

    # ------------------- Plot ----------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 9.0))

    # Background: all non-tram routes (S-Bahn, IR, IC) in light grey
    bg_routes = [r for r in route_lines if r not in TRAM_LINES]
    for r in bg_routes:
        for seg in route_lines[r]:
            xs, ys = seg.xy
            ax.plot(xs, ys, color="#cccccc", lw=0.6, zorder=1, alpha=0.8)

    # Tram lines: all colored, disrupted ones thick
    for r in TRAM_LINES:
        if r not in route_lines:
            continue
        col = TRAM_COLORS[r]
        is_disrupted = r in DISRUPTED_ROUTES
        is_affected = r in HEAVILY_AFFECTED
        if is_disrupted:
            lw, alpha, z = 2.6, 1.0, 5
        elif is_affected:
            lw, alpha, z = 2.4, 1.0, 4
        else:
            lw, alpha, z = 1.4, 0.85, 3
        for seg in route_lines[r]:
            xs, ys = seg.xy
            ax.plot(xs, ys, color=col, lw=lw, alpha=alpha, zorder=z,
                    solid_capstyle="round")

    # Mark all stops with tiny dots
    sx = [p[0] for p in proj_xy.values()]
    sy = [p[1] for p in proj_xy.values()]
    ax.scatter(sx, sy, s=2, c="#666666", zorder=2, alpha=0.6, edgecolors="none")

    # Highlight the focused-OD endpoints with stars
    for sid, (name, marker, size) in KEY_DESTS.items():
        if sid in proj_xy:
            x, y = proj_xy[sid]
            ax.scatter(x, y, s=size, c="gold", marker=marker,
                       edgecolors="black", linewidths=1.2, zorder=10,
                       label=f"OD: {name}")
            ax.annotate(name, (x, y), xytext=(8, 8),
                        textcoords="offset points", fontsize=9,
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.25",
                                  fc="white", ec="black", alpha=0.85),
                        zorder=11)

    # Origin (Paradeplatz) with diamond
    if ORIGIN_ID in proj_xy:
        x, y = proj_xy[ORIGIN_ID]
        ax.scatter(x, y, s=240, c="#1f77b4", marker="D",
                   edgecolors="black", linewidths=1.2, zorder=10,
                   label="Paradeplatz (multi-OD origin)")
        ax.annotate("Paradeplatz\n(multi-OD origin)", (x, y), xytext=(10, -22),
                    textcoords="offset points", fontsize=9,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25",
                              fc="white", ec="#1f77b4", alpha=0.9),
                    zorder=11)

    # Add basemap (real OSM)
    print("Fetching basemap tiles...")
    sources = [
        cx.providers.OpenStreetMap.Mapnik,
        cx.providers.CartoDB.Positron,
    ]
    for src in sources:
        try:
            cx.add_basemap(ax, source=src, alpha=0.75, zoom=12)
            print(f"  basemap loaded from {src.get('name', 'unknown')}")
            break
        except Exception as e:
            print(f"  failed {src.get('name','?')}: {e}")
            continue
    else:
        print("  all basemap sources failed; using grey background")
        ax.set_facecolor("#f0f0f0")

    # Axis cosmetics
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    ax.set_title("Zürich GTFS sub-network used in BAPR-HRO real-data evaluation\n"
                 "(687 stops, 2000 connections, 31 routes, 35 days of GTFS-RT)",
                 fontsize=10)

    # Legend
    legend_handles = [
        mlines.Line2D([], [], color=TRAM_COLORS["2"], lw=3,
                      label="Disrupted on Oct 29 (lines 2,3,4,5,6,23)"),
        mlines.Line2D([], [], color=TRAM_COLORS["7"], lw=3,
                      label="Heavily affected (Line 7, +48min, 80% cancel)"),
        mlines.Line2D([], [], color="#9467bd", lw=2,
                      label="Other tram lines"),
        mlines.Line2D([], [], color="#cccccc", lw=2,
                      label="S-Bahn / regional rail"),
        mlines.Line2D([], [], color="#666666", marker="o", linestyle="",
                      markersize=4, label="Stops (687 total)"),
        mlines.Line2D([], [], color="#1f77b4", marker="D", linestyle="",
                      markeredgecolor="black", markersize=10,
                      label="Multi-OD origin (Paradeplatz)"),
        mlines.Line2D([], [], color="gold", marker="*", linestyle="",
                      markeredgecolor="black", markersize=14,
                      label="Focused-OD endpoints"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=8,
              framealpha=0.95, ncol=1)

    plt.tight_layout()
    plt.savefig(OUT + ".pdf", bbox_inches="tight", dpi=150)
    plt.savefig(OUT + ".png", bbox_inches="tight", dpi=180)
    print(f"\nSaved: {OUT}.pdf / .png")


if __name__ == "__main__":
    main()
