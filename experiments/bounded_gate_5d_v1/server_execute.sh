#!/usr/bin/env bash
set -euo pipefail
cd /home/erzhu419/BAPR-HRO
export PYTHONPATH=/home/erzhu419/BAPR-HRO
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PY=""
for cand in /home/erzhu419/miniconda3/envs/*/bin/python3 /home/erzhu419/miniconda3/bin/python3 /home/erzhu419/.conda/envs/*/bin/python3 /home/erzhu419/anaconda3/envs/*/bin/python3 /home/erzhu419/anaconda3/bin/python3 /home/erzhu419/.venvs/*/bin/python3 /usr/bin/python3; do
    if [ -x "$cand" ] && "$cand" -c "import numpy, google.transit.gtfs_realtime_pb2, matplotlib" >/dev/null 2>&1; then
        PY="$cand"
        break
    fi
done
if [ -z "$PY" ]; then
    echo "ERROR: No Python with numpy + gtfs_realtime_pb2 + matplotlib found!"
    for cand in /home/erzhu419/miniconda3/envs/*/bin/python3 /home/erzhu419/miniconda3/bin/python3 /home/erzhu419/.conda/envs/*/bin/python3; do
        if [ -x "$cand" ]; then
            echo "Checking $cand:"
            "$cand" -c "import sys; print(sys.executable); import numpy; print('numpy ok'); import google.transit.gtfs_realtime_pb2; print('gtfs ok'); import matplotlib; print('mpl ok')" || true
        fi
    done
    exit 1
fi

echo "=== Selected Python: $PY ==="
"$PY" -c "import experiments.bounded_gate_5d_v1.pipeline, experiments.bounded_gate_5d_v1.run, experiments.bounded_gate_5d_v1.summarize; print('Remote module imports verified')"

echo "=== [1/5] Metadata Prescreen (No Target Values Opened) ==="
"$PY" -m experiments.bounded_gate_5d_v1.pipeline prescreen --directory /home/erzhu419/BAPR-HRO/experiments/bounded_gate_5d_v1

echo "=== [2/5] Freeze 5 Unexposed Dates & 60 Public Menus ==="
"$PY" -m experiments.bounded_gate_5d_v1.pipeline freeze --directory /home/erzhu419/BAPR-HRO/experiments/bounded_gate_5d_v1

echo "=== [3/5] Attach ist-daten & Decode GTFS-RT ==="
"$PY" -m experiments.bounded_gate_5d_v1.pipeline decode --directory /home/erzhu419/BAPR-HRO/experiments/bounded_gate_5d_v1 --workers 5

echo "=== [4/5] Run 8-Arm Paired Replay + Sensitivity (60 Roots) ==="
"$PY" -m experiments.bounded_gate_5d_v1.run --directory /home/erzhu419/BAPR-HRO/experiments/bounded_gate_5d_v1 --workers 18

echo "=== [5/5] Summarize, Calibrate, Audit & Plot ==="
"$PY" -m experiments.bounded_gate_5d_v1.summarize --directory /home/erzhu419/BAPR-HRO/experiments/bounded_gate_5d_v1

echo "=== ALL_STAGES_COMPLETE ==="
