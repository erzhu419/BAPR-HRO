#!/bin/bash
# Run all Swiss experiments
# Usage: cd BAPR-HRO && bash experiments/swiss_full/run_all.sh

set -e
cd "$(dirname "$0")/../.."

echo "=========================================="
echo " BAPR-HRO Swiss Full Experiments"
echo "=========================================="
echo ""

mkdir -p experiments/swiss_full/results

echo "[1/5] CSA-MEAT baseline..."
python3 experiments/swiss_full/run_csa_meat_baseline.py 2>&1 | tee experiments/swiss_full/results/csa_meat.log
echo ""

echo "[2/5] Ablation study..."
python3 experiments/swiss_full/run_ablation.py 2>&1 | tee experiments/swiss_full/results/ablation.log
echo ""

echo "[3/5] Adapt-β convergence..."
python3 experiments/swiss_full/run_adapt_beta_convergence.py 2>&1 | tee experiments/swiss_full/results/convergence.log
echo ""

echo "[4/5] Scalability benchmark..."
python3 experiments/swiss_full/run_scalability.py 2>&1 | tee experiments/swiss_full/results/scalability.log
echo ""

echo "[5/5] Multi-OD experiment (longest, run last)..."
python3 experiments/swiss_full/run_multi_od.py 2>&1 | tee experiments/swiss_full/results/multi_od.log
echo ""

echo "=========================================="
echo " All experiments complete!"
echo " Results in: experiments/swiss_full/results/"
echo "=========================================="
ls -lh experiments/swiss_full/results/*.json 2>/dev/null
