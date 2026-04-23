#!/bin/bash
# Download Swiss GTFS Static + GTFS-RT data for Durner replication
# Medium plan: 1 month of RT data (2023-10-01 ~ 2023-11-04)
#
# Usage:
#   chmod +x download_swiss_gtfs.sh
#   ./download_swiss_gtfs.sh [output_dir]
#
# Default output: ./swiss_data/

set -e

BASE_URL="https://mirror.traines.eu"
OUT_DIR="${1:-./swiss_data}"
STATIC_DIR="$OUT_DIR/gtfs-static"
RT_DIR="$OUT_DIR/gtfs-rt"

mkdir -p "$STATIC_DIR" "$RT_DIR"

echo "============================================"
echo " Swiss GTFS Data Downloader"
echo " Static: 2023-10-30"
echo " RT:     2023-10-01 ~ 2023-11-04 (35 days)"
echo " Est. download: ~5GB"
echo "============================================"
echo ""

# --- 1. GTFS Static (timetable for simulation period) ---
echo "[1/2] Downloading GTFS Static (2023-10-30)..."
wget -q --show-progress -P "$STATIC_DIR" \
    "$BASE_URL/swiss-gtfs/2023-10-30/" \
    -r -np -nH --cut-dirs=2 -R "index.html*" \
    2>&1 || {
    echo "wget recursive failed, trying direct ZIP..."
    wget -q --show-progress -O "$STATIC_DIR/gtfs-2023-10-30.zip" \
        "$BASE_URL/swiss-gtfs/2023-10-30/gtfs-2023-10-30.zip" 2>&1 || true
}
echo "  -> Static data saved to $STATIC_DIR"
echo ""

# --- 2. GTFS-RT (realtime, one per day, tar.bz2) ---
echo "[2/2] Downloading GTFS-RT (2023-10-01 ~ 2023-11-04)..."

START_DATE="2023-10-01"
END_DATE="2023-11-04"

# Cross-platform date iteration
current="$START_DATE"
total=0
downloaded=0
failed=0

# Count total days first
d="$START_DATE"
while [[ "$d" < "$END_DATE" ]] || [[ "$d" == "$END_DATE" ]]; do
    total=$((total + 1))
    d=$(date -d "$d + 1 day" +%Y-%m-%d 2>/dev/null || date -j -v+1d -f "%Y-%m-%d" "$d" +%Y-%m-%d 2>/dev/null)
done

echo "  Total days to download: $total"
echo ""

current="$START_DATE"
while [[ "$current" < "$END_DATE" ]] || [[ "$current" == "$END_DATE" ]]; do
    file="${current}.tar.bz2"
    url="$BASE_URL/swiss-gtfs-rt/$file"
    dest="$RT_DIR/$file"

    downloaded=$((downloaded + 1))

    if [ -f "$dest" ]; then
        echo "  [$downloaded/$total] $file (already exists, skipping)"
    else
        echo -n "  [$downloaded/$total] $file ... "
        if wget -q --show-progress -O "$dest" "$url" 2>&1; then
            size=$(du -h "$dest" | cut -f1)
            echo "  done ($size)"
        else
            echo "  FAILED (may not exist on server)"
            rm -f "$dest"
            failed=$((failed + 1))
        fi
    fi

    # Next day
    current=$(date -d "$current + 1 day" +%Y-%m-%d 2>/dev/null || \
              date -j -v+1d -f "%Y-%m-%d" "$current" +%Y-%m-%d 2>/dev/null)
done

echo ""
echo "============================================"
echo " Download complete!"
echo " Static:     $STATIC_DIR"
echo " RT files:   $RT_DIR"
echo " Downloaded: $((downloaded - failed))/$total days"
[ $failed -gt 0 ] && echo " Failed:     $failed days"
echo ""
echo " Total disk usage:"
du -sh "$OUT_DIR"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Unzip static:  cd $STATIC_DIR && unzip *.zip"
echo "  2. Extract RT:    cd $RT_DIR && for f in *.tar.bz2; do tar xjf \$f; done"
echo "  3. Feed into your GTFS loader"