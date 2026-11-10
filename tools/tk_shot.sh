#!/bin/bash
# Capture one benchmark frame on the cell and file it on the workstation.
#
# Usage: tools/tk_shot.sh SESSION_DIR NAME "LAYOUT" [EXPECTED_MARKER_ID]
#   SESSION_DIR  folder name under data/zdf/ on the workstation, e.g. benchmark_2026-10-01
#   NAME         S02_<block>_<scene>_<NN>, e.g. S02_fill_poty-a_01
#   LAYOUT       slot occupancy as dictated, e.g. "L1-L8:U R1-R8:HO" (logged verbatim)
#
# Captures on the Jetson (3 tries), streams the ZDF to the workstation, verifies
# the checksum, deletes the Jetson copy (its disk is nearly full), appends the
# layout to capture_log.csv and runs the capture QA.
set -euo pipefail
SESSION="$1"; NAME="$2"; LAYOUT="$3"; EXPECT="${4:-}"
BENCH=Desktop/Gruppe3/testkoerper_benchmark
D=$BENCH/data/zdf/$SESSION
J=testkoerper_benchmark_capture/out
PREVIEWS="${TK_PREVIEWS:-/tmp/tk_previews}"
mkdir -p "$PREVIEWS"

ssh Workstationlernfabrik "mkdir -p ~/$D && test ! -e ~/$D/'$NAME'.zdf" || { echo "$NAME exists already"; exit 1; }
ssh lara5roboter "cd ~/testkoerper_benchmark_capture && for i in 1 2 3; do python3 tk_capture.py '$NAME' 2>&1 | grep -E '^$NAME:' && break; rm -f out/'$NAME'.zdf; sleep 2; done; test -e out/'$NAME'.zdf"
ssh lara5roboter "cat ~/$J/'$NAME'.zdf" | ssh Workstationlernfabrik "cat > ~/$D/'$NAME'.zdf"
A=$(ssh lara5roboter "sha256sum ~/$J/'$NAME'.zdf" | cut -c1-64)
B=$(ssh Workstationlernfabrik "sha256sum ~/$D/'$NAME'.zdf" | cut -c1-64)
[ "$A" = "$B" ] || { echo "CHECKSUM MISMATCH, Jetson copy kept"; exit 1; }
scp -q lara5roboter:"$J/${NAME}_preview.jpg" "$PREVIEWS/" || true
ssh lara5roboter "rm -f ~/$J/'$NAME'.zdf ~/$J/'${NAME}'_preview.jpg"

BLOCK=$(echo "$NAME" | cut -d_ -f2)
ssh Workstationlernfabrik "cd ~/$D && { test -e capture_log.csv || echo 'file,block,layout,time' > capture_log.csv; } && echo '$NAME.zdf,$BLOCK,\"$LAYOUT\",$(date -u +%FT%TZ)' >> capture_log.csv"

EXP=""; [ -n "$EXPECT" ] && EXP="--expect $EXPECT"
ssh Workstationlernfabrik "cd ~/$BENCH && .venv/bin/python tools/check_captures.py data/zdf/$SESSION --samples-dir data/samples/$SESSION $EXP --pattern '$NAME.zdf' 2>&1 | grep -E '^$NAME|WARN|warning'" || true
