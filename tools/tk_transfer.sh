#!/bin/bash
# Move finished auto-captures from the Jetson to the workstation, forever.
#
# Usage: tools/tk_transfer.sh SESSION_DIR
# For every <name>.done on the Jetson: stream the ZDF to the workstation, verify
# the checksum, copy the preview, delete the Jetson copy, convert to a sample and
# append a PENDING_REVIEW row to capture_log.csv (the reviewer fills in the layout).
set -uo pipefail
SESSION="$1"
BENCH=Desktop/Gruppe3/testkoerper_benchmark
D=$BENCH/data/zdf/$SESSION
J=testkoerper_benchmark_capture/out
ssh Workstationlernfabrik "mkdir -p ~/$D/previews"
while true; do
  for NAME in $(ssh lara5roboter "cd ~/$J && ls *.done 2>/dev/null | sed 's/\.done$//'"); do
    ssh lara5roboter "cat ~/$J/$NAME.zdf" | ssh Workstationlernfabrik "cat > ~/$D/$NAME.zdf"
    A=$(ssh lara5roboter "sha256sum ~/$J/$NAME.zdf" | cut -c1-64)
    B=$(ssh Workstationlernfabrik "sha256sum ~/$D/$NAME.zdf" | cut -c1-64)
    if [ "$A" != "$B" ]; then echo "$(date +%T) $NAME checksum mismatch, retrying later"; continue; fi
    ssh lara5roboter "cat ~/$J/${NAME}_preview.jpg" | ssh Workstationlernfabrik "cat > ~/$D/previews/${NAME}_preview.jpg"
    ssh lara5roboter "rm -f ~/$J/$NAME.zdf ~/$J/${NAME}_preview.jpg ~/$J/$NAME.done"
    ssh Workstationlernfabrik "cd ~/$BENCH && .venv/bin/python tools/check_captures.py data/zdf/$SESSION --samples-dir data/samples/$SESSION --pattern '$NAME.zdf' >/dev/null 2>&1; cd ~/$D && echo '$NAME.zdf,auto,\"PENDING_REVIEW\",$(date -u +%FT%TZ)' >> capture_log.csv"
    echo "$(date +%T) $NAME transferred"
  done
  sleep 3
done
