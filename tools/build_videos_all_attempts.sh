#!/bin/bash
# build_videos_all_attempts.sh -- one video per ATTEMPT folder (including
# failed attempts), reading frames directly from the ehsan-rw archive.
# Nothing is written to the local disk; videos land next to the archive.
#
# Usage:
#   bash tools/build_videos_all_attempts.sh            # build all missing
#   PARALLEL_JOBS=4 bash tools/build_videos_all_attempts.sh
# Idempotent: existing videos are skipped, so it can be re-run anytime.

VIZ_ROOT="/ehsan-rw/m.sadegh/eval220v2/viz"
OUT_DIR="/ehsan-rw/m.sadegh/eval220v2/videos_all"
PARALLEL_JOBS="${PARALLEL_JOBS:-6}"
mkdir -p "$OUT_DIR"

build_one() {
  local img_dir=$1 out_path=$2
  if [ -f "$out_path" ]; then
    echo "SKIP (exists): $out_path"
    return 0
  fi
  FRAMELIST="/tmp/frames_all_$$"
  > "$FRAMELIST"
  for f in $(ls "$img_dir"/*.png 2>/dev/null | xargs -n1 basename 2>/dev/null | sed 's/\.png$//' | sort -n); do
    echo "file '$img_dir/$f.png'" >> "$FRAMELIST"
    echo "duration 0.25" >> "$FRAMELIST"
  done
  if ! [ -s "$FRAMELIST" ]; then
    echo "SKIP (no frames): $out_path"
    rm -f "$FRAMELIST"
    return 0
  fi
  # ffmpeg concat quirk: the last file must be repeated once
  tail -n 2 "$FRAMELIST" | head -n 1 >> "$FRAMELIST"
  if ffmpeg -y -f concat -safe 0 -i "$FRAMELIST" -vsync vfr -pix_fmt yuv420p "$out_path" -loglevel error; then
    echo "OK: $(basename "$out_path")"
  else
    echo "FAILED: $out_path"
  fi
  rm -f "$FRAMELIST"
}
export -f build_one
export OUT_DIR VIZ_ROOT

find "$VIZ_ROOT" -type d -name images | while read -r IMG; do
  REL=${IMG#"$VIZ_ROOT"/}                     # bench2drive_XXX/<save_name>/.../images
  ROUTE=$(echo "$REL" | cut -d/ -f1)
  SAVE=$(echo "$REL" | cut -d/ -f2)
  echo "$IMG|$OUT_DIR/${ROUTE}__${SAVE}.mp4"
done | xargs -P "$PARALLEL_JOBS" -I{} bash -c 'IFS="|" read -r i o <<< "{}"; build_one "$i" "$o"'

echo "=== done: $(ls "$OUT_DIR"/*.mp4 2>/dev/null | wc -l) video(s) in $OUT_DIR"