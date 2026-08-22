#!/bin/bash
# build_videos_220.sh -- v2: parallel encoding, skip-if-already-built,
# and actually checks ffmpeg's exit code instead of always claiming success.
#
# Usage:
#   ./build_videos_220.sh              # build videos for every route with viz output
#   ./build_videos_220.sh 4 42 167     # build videos only for these route indices
#   PARALLEL_JOBS=12 ./build_videos_220.sh   # override the default parallelism

VIZ_ROOT="/tmp/video_test/eval220_viz"
OUT_DIR="/tmp/video_test/eval220_out"
# CPU-bound work, no GPU/CARLA involved -- but still a shared 32-core box
# with other users regularly at ~50% load, so leaving real headroom rather
# than maxing out every core.
PARALLEL_JOBS="${PARALLEL_JOBS:-6}"
mkdir -p "$OUT_DIR"

build_video() {
  local viz_dir=$1
  local out_path=$2
  local label=$3

  if [ -f "$out_path" ]; then
    echo "SKIP (already built): $label"
    return 0
  fi

  # Retried routes have one images/ folder per attempt (each attempt gets a
  # fresh timestamped debug_viz path) -- take the most recently modified
  # one, which is the attempt that actually finished.
  IMG_DIR=$(find "$viz_dir" -type d -name "images" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
  if [ -z "$IMG_DIR" ]; then
    echo "SKIP (no images found): $label"
    return 0
  fi

  FRAMELIST="/tmp/frames_$(basename "$out_path" .mp4)_$$.txt"
  > "$FRAMELIST"
  for f in $(ls "$IMG_DIR"/*.png | xargs -n1 basename | sed 's/\.png$//' | sort -n); do
    echo "file '$IMG_DIR/$f.png'" >> "$FRAMELIST"
    echo "duration 0.25" >> "$FRAMELIST"
  done
  # ffmpeg concat quirk: duration applies to the NEXT entry, so the last
  # file needs to be repeated once more at the end.
  tail -n 2 "$FRAMELIST" | head -n 1 >> "$FRAMELIST"

  if ffmpeg -y -f concat -safe 0 -i "$FRAMELIST" -vsync vfr -pix_fmt yuv420p "$out_path" -loglevel error; then
    echo "OK: $label"
  else
    echo "FAILED (ffmpeg exit $?): $label"
  fi
  rm -f "$FRAMELIST"
}
export -f build_video

if [ "$#" -gt 0 ]; then
  ROUTES=()
  for idx in "$@"; do
    ROUTES+=("$(printf "bench2drive_%02d" "$idx")")
  done
else
  ROUTES=($(ls "$VIZ_ROOT" 2>/dev/null))
fi

echo "Building ${#ROUTES[@]} route video(s), $PARALLEL_JOBS at a time..."

export VIZ_ROOT OUT_DIR
printf '%s\n' "${ROUTES[@]}" | xargs -P "$PARALLEL_JOBS" -I{} \
  bash -c 'build_video "$VIZ_ROOT/{}/" "$OUT_DIR/{}.mp4" "{}"'

echo "=== FINAL COUNT ==="
built_count=$(ls "$OUT_DIR"/*.mp4 2>/dev/null | wc -l)
echo "$built_count video(s) present in $OUT_DIR (requested: ${#ROUTES[@]} -- check output above for any SKIP/FAILED lines)"