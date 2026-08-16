#!/bin/bash
# build_videos_220.sh -- adapted from your own build_videos.sh for the
# eval220 batch_runner.py output layout (single camera pass, no
# driver_pov/chase_cam split; route naming is bench2drive_XX not a
# hardcoded list).
#
# Usage:
#   ./build_videos_220.sh              # build videos for every route with viz output
#   ./build_videos_220.sh 4 42 167     # build videos only for these route indices

VIZ_ROOT="/data/ghazaleh/simlingo/eval220/viz"
OUT_DIR="/data/ghazaleh/simlingo/eval220/videos"
mkdir -p "$OUT_DIR"

build_video() {
  local viz_dir=$1
  local out_path=$2
  local label=$3

  IMG_DIR=$(find "$viz_dir" -type d -name "images" 2>/dev/null | head -1)
  if [ -z "$IMG_DIR" ]; then
    echo "!!! No images for $label -- skipping"
    return
  fi

  FRAMELIST=/tmp/frames_$(basename "$out_path" .mp4).txt
  > "$FRAMELIST"
  for f in $(ls "$IMG_DIR"/*.png | xargs -n1 basename | sed 's/\.png$//' | sort -n); do
    echo "file '$IMG_DIR/$f.png'" >> "$FRAMELIST"
    echo "duration 0.25" >> "$FRAMELIST"
  done
  # ffmpeg concat quirk: duration applies to the NEXT entry, so the last
  # file needs to be repeated once more at the end.
  tail -n 2 "$FRAMELIST" | head -n 1 >> "$FRAMELIST"

  ffmpeg -y -f concat -safe 0 -i "$FRAMELIST" -vsync vfr -pix_fmt yuv420p "$out_path" -loglevel error
  echo "Built: $out_path"
}

if [ "$#" -gt 0 ]; then
  ROUTES=()
  for idx in "$@"; do
    ROUTES+=("$(printf "bench2drive_%02d" "$idx")")
  done
else
  ROUTES=($(ls "$VIZ_ROOT" 2>/dev/null))
fi

for r in "${ROUTES[@]}"; do
  build_video "${VIZ_ROOT}/${r}/" "${OUT_DIR}/${r}.mp4" "$r"
done

echo "=== FINAL COUNT ==="
ls "$OUT_DIR"/*.mp4 2>/dev/null | wc -l