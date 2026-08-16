#!/bin/bash
build_video() {
  local viz_dir=$1
  local out_path=$2
  local label=$3

  IMG_DIR=$(find "$viz_dir" -type d -name "images" 2>/dev/null | head -1)
  if [ -z "$IMG_DIR" ]; then
    echo "!!! No images for $label -- skipping"
    return
  fi

  FRAMELIST=/tmp/frames_$(basename $out_path .mp4).txt
  > $FRAMELIST
  for f in $(ls "$IMG_DIR"/*.png | xargs -n1 basename | sed 's/\.png$//' | sort -n); do
    echo "file '$IMG_DIR/$f.png'" >> $FRAMELIST
    echo "duration 0.25" >> $FRAMELIST
  done
  tail -n 2 $FRAMELIST | head -n 1 >> $FRAMELIST

  ffmpeg -y -f concat -safe 0 -i $FRAMELIST -vsync vfr -pix_fmt yuv420p "$out_path" -loglevel error
  echo "Built: $out_path"
}

declare -a ROUTES=(bench2drive_12 bench2drive_64 bench2drive_77 bench2drive_118 bench2drive_167 bench2drive_168 bench2drive_170 bench2drive_187 bench2drive_200 bench2drive_213)

for r in "${ROUTES[@]}"; do
  build_video "/data/ghazaleh/simlingo/eval10_orion_match/driver_pov/viz/${r}/" "/data/ghazaleh/simlingo/eval10_orion_match/driver_pov/${r}.mp4" "${r} (driver POV)"
  build_video "/data/ghazaleh/simlingo/eval10_orion_match/chase_cam/viz/${r}/" "/data/ghazaleh/simlingo/eval10_orion_match/chase_cam/${r}.mp4" "${r} (chase cam)"
done

echo "=== FINAL COUNT ==="
ls /data/ghazaleh/simlingo/eval10_orion_match/driver_pov/*.mp4 2>/dev/null | wc -l
ls /data/ghazaleh/simlingo/eval10_orion_match/chase_cam/*.mp4 2>/dev/null | wc -l