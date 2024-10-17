#!/bin/bash
set -Eeuo pipefail
eval "$(conda shell.bash hook)"

PYTHON="python"
# PYTHON="scalene --cpu --gpu --memory ---" # require !pip install scalene

for dev in /dev/video*; do
    if [[ -e $dev ]]; then
        echo "Found camera at $dev"
    fi
done

CAM=${CAM:-0}

v4l2-ctl -d /dev/video${CAM} --set-ctrl power_line_frequency=0,sharpness=255
# v4l2-ctl -d /dev/video2 --set-ctrl power_line_frequency=0

conda activate ${CONDA_ENV:-Demo}
$PYTHON src/main.py cam.id=$CAM $@
