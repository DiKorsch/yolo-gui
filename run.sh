#!/bin/bash
set -Eeuo pipefail
CONDA_ENV=${CONDA_ENV:-Demo}

# if CONDA_ENV is pip, then use pipenv
if [[ "$CONDA_ENV" != "pip" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV:-Demo}
fi

PYTHON="python"
# PYTHON="scalene --cpu --gpu --memory ---" # require !pip install scalene
v4l2-ctl --list-devices

CAM=${CAM:-0}

# v4l2-ctl -d /dev/video${CAM} --set-ctrl power_line_frequency=0 #,sharpness=255
# v4l2-ctl -d /dev/video2 --set-ctrl power_line_frequency=0

$PYTHON src/main.py cam.id=$CAM $@
