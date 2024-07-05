#!/bin/bash
set -Eeuo pipefail
eval "$(conda shell.bash hook)"

PYTHON="python"
# PYTHON="scalene --cpu --gpu --memory ---" # require !pip install scalene

v4l2-ctl -d /dev/video0 --set-ctrl=power_line_frequency=1

conda activate ${CONDA_ENV:-Demos}
PARAMS=""
if [[ ! -z $@ ]]; then
    PARAMS="--config-name $@"
fi

$PYTHON src/main.py $PARAMS
