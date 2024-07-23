# Demo

## Installation
Currently tested on **Linux** only. Install **anaconda** or **miniconda**

```bash
conda create -n Demo python=3.10 pip numpy>=1.26,<2.0 -y
conda activate Demo
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```
Install `mmpose` package: https://mmpose.readthedocs.io/en/latest/installation.html


## Running the demo

### Default
To run the demo with the default parameters, just execute the `run.sh` script:

```bash
./run.sh
```
This will run the size estimation model with a YoloV8 detector. To select a YoloV8 segmentation model just change the `model` parameter (default value is `scalebar`):

```bash
./run.sh model=scalebar-seg
```

### Camera selection

You can change the camera by setting the camera ID in the terminal:

```bash
CAM=2 ./run.sh
```

### Pose Estimation

Another supported model is the pose estimator from `mmpose`. This one runs the 2D-Pose estimation model:

```bash
./run.sh model=pose
```

To run the 3D-Pose estimation model use the following:
```bash
./run.sh model=pose model.threed=true
```
