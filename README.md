# Demo

## Installation
Currently tested on **Linux** only. Install **anaconda** or **miniconda**

```bash
conda create -n Demo python=3.10 pip numpy>=1.26,<2.0 -y
conda activate Demo
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
# or for cpu-only (e.g. raspberry):
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```
(optional) Install `mmpose` package: https://mmpose.readthedocs.io/en/latest/installation.html


### Installation for hailo
First, install the Hailo drivers based on [Getting Started](https://www.raspberrypi.com/documentation/computers/ai.html#getting-started) from their website.
After the installation check if you can access the hailo platform package:

```bash
python -c "import hailo_platform"
```
If no ImportError is raised, you are ready to set up the virtual environment:

```bash
python -m venv --system-site-packages demo-env
source demo-env/bin/activate
```
Check again, that the hailo platform package is accessible from the virtual environment (`--system-site-packages` is important to allow access to the system packages):

```bash
python -c "import hailo_platform"
```

Now, get and install hailo-model-zoo:

```bash
cd .. # go to the parent directory
git clone git@github.com:tzlr-de/hailo_model_zoo.git
cd hailo_model_zoo
pip install -e . # this will take some minutes, since some packages has to be built
```

Finally, you can install the requirements for the demo. But before that uninstall the `opencv-python` package, because we require the headless version:

```bash
pip uninstall opencv-python
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu # < since we are on a raspberry, we need the CPU version of pytorch
```


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
