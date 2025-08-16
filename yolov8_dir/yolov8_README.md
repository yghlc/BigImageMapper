## YOLOv8
Using [YOLOv8](https://github.com/ultralytics/ultralytics) for detecting landforms 
from remote sensing imagery.

## Install
Please following the instruction at [YOLOv8](https://github.com/ultralytics/ultralytics) 
to install Python and create a conda environment (name: pytorch). 

```
## In the conda environment, install the forked version
#git clone https://github.com/yghlc/ultralytics.git
#cd ultralytics
#pip install -e '.[dev]'

# Install the ultralytics package from PyPI
pip install ultralytics

# other packages
conda install -c conda-forge rasterio
pip install scikit-image
pip install GPUtil

```

Other dependencies please follow the main readme file to install.

## How to use
