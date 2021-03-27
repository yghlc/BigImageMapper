# YOLTv4
Objection Detection based on YoloV4: https://github.com/avanetten/yoltv4

My fork version: https://github.com/yghlc/yoltv4


## Installation
Because our CURC and other system do not support Docker, we build Singularity container for running YOLTv4.
Following by Dockerfile to create a singularity definition file, but install conda environment in the host machine. 

Install conda YOLTv4:
    conda update conda && conda config --prepend channels conda-forge
    conda create -n yoltv4 python=3.6
    conda activate yoltv4
    conda install -n yoltv4  gdal=2.4.2 geopandas=0.6.3 fiona rasterio awscli affine pyproj pyhamcrest cython fiona h5py \
                jupyter jupyterlab ipykernel libgdal matplotlib ncurses numpy statsmodels pandas pillow pip scipy \
                scikit-image scikit-learn shapely rtree testpath tqdm opencv
    






