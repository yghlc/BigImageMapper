## MMSeg:

The scripts in this folder uses [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
for delineating landforms. [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) is OpenMMLab 
Semantic Segmentation Toolbox and Benchmark and provides many semantic segmentation
algorithms. 

## How to use

### Install MMSeg
Please follow the instruction available on [MMsegmentation user guide](https://mmsegmentation.readthedocs.io/en/latest/get_started.html#installation) 
to install MMsegmentation. Here is a few brief command lines 
(Python=3.8, Pytorch=1.10.0 with CUDA=11.3):

```
conda create -n open-mmlab python=3.8 -y
conda activate open-mmlab  #(or source activate open-mmlab)
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
pip install mmsegmentation
```

Clone the modified version of MMSegmentation
```
git clone https://github.com/yghlc/mmsegmentation ~/codes/PycharmProjects/yghlc_mmsegmentation
```

Other packages: GDAL, rasterio, geopandas etc. 
```
conda activate open-mmlab #(or source activate open-mmlab)
conda env update -f uist_open-mmlab.yml
```

### Setting

Create a working folder, then copy files for training and prediciton. 
```
mmseg=~/codes/PycharmProjects/Landuse_DL/mmseg_dir
mkdir mmseg-working-dir #(change to a meaningful name)
cd mmseg-working-dir
cp ${mmseg}/main_para.ini  .    # this is the main setting file and need to modifiy it 
cp ${mmseg}/mmseg_deeplabv3plus.ini . # the setting of network, we can choose differnt backbones and methods from mmsegmentation
cp ${mmseg}/exe_mmseg.sh  .     # the bash to run training and prediction
cp ${mmseg}/../ini_files/area_Willow_River.ini  # setting of study area and data
```

### Training and prediction 

Run training and prediction
```
./exe_mmseg.sh 
```

## Acknowledgement
Please cite [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) 
and the corresponding algorithms you used. 








