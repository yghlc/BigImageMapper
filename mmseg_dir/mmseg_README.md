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

```commandline
conda create -n open-mmlab python=3.8 -y
source activate open-mmlab
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
pip install mmsegmentation
```

Clone the modified version of MMSegmentation
```commandline
git clone https://github.com/yghlc/mmsegmentation ~/codes/PycharmProjects/yghlc_mmsegmentation
```

### Setting


### Training and prediction 


## Acknowledgement
Please cite [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) 
and the corresponding algorithms you used. 








