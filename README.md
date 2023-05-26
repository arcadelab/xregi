<p align="center"><img src="doc/figs/xregi logo.png" alt="xregi logo" style="zoom:5%;"></p>

<div align='center'>
<a href="https://colab.research.google.com/github/shez12/xregi/blob/dev-syn/xregi.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Tutorial in Colab" />
</a>
<a href="https://www.python.org/"><img src='https://img.shields.io/badge/Made%20with-Python-1f425f.svg'>
</a>
<a href="https://pypi.org/project/xregi/"><img src='https://badge.fury.io/py/xregi.svg'>
</a>
<a href="https://opensource.org/licenses/MIT">
<img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
</a>
</div>


This is a python package for registering x-ray images and CT scans. It is based on the [xReg](https://github.com/rg2/xreg), a C++ library for medical image registration, and [synthex](https://github.com/arcadelab/SyntheX), a python package for synthetic x-ray imaging.

## Third-party libraries
Before you start, please make sure you have all the dependencies installed. The following libraries are required:
- [Total Segmentator](https://github.com/wasserth/TotalSegmentator)
- [xReg](https://github.com/rg2/xreg)
- [SyntheX](https://github.com/arcadelab/SyntheX)

### Install TotalSegmentator
Total segmentator can be installed through pip
```bash
pip install TotalSegmentator
```

### Install xReg
xReg is a C++ library for medical image registration. It is used as the backend of xregi. To install xReg, please follow the instructions in the [README.md of xreg](https://github.com/rg2/xreg/blob/master/README.md)
On other environments, such as Windows, MacOS and Ubuntu，you may need to install xreg aside according to your system. The detailed information can be found at the `Building` section in the README.md of xreg.

### Install SyntheX
Synthex will be installed along with xregi. If you want to install it separately, here is the installation for SyntheX:
```bash
git clone https://github.com/arcadelab/SyntheX.git
cd /SyntheX
conda install .
```

## Xregi Installation Guide
### Install through pip
On ubuntu 20.04, simply install this package using pip
```bash
pip install xregi
```


### Install from source
On ubuntu 20.04, download the source code and install it under xregi path
```bash
git clone https://github.com/shez12/xregi
git checkout master
```
Fetch the source data and example images from [this link](https://drive.google.com/drive/folders/1XzQgWfMVtkCq-Nnk2l_lE3UWeG2kEnyc?usp=share_link) or 
```bash
#download data.zip
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1wjrxNE6B0pX3IooGxwC_cjf4n8MhxU1p' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1wjrxNE6B0pX3IooGxwC_cjf4n8MhxU1p" -O data.zip && rm -rf /tmp/cookies.txt

#unzip data.zip
cd Downloads/
unzip data.zip
```


## Usage
Before using xregi, you need to move the data folder to xregi directory.
```bash
mv data xregi/
```

xregi supports command line interactions and API. To use the API, 
```python
import xregi

reg_ = Registration2D3D().load()

reg_.solve()

...
```

To use the command line interface, simply run
```bash
python -m xregi 
```

To specify the input and output path, go to the xregi directory and change the values in config.yaml, for example:
```yaml
"CT_segmentation_path": "/home/jeremy/Documents/xregi/xregi/data/pelvis_seg.nii.gz" 
# change this directory to your own absolute path
```

## Contributors
[Jiaming (Jeremy) Zhang](https://jeremyzz830.github.io/), MSE in Robotics, Johns Hopkins University

[Zhangcong She](https://github.com/shez12), MSE in Mechanical Engineering, Johns Hopkins University

[Benjamin D. Killeen](https://benjamindkilleen.com/), PhD in Computer Science, Johns Hopkins University
