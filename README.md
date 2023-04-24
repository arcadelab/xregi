<h1 align='center'>XREGI</h1>

<div align='center'>
<a href="https://colab.research.google.com/github/shez12/xregi/blob/dev-syn/xregi.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Tutorial in Colab" />
</a>
<a href="https://www.python.org/"><img src='https://img.shields.io/badge/Made%20with-Python-1f425f.svg'>
</a>
<a href="https://pypi.python.org/pypi/ansicolortags/"><img src='https://badge.fury.io/py/ansicolortags.svg'>
</a>
</div>


This is a python package for registering x-ray images and CT scans. It is based on the [xReg](https://github.com/rg2/xreg), a C++ library for medical image registration, and [synthex](https://github.com/arcadelab/SyntheX), a python package for synthetic x-ray imaging.

## Installation
### Install through pip
On ubuntu 20.04, simply install this package using pip
```bash
pip install xregi
```
On other environments, such as Windows, MacOS and Ubuntu，you may need to install xreg aside according to your system. The detailed information can be found at the `Building` section in the [README.md of xreg](https://github.com/rg2/xreg/blob/master/README.md)

### Install from source
On ubuntu 20.04, download the source code and install it under xregi path
```bash
pip install .
```
Fetch the source data and example images from [here](https://drive.google.com/drive/folders/1XzQgWfMVtkCq-Nnk2l_lE3UWeG2kEnyc?usp=share_link) or 
```bash
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-FGszriem5Vr130kw0BYtKM1QXJnD3_f' -O real_label.h5

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1hENQgZ0s1t0BzF28Ca8DaLJN7HFFf9p6" -O yy_checkpoint_net_20.pt && rm -rf /tmp/cookies.txt

```


## Usage
xregi supports command line interactions and API. To use the 
```python
import xregi

reg_ = Registration2D3D().load()

reg_.solve()

...
```

## Third-party libraries
- [Total Segmentator](https://github.com/wasserth/TotalSegmentator)

## Contributors
[Jiaming (Jeremy) Zhang](https://jeremyzz830.github.io/), MSE in Robotics, Johns Hopkins University

[Zhangcong She](https://github.com/shez12), MSE in Mechanical Engineering, Johns Hopkins University

[Benjamin D. Killeen](https://benjamindkilleen.com/), PhD in Computer Science, Johns Hopkins University
