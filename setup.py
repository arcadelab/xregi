from setuptools import find_packages
from distutils.core import setup
import os

# User-friendly description from README.md
current_directory = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(current_directory, "README.md"), encoding="utf-8") as f:
        long_description = f.read()
except Exception:
    long_description = ""

setup(
    # Name of the package
    name="xregi",
    # Packages to include into the distribution
    packages=find_packages("."),
    # Start with a small number and increase it with
    # every change you make https://semver.org
    version="0.1.0",
    # Chose a license from here: https: //
    # help.github.com / articles / licensing - a -
    # repository. For example: MIT
    license="MIT",
    # Short description of your library
    description="A package for automatic 2D/3D registration of X-ray and CT images",
    # Long description of your library
    # long_description=long_description,
    long_description_content_type="markdown",
    # Your name
    author="Jiaming Zhang",
    # Your email
    author_email="jzhan282@jhu.edu",
    # Either the link to your github or to your website
    url="https://github.com/shez12/xregi",
    # Link from which the project can be downloaded
    download_url="",
    # List of keywords
    keywords=[],
    # List of packages to install with this one
    install_requires=[
        "absl-py==1.2.0",
        "argparse==1.4.0",
        "batchgenerators==0.24",
        "cachetools==5.2.0",
        "charset-normalizer==2.1.0",
        "contextlib2==21.6.0",
        "cycler==0.11.0",
        "dicom2nifti==2.4.3",
        "fonttools==4.34.4",
        "future==0.18.2",
        # 'google-auth==2.9.1',
        # 'google-auth-oauthlib==0.4.6',
        "grpcio==1.48.2",
        "h5py==3.7.0",
        "idna==3.3",
        "imageio==2.21.0",
        "imgaug==0.4.0",
        "importlib-metadata==4.12.0",
        "joblib==1.1.1",
        "kiwisolver==1.4.4",
        "linecache2==1.0.0",
        "markdown==3.4.1",
        "markupsafe==2.1.1",
        "matplotlib==3.5.2",
        "medpy==0.4.0",
        "ml-collections==0.1.1",
        "networkx==2.8.5",
        "nibabel==4.0.1",
        "numpy==1.23.1",
        "oauthlib==3.2.0",
        "opencv-python==4.6.0.66",
        "packaging==21.3",
        "pandas==1.5.3",
        "pillow==9.2.0",
        # 'protobuf==3.19.4',
        "pyasn1==0.4.8",
        "pyasn1-modules==0.2.8",
        "pydicom==2.3.0",
        "pyparsing==3.0.9",
        "python-dateutil==2.8.2",
        "python-gdcm==3.0.14",
        "pytz==2022.5",
        "pywavelets==1.3.0",
        "pyyaml==6.0",
        "requests==2.28.1",
        "requests-oauthlib==1.3.1",
        "rsa==4.9",
        "scikit-image==0.19.3",
        "scikit-learn==1.1.1",
        "scipy==1.9.0",
        "shapely==1.8.2",
        "simpleitk==2.1.1.2",
        "six==1.16.0",
        "sklearn==0.0",
        # 'tensorboard==2.9.1',
        # 'tensorboard-data-server==0.6.1',
        # 'tensorboard-plugin-wit==1.8.1',
        "threadpoolctl==3.1.0",
        "tifffile==2022.7.31",
        "torch==2.0.0",
        # 'torchvision==0.13.0',
        "tqdm==4.64.0",
        "traceback2==1.4.0",
        "typing-extensions==4.3.0",
        "unittest2==1.1.0",
        "urllib3==1.26.11",
        # 'werkzeug==2.2.1',
        "zipp==3.8.1",
    ],
    # https://pypi.org/classifiers/
    classifiers=[],
)
