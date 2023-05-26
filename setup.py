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
    version="0.3.2",
    # Chose a license from here: https: //
    # help.github.com / articles / licensing - a -
    # repository. For example: MIT
    license="MIT",
    # Short description of your library
    description="A package for automatic 2D/3D registration of X-ray and CT images",
    # Long description of your library
    long_description="This package is for automatic 2D/3D registration of X-ray and CT images. The package contains a 2D/3D registration solver and a 2D landmark detector. The 2D/3D registration solver is inherited from xReg, and the 2D landmark detector is inherited from Synthex.",
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
        "h5py>=3.7.0",
        "imgaug>=0.4.0",
        "matplotlib>=3.5.2",
        "ml_collections>=0.1.1",
        "numpy>=1.24.2",
        "opencv_python>=4.6.0.66",
        "pandas>=1.4.3",
        "Pillow>=9.3.0",
        "pydicom>=2.3.0",
        "scipy>=1.9.0",
        "seaborn>=0.12.2",
        "torch>=1.12.0",
        "torchvision>=0.13.0",
    ],
    # https://pypi.org/classifiers/
    classifiers=[],
)
