import os

from setuptools import find_packages, setup
__version__ = 1
setup(
    name="toy_gym",
    description="Set of OpenAI/gym robotic environments based on PyBullet physics engine.",
    author="Khanh Quynh Nguyen",
    author_email="nkquynh1998@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    package_data={"toy_gym": ["version.txt"]},
    version=__version__,
    install_requires=["gym", "pybullet", "numpy", "scipy"],
    extras_require={
        "tests": ["pytest", "black", "pytype"],
        "extra": ["numpngw", "stable-baselines3"],
        "docs": ["sphinx", "sphinx-rtd-theme"],
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
