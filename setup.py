#!/usr/bin/env python3

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
     name='PE2I_denoiser',
     version="1.0",
     author="Claes Ladefoged",
     author_email="claes.noehr.ladefoged@regionh.dk",
     description="Denoise PE2I data",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/CAAI/PE2I_denoiser",
     scripts=[
             'PE2I_denoiser/PE2I_denoiser',
     ],
     packages=find_packages(include=['PE2I_denoiser']),
     install_requires=[
         'onnxruntime',
         'numpy',
         'torchio',
         'nipype',
         'torch>=1.12',
     ],
     classifiers=[
         'Programming Language :: Python :: 3.8',
     ],
 )
