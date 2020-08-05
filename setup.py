# encoding: utf-8
"""
@author: ccj
@contact:
"""

import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()


setuptools.setup(
    name="template_cv",
    version="0.0.1",
    author="ccj",
    author_email='changjian1026@gmail.com',
    description="A template cv model for kaggle",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CHC278Cao/template_cv",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)