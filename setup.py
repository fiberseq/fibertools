#!/usr/bin/env python

"""The setup script."""

from operator import contains
from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

# requirements = [
#    "pandas>=1.2",
#    "numpy>=1.20",
#    "polars>=0.13" "pysam>=0.16",
#    "numba>=0.53",
#    "pyranges",
#    "xgboost",
#    "sklearn",
# ]
# with open("requirements.txt") as requirements_file:
# requirements = [line.strip() for line in requirements_file]
with open("environment.yml") as requirements_file:
    requirements = []
    in_dependencies = False
    for line in requirements_file:
        line = line.strip()
        if "dependencies:" in line:
            in_dependencies = True
            continue
        if line.startswith("#") or not line.startswith("- ") or ":" in line:
            continue
        if in_dependencies:
            line = line.strip("- ")
            if "# skip" in line:
                continue
            requirements.append(line)

test_requirements = []

setup(
    author="Mitchell R. Vollger",
    author_email="mrvollger@gmail.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    description="A project for handling fiber-seq data.",
    entry_points={
        "console_scripts": [
            "fibertools=fibertools.fibertools:main",
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="fibertools",
    name="fibertools",
    packages=find_packages(include=["fibertools", "fibertools.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/mrvollger/fibertools",
    version="0.2.9",
    zip_safe=False,
)
