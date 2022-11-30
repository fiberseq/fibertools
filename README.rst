==========
fibertools
==========


.. image:: https://img.shields.io/pypi/v/fibertools.svg
        :target: https://pypi.python.org/pypi/fibertools

.. image:: https://img.shields.io/travis/mrvollger/fibertools.svg
        :target: https://travis-ci.com/mrvollger/fibertools

.. image:: https://readthedocs.org/projects/fibertools/badge/?version=latest
        :target: https://fibertools.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status


.. image:: https://pyup.io/repos/github/mrvollger/fibertools/shield.svg
     :target: https://pyup.io/repos/github/mrvollger/fibertools/
     :alt: Updates



A python package for handling Fiber-seq data.


* Free software: MIT license
* Documentation: https://fibertools.readthedocs.io.

Install
-------

::

    pip install fibertools

Features
--------

::

    usage: fibertools [-h] [-t THREADS] [-v] [-V] {bam2bed,add-m6a,add-nucleosomes,model,split,trackhub,bed2d4} ...

    positional arguments:
    {bam2bed,add-m6a,add-nucleosomes,model,split,trackhub,bed2d4}
                            Available subcommand for fibertools
        bam2bed             Extract m6a calls from bam and output bed12.
        add-m6a             Add m6A tag
        add-nucleosomes     Add Nucleosome and MSP tags
        model               Make MSP features
        split               Split a bed over many output files.
        trackhub            Make a trackhub from a bed file.
        bed2d4              Make a multi-track d4 file from a bed file.

    optional arguments:
    -h, --help            show this help message and exit
    -t THREADS, --threads THREADS
                            n threads to use (default: 1)
    -v, --verbose         increase logging verbosity (default: False)
    -V, --version         show program's version number and exit


Add nucleosomes and MSPs to a fibertools-rs m6A bam
---------------------------------------------------

Create the model ::

    fibertools add-nucleosomes --input input.bam > model.json

Add nucleosomes with the model::

    fibertools add-nucleosomes --input input.bam --model model.json > output.bam

Note that by default the input bam file is read from stdin and the output bam file is written to stdout.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
