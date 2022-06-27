.. image:: https://github.com/TUW-GEO/pynetcf/workflows/ubuntu/badge.svg
   :target: https://github.com/TUW-GEO/pynetcf/actions/workflows/ubuntu.yml

.. image:: https://github.com/TUW-GEO/pynetcf/workflows/windows/badge.svg
   :target: https://github.com/TUW-GEO/pynetcf/actions/workflows/windows.yml

.. image:: https://coveralls.io/repos/github/TUW-GEO/pynetcf/badge.svg?branch=master
   :target: https://coveralls.io/github/TUW-GEO/pynetcf?branch=master

.. image:: https://badge.fury.io/py/pynetcf.svg
    :target: https://badge.fury.io/py/pynetcf

.. image:: https://readthedocs.org/projects/pynetcf/badge/?version=latest
   :target: http://pynetcf.readthedocs.org/

|

=======
pynetcf
=======

Basic python classes that map to netCDF files on disk written according to the `Climate and Forecast metadata conventions`_

.. _Climate and Forecast metadata conventions: (http://cfconventions.org/Data/cf-conventions/cf-conventions-1.6/build/cf-conventions.html)

This is a first draft which has a lot of room for improvements, this is especially true for the time series based representations.

Citation
========

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.846767.svg
   :target: https://doi.org/10.5281/zenodo.846767

If you use the software in a publication then please cite it using the Zenodo DOI.
Be aware that this badge links to the latest package version.

Please select your specific version at https://doi.org/10.5281/zenodo.846767 to get the DOI of that version.
You should normally always use the DOI for the specific version of your record in citations.
This is to ensure that other researchers can access the exact research artefact you used for reproducibility.

You can find additional information regarding DOI versioning at http://help.zenodo.org/#versioning

Installation
============

This package should be installable through pip:

.. code::

    pip install pynetcf

Contribute
==========

We are happy if you want to contribute. Please raise an issue explaining what
is missing or if you find a bug. We will also gladly accept pull requests
against our master branch for new features or bug fixes.

Development setup
-----------------

For Development we also recommend a ``conda`` environment. You can create one
including test dependencies and debugger by running
``conda env create -f environment.yml``. This will create a new ``pynetcf``
environment which you can activate by using ``source activate pynetcf``.

Guidelines
----------

If you want to contribute please follow these steps:

- Fork the pynetcf repository to your account
- make a new feature branch from the pynetcf master branch
- Add your feature
- Please include tests for your contributions in one of the test directories.
  We use py.test so a simple function called test_my_feature is enough
- submit a pull request to our master branch

.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.2.3. For details and usage
information on PyScaffold see https://pyscaffold.org/.
