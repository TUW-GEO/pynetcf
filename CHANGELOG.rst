=========
Changelog
=========

Version 0.5.0
=============

- Add test for writing strings to netcdf
- Zlib compress always off for non-numeric data

Version 0.4.0
=============

- Replace read_ts, write_ts with read, write
- Replace read_all_ts with read_all
- Replace write_ts_all_loc with write_all
- Pep8 fix

Version 0.3.0
=============

- Code refactor of time series classes
- Add destructor closing files
- Minor documentation update
- Fix warnings in tests
- Update copyright date

Version 0.2.2
=============

- Change numpy.num2date behavior to return datetime.datetime

Version 0.2.1
=============

- Force date type conversion to numpy.datetime64
- Update copyright date

Version 0.2.0
=============

- Package no longer Python 2.7 compatible
- Update pyscaffold v3.2.3
- Remove build_script and changes.md
- Update environment.yml
- Package name changed from pynetCF to pynetcf
- Updating travis settings

Version 0.1.19
==============

- Update readme.
- Remove unnecessary dimensions while reading point data.
- Pin squinx version to fix rtd
- Update netcdf4 requirement to v1.4.2

Version 0.1.18
==============

- Update installation to pyscaffold 2.5.x to fix https://github.com/blue-yonder/pyscaffold/issues/148
- Restrict netcdf4 package to versions <=1.2.8 because of https://github.com/Unidata/netcdf4-python/issues/784

Version 0.1.17
==============

- Allow writing and reading of PointData in append mode.
- Set default filenames for GriddedPointData to include .nc ending.

Version 0.1.16
==============

- Translate RuntimeError of older versions of the netCDF4 library to IOError.
- Avoid race conditions when creating directories for new files.

Version 0.1.15
==============

- Fix bug that lost the datatype during writing of timeseries with
  pandas > 0.17.1
- Fix Python 3 compability.

Version 0.1.14
==============

- fix bug that read the wrong timeseries if a non existing location id was
  given.
- fix bug when reading not completely filled files in read_bulk mode in
  contigous ragged.

Version 0.1.13
==============

- Catch RuntimeError and IOError to be compatible with older netCDF4 versions.
- Fix compression of variables in point_data.

Version 0.1.12
==============

- IndexedRaggedTs are now compatible with numpy record arrays and dictionaries.

Version 0.1.11
==============

- IndexedRaggedTs.write_ts can now write data for multiple grid points at once.
- Add interface to write a complete cell to GriddedNcIndexedRaggedTs

Version 0.1.10
==============

- No changes in functionality
- Fix setup.py for correct installation

Version 0.1.9
=============

- Fix n_loc bug
- Add recarray for point data
- Excluding pandas==0.19.0

Version 0.1.8
=============

- Deprecate pynetcf.time_series.GriddedTs please use
  pynetcf.time_series.GriddedNcTs in the future. Be aware that the __init__
  arguments have changed slightly to path, grid, ioclass.

Version 0.1.7
=============

- Add support of read/write netCDF point data following CF conventions
- Add support for disabling automatic masking during reading. Useful if the data
  has fill values but needs to be scaled to a datatype that does not support NaN
  values.

Version 0.1.6
=============

- Add support for disabling automatic scaling in base netCDF4 library.
- Add support for dtype conversion before scaling and offset.

Version 0.1.5
=============

- Add classes for gridded datasets based on pygeobase
- improve test coverage
- make compatible with newest netCDF4 releases
- support read_bulk keyword for all dataset types

Version 0.1.4
=============

- fix open/closing of netCDF file

Version 0.1.2
=============

- fixed issue #9

Version 0.1.3
=============

- fixed issue #10

Version 0.1.1
=============

- fixed issue #4

Version 0.1
===========

- moved netcdf classes out of rs data readers
