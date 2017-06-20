## v0.1.14 - 2017-06-20

- fix bug that read the wrong timeseries if a non existing location id was given.
- fix bug when reading not completely filled files in read_bulk mode in contigous ragged.

## v0.1.13 - 2017-05-02

- Catch RuntimeError and IOError to be compatible with older netCDF4 versions.
- Fix compression of variables in point_data.

## v0.1.12 - 2017-04-05

- IndexedRaggedTs are now compatible with numpy record arrays and dictionaries.

## v0.1.11 - 2017-04-03

- IndexedRaggedTs.write_ts can now write data for multiple grid points at once.
- Add interface to write a complete cell to GriddedNcIndexedRaggedTs

## v0.1.10 - 2017-01-19

- No changes in functionality
- Fix setup.py for correct installation

## v0.1.9 - 2017-01-19
- Fix n_loc bug
- Add recarray for point data
- Excluding pandas==0.19.0

## v0.1.8 - 2016-07-18
- Deprecate pynetcf.time_series.GriddedTs please use
  pynetcf.time_series.GriddedNcTs in the future. Be aware that the __init__
  arguments have changed slightly to path, grid, ioclass.

## v0.1.7 - 2016-04-12
- Add support of read/write netCDF point data following CF conventions
- Add support for disabling automatic masking during reading. Useful if the data
  has fill values but needs to be scaled to a datatype that does not support NaN
  values.

## v0.1.6 - 2016-02-03
- Add support for disabling automatic scaling in base netCDF4 library.
- Add support for dtype conversion before scaling and offset.

## v0.1.5 - 2016-01-27
- Add classes for gridded datasets based on pygeobase
- improve test coverage
- make compatible with newest netCDF4 releases
- support read_bulk keyword for all dataset types

## v0.1.4 - 2015-11-04
- fix open/closing of netCDF file

## v0.1.2 - 2015-08-04
- fixed issue #9

## v0.1.3 - 2015-08-10
- fixed issue #10

## v0.1.1 - 2015-02-19
- fixed issue #4

## v0.1 - 2015-02-18
- moved netcdf classes out of rs data readers



