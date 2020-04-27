# Copyright (c) 2017, Vienna University of Technology,
# Department of Geodesy and Geoinformation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the Vienna University of Technology,
#      Department of Geodesy and Geoinformation nor the
#      names of its contributors may be used to endorse or promote products
#      derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY,
# DEPARTMENT OF GEODESY AND GEOINFORMATION BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

"""

import numpy as np
import netCDF4
import datetime
import pandas as pd

from pynetcf.time_series import OrthoMultiTs
from pynetcf.base import Dataset
import pygeogrids.grids as grids


class ArrayStack(OrthoMultiTs):

    """
    Class for writing stacks of arrays (1D) into netCDF.
    Array stacks are basically orthogonal multidimensional
    array representation netCDF files.
    """

    def __init__(self, filename, grid=None, times=None,
                 mode='r', name=''):
        self.grid = grid
        self.filename = filename
        self.times = times
        self.variables = []
        self.time_var = 'time'
        self.time_units = "days since 1900-01-01"
        self.time_chunksize = 1
        self.lon_chunksize = 1

        if mode in ['a', 'r']:
            super(ArrayStack, self).__init__(
                filename, name=name, mode=mode, read_dates=False)
            self._load_grid()
            self._load_times()

        if mode == 'w':
            if grid is None:
                raise IOError("grid needs to be defined")

            super(ArrayStack, self).__init__(
                filename, n_loc=len(self.grid.activegpis),
                name=name, mode=mode, read_dates=False)

            self.dataset.variables[self.lon_var][:] = self.grid.activearrlon
            self.dataset.variables[self.lat_var][:] = self.grid.activearrlat
            self.dataset.variables[self.loc_ids_name][:] = self.grid.activegpis

        self.lat_chunksize = len(self.grid.activegpis)

    def _load_grid(self):
        lons = self.dataset.variables[self.lon_var][:]
        lats = self.dataset.variables[self.lat_var][:]
        self.grid = grids.BasicGrid(lons, lats)

    def _load_times(self):
        self.times = netCDF4.num2date(self.dataset.variables['time'][:],
                                      self.time_units)

    def write_ts(self, gpi, data):
        """
        write a time series into the imagestack
        at the given gpi

        Parameters
        ----------
        self: type
            description
        gpi: int or numpy.array
            grid point indices to write to
        data: dictionary
            dictionary of int or numpy.array for each variable
            that should be written
            shape must be (len(gpi), len(times))
        """
        gpi = np.atleast_1d(gpi)

        for i, gp in enumerate(gpi):
            for var in data:
                super(ArrayStack, self).write_ts(
                    gp, {var: np.atleast_1d(np.atleast_2d(data[var])[i, :])},
                    np.array(self.times))

    def __setitem__(self, gpi, data):
        """
        write a time series into the imagestack
        at the given gpi

        Parameters
        ----------
        self: type
            description
        gpi: int or numpy.array
            grid point indices to write to
        data: dictionary
            dictionary of int or numpy.array for each variable
            that should be written
            shape must be (len(gpi), len(times))
        """
        self.write_ts(gpi, data)

    def __getitem__(self, key):

        if type(key) == datetime.datetime:
            index = netCDF4.date2index(
                key, self.dataset.variables[self.time_var])
            data = {}
            for var in self._get_all_ts_variables():
                data[var] = self.dataset.variables[var][:, index]
            return data
        else:
            gpi = np.atleast_1d(key)
            for i, gp in enumerate(gpi):
                data = super(ArrayStack, self).read_all_ts(gp)

            return pd.DataFrame(data, index=self.times)


class ImageStack(Dataset):

    """
    Class for writing stacks of 2D images into netCDF.
    """

    def __init__(self, filename, grid=None, times=None,
                 mode='r', name=''):
        self.grid = grid
        self.filename = filename
        self.times = times
        self.variables = []
        self.time_var = 'time'
        self.time_units = "days since 1900-01-01"
        self.time_chunksize = 1
        self.lon_chunksize = 1
        self.lat_chunksize = self.grid.lat2d.shape[1]
        super(ImageStack, self).__init__(filename, name=name, mode=mode)

        if self.mode == 'w':
            self._init_dimensions()
            self._init_time()
            self._init_location_variables()
        elif self.mode in ['a', 'r']:
            self._load_grid()
            self._load_variables()

    def _init_dimensions(self):
        self.create_dim('lon', self.grid.lon2d.shape[0])
        self.create_dim('lat', self.grid.lat2d.shape[1])
        self.create_dim('time', len(self.times))

    def _load_grid(self):
        lons = self.dataset.variables['lon'][:]
        lats = self.dataset.variables['lat'][:]
        self.grid = grids.gridfromdims(lons, lats)

    def _load_variables(self):
        for var in self.dataset.variables:
            if self.dataset.variables[var].dimensions == ('time', 'lat', 'lon'):
                self.variables.append(var)

    def _load_times(self):
        self.times = netCDF4.num2date(self.dataset.variables['time'][:],
                                      self.time_units)

    def _init_time(self):
        """
        initialize the dimensions and variables that are the basis of
        the format
        """
        # initialize time variable
        time_data = netCDF4.date2num(self.times, self.time_units)
        self.write_var(self.time_var, data=time_data, dim='time',
                       attr={'standard_name': 'time',
                             'long_name': 'time of measurement',
                             'units': self.time_units},
                       dtype=np.double,
                       chunksizes=[self.time_chunksize])

    def _init_location_variables(self):
        # write station information, longitude, latitude and altitude
        self.write_var('lon', data=self.grid.lon2d[:, 0], dim='lon',
                       attr={'standard_name': 'longitude',
                             'long_name': 'location longitude',
                             'units': 'degrees_east',
                             'valid_range': (-180.0, 180.0)},
                       dtype=np.float)
        self.write_var('lat', data=self.grid.lat2d[0, :], dim='lat',
                       attr={'standard_name': 'latitude',
                             'long_name': 'location latitude',
                             'units': 'degrees_north',
                             'valid_range': (-90.0, 90.0)},
                       dtype=np.float)

    def init_variable(self, var):
        self.write_var(var, data=None, dim=('time', 'lat', 'lon'),
                       dtype=np.float,
                       attr={'_FillValue': -9999.})

    def write_ts(self, gpi, data):
        """
        write a time series into the imagestack
        at the given gpi

        Parameters
        ----------
        self: type
            description
        gpi: int or numpy.array
            grid point indices to write to
        data: dictionary
            dictionary of int or numpy.array for each variable
            that should be written
            shape must be (len(gpi), len(times))
        """
        gpi = np.atleast_1d(gpi)

        for i, gp in enumerate(gpi):
            row, column = self.grid.gpi2rowcol(gp)
            for var in data:
                if var not in self.variables:
                    self.variables.append(var)
                    self.init_variable(var)
                self.dataset.variables[var][
                    :, row, column] = np.atleast_2d(data[var])[i, :]

    def __setitem__(self, gpi, data):
        """
        write a time series into the imagestack
        at the given gpi

        Parameters
        ----------
        self: type
            description
        gpi: int or numpy.array
            grid point indices to write to
        data: dictionary
            dictionary of int or numpy.array for each variable
            that should be written
            shape must be (len(gpi), len(times))
        """
        self.write_ts(gpi, data)

    def __getitem__(self, key):

        gpi = np.atleast_1d(key)
        data = {}
        for i, gp in enumerate(gpi):
            row, column = self.grid.gpi2rowcol(gp)
            for var in self.variables:
                data[var] = self.dataset.variables[var][
                    :, row, column]

        return pd.DataFrame(data, index=self.times)
