# Copyright (c) 2016, Vienna University of Technology, Department of
# Geodesy and Geoinformation.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#   * Neither the name of the Vienna University of Technology,
#     Department of Geodesy and Geoinformation nor the
#     names of its contributors may be used to endorse or promote products
#     derived from this software without specific prior written permission.

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
Classes for reading and writing point data in NetCDF files according to
the Climate Forecast Metadata Conventions (http://cfconventions.org/).
"""

import numpy as np
import datetime
import netCDF4


class AccessItem(object):

    def __init__(self, obj, name, **kwargs):
        self.obj = obj
        self.name = name
        self.kwargs = kwargs

    def __getitem__(self, item):
        return self.obj.data.variables[self.name][item]

    def __setitem__(self, idx, value):

        if self.name in self.obj.data.variables.keys():
            var = self.obj.data.variables[self.name]
        else:
            fill_value = None

            if 'attr' in self.kwargs:
                if '_FillValue' in self.kwargs['attr']:
                    fill_value = self.kwargs['attr'].pop('_FillValue')

            if 'dtype' in self.kwargs:
                dtype = self.kwargs['dtype']
            else:
                dtype = value.dtype

            if 'zlib' in self.kwargs:
                zlib = self.kwargs['zlib']
            else:
                zlib = self.obj.zlib

            if 'complevel' in self.kwargs:
                complevel = self.kwargs['complevel']
            else:
                complevel = self.obj.complevel

            if 'chunksizes' in self.kwargs:
                chunksizes = self.kwargs['chunksizes']
            else:
                chunksizes = None

            dim = self.kwargs['dim']

            var = self.obj.data.createVariable(
                self.name, dtype, dim, fill_value=fill_value,
                zlib=zlib, complevel=complevel, chunksizes=chunksizes)

        var[idx] = value


class AccessMultipleItems(object):

    def __init__(self, obj, names, **kwargs):
        self.obj = obj
        self.names = names
        self.kwargs = kwargs

    def __getitem__(self, item):

        data = []
        for name in self.names:
            data.append(self.obj.data.variables[name][item])

        return data

    def __setitem__(self, idx, value):

        for name in self.names:

            if name in self.obj.data.variables.keys():
                var = self.obj.data.variables[name]
            else:
                fill_value = None

                if 'attr' in self.kwargs:
                    if '_FillValue' in self.kwargs['attr']:
                        fill_value = self.kwargs['attr'].pop('_FillValue')

                if 'dtype' in self.kwargs:
                    dtype = self.kwargs['dtype']
                else:
                    dtype = value.dtype

                if 'zlib' in self.kwargs:
                    zlib = self.kwargs['zlib']
                else:
                    zlib = self.obj.zlib

                if 'complevel' in self.kwargs:
                    complevel = self.kwargs['complevel']
                else:
                    complevel = self.obj.complevel

                if 'chunksizes' in self.kwargs:
                    chunksizes = self.kwargs['chunksizes']
                else:
                    chunksizes = None

                dim = self.kwargs['dim']

                var = self.obj.data.createVariable(
                    name, dtype, dim, fill_value=fill_value, zlib=zlib,
                    complevel=complevel, chunksizes=chunksizes)

            var[idx] = value[name]


class NcData(object):

    def __init__(self, filename, dims=None, file_format="NETCDF4", mode='r',
                 zlib=True, complevel=4):

        self.filename = filename

        self.gattr = {}
        self.gattr['id'] = os.path.split(self.filename)[1]

        s = "%Y-%m-%d %H:%M:%S"
        self.gattr['date_created'] = datetime.datetime.now().strftime(s)

        self.zlib = zlib
        self.complevel = complevel
        self.mode = mode

        if self.mode == "a" and not os.path.exists(self.filename):
            self.mode = "w"

        if self.mode == 'w':
            path = os.path.dirname(self.filename)
            if not os.path.exists(path):
                os.makedirs(path)

        self.data = netCDF4.Dataset(self.filename, self.mode,
                                    format=file_format)

        if self.mode == 'w':
            if dims is None:
                raise ValueError("Dimensions not defined.")
            else:
                self._create_dims(dims)

    def _create_dims(self, dims):
        """
        Create dimension for NetCDF file.

        Parameters
        ----------
        dims : dict
            NetCDF dimension.
        """
        for name, size in dims.iteritems():
            self.data.createDimension(name, size=size)

    def _setncatts(self):
        """
        Write global attributes to NetCDF file.
        """
        self.data.setncatts(self.gattr)

    def write(self, name, data, **kwargs):
        """
        Write data.

        Parameters
        ----------
        name : str
            Variable name.
        data :
            Variable data.
        dim : tuple
            Variable dimensions.
        """
        if self.mode in ['w', 'r+', 'a']:
            self.io(name, **kwargs)[:] = data

    def read(self, name, **kwargs):
        """
        reads variable from netCDF file

        Parameters
        ----------
        name : str
            Name of the variable.
        """
        if self.mode in ['r', 'r+']:
            return self.io(name, **kwargs)[:]

    def io(self, name, **kwargs):
        """
        Description.

        Parameters
        ----------
        name : str
            Name of the variable.

        Returns
        -------
        var : numpy.ndarray
            Data stored in variable.
        """
        return AccessItem(self, name, **kwargs)

    def flush(self):
        """
        Flush data.
        """
        if self.data is not None:
            if self.mode in ['w', 'r+']:
                self._setncatts()
                self.data.sync()

    def close(self):
        """
        Close file.
        """
        if self.data is not None:
            self.flush()
            self.data.close()
            self.data = None

    def __enter__(self):
        """
        Description.
        """
        return self

    def __exit__(self, value_type, value, traceback):
        """
        Description.
        """
        self.close()


class PointData(NcData):

    """
    PointData class description.
    """

    def __init__(self, filename, n_obs=None, obs_dim_name='obs',
                 loc_ids_name='location_id',
                 loc_descr_name='location_description',
                 time_units="days since 1900-01-01 00:00:00",
                 time_var='time', lat_var='lat', lon_var='lon', alt_var='alt',
                 unlim_chunksize=None, read_bulk=False, **kwargs):

        self.obs_dim_name = obs_dim_name

        self.dims = {}
        self.dims[self.obs_dim_name] = n_obs

        # location names
        self.loc_ids_name = loc_ids_name
        self.loc_descr_name = loc_descr_name

        # time, time units and location
        self.time_var = time_var
        self.time_units = time_units
        self.lat_var = lat_var
        self.lon_var = lon_var
        self.alt_var = alt_var

        self.unlim_chunksize = unlim_chunksize

        if unlim_chunksize is not None:
            self.unlim_chunksize = [unlim_chunksize]

        super(PointData, self).__init__(filename, dims=self.dims, **kwargs)

        if self.mode == 'w':
            self._init_location_variables()
            self.gattr['featureType'] = 'point'

    def _init_location_variables(self):
        """
        Initialize location information: longitude, latitude and altitude.
        """
        self.write({self.lon_var: None}, dim=self.obs_dim_name,
                   attr={'standard_name': 'longitude',
                         'long_name': 'location longitude',
                         'units': 'degrees_east',
                         'valid_range': (-180.0, 180.0)},
                   dtype=np.float32)

        self.write({self.lat_var: None}, dim=self.obs_dim_name,
                   attr={'standard_name': 'latitude',
                         'long_name': 'location latitude',
                         'units': 'degrees_north',
                         'valid_range': (-90.0, 90.0)},
                   dtype=np.float32)

        self.write({self.alt_var: None}, dim=self.obs_dim_name,
                   attr={'standard_name': 'height',
                         'long_name': 'vertical distance above the '
                         'surface', 'units': 'm', 'positive': 'up',
                         'axis': 'Z'},
                   dtype=np.float32)

        self.write({self.time_var: None}, dim=self.obs_dim_name,
                   attr={'standard_name': 'time'},
                   dtype=np.float64)

    def write(self, data, **kwargs):
        """
        Write.

        Parameters
        ----------
        data : dict
            Dictionary.
        """
        if self.mode in ['w', 'r+', 'a']:
            self.io(data.keys(), **kwargs)[:] = data

    def read(self, name, **kwargs):
        """
        reads variable from netCDF file

        Parameters
        ----------
        name : str or list of str
            Name of the variable.

        Returns
        -------
        var : numpy.ndarray
            Data stored in variable.
        """
        if self.mode in ['r', 'r+']:
            return self.io(name, **kwargs)[:]

    def io(self, name, **kwargs):
        """
        Description.

        Parameters
        ----------
        name : str or list of str
            Name of the variable.

        Returns
        -------
        var : numpy.ndarray
            Data stored in variable.
        """
        return AccessMultipleItems(self, name, **kwargs)


import os
import unittest
from tempfile import mkdtemp


class NcDataTest(unittest.TestCase):

    def setUp(self):
        self.fn = os.path.join(mkdtemp(), 'test.nc')

    def tearDown(self):
        os.remove(self.fn)

    def test_read_write(self):

        dims = {'obs': 10}

        with NcData(self.fn, dims, mode='w') as nc:
            var = np.arange(10)
            nc.write('myvar1', var, dim=('obs',))
            nc.write('myvar2', var, dim=('obs',))
            nc.io('myvar3', dim=('obs', ))[0:5] = np.arange(5)

        with NcData(self.fn) as nc:
            print nc.io('myvar1')[0:4]
            print nc.io('myvar3')[:]


class NcPointDataTest(unittest.TestCase):

    def setUp(self):
        self.fn = os.path.join('/home', 'shahn', 'test.nc')

    def tearDown(self):
        pass
        # os.remove(self.fn)

    def test_read_write(self):

        data = {"var1": np.arange(5)}
        with PointData(self.fn, 5, mode='w') as nc:
            nc.write(data)


if __name__ == "__main__":
    unittest.main()
