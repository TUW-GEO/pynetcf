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


class PointData(object):

    """
    PointData class description.
    """

    def __init__(self, filename, mode='r', file_format='NETCDF4', zlib=True,
                 complevel=4, n_obs=None, obs_dim='obs', loc_id_var='location_id',
                 time_units="days since 1900-01-01 00:00:00",
                 time_var='time', lat_var='lat', lon_var='lon', alt_var='alt',
                 unlim_chunksize=None, **kwargs):

        if mode == 'a' and not os.path.exists(filename):
            mode = 'w'

        if mode == 'w':
            path = os.path.dirname(filename)
            if not os.path.exists(path):
                os.makedirs(path)

        self.nc_finfo = {'filename': filename, 'mode': mode,
                         'format': file_format, 'zlib': zlib,
                         'complevel': 4, 'unlim_chunksize': unlim_chunksize}

        self.nc = netCDF4.Dataset(**self.nc_finfo)

        loc_id_attr = {'standard_name': 'location_id'}

        lon_attr = {'standard_name': 'longitude',
                    'long_name': 'location longitude',
                    'units': 'degrees_east',
                    'valid_range': (-180.0, 180.0)}

        lat_attr = {'standard_name': 'latitude',
                    'long_name': 'location latitude',
                    'units': 'degrees_north', 'valid_range': (-90.0, 90.0)}

        alt_attr = {'standard_name': 'height',
                    'long_name': 'vertical distance above the '
                    'surface', 'units': 'm', 'positive': 'up', 'axis': 'Z'}

        time_attr = {'standard_name': 'time'}

        self.dim = {obs_dim: n_obs}

        self.var = {'loc_id': {'name': loc_id_var, 'dim': obs_dim,
                               'attr': loc_id_attr, 'dtype': np.int32},
                    'lon': {'name': lon_var, 'dim': obs_dim,
                            'attr': lon_attr, 'dtype': np.float32},
                    'lat': {'name': lat_var, 'dim': obs_dim,
                            'attr': lat_attr, 'dtype': np.float32},
                    'alt': {'name': alt_var, 'dim': obs_dim,
                            'attr': alt_attr, 'dtype': np.float32},
                    'time': {'name': time_var, 'dim': obs_dim,
                             'unit': time_units, 'dtype': np.float64,
                             'attr': time_attr}}

        if self.nc_finfo['mode'] == 'w':

            s = "%Y-%m-%d %H:%M:%S"
            attr = {'id': os.path.split(self.nc_finfo['filename'])[1],
                    'date_created': datetime.datetime.now().strftime(s),
                    'featureType': 'point'}

            self.nc.setncatts(attr)
            self._create_dims(self.dim)
            self._init_loc_var()

      # find next free position, i.e. next empty loc_id
        self.loc_idx = 0
        if self.nc_finfo['mode'] in ['r+', 'a']:
            self.loc_idx = 0

    def __str__(self):
        if self.nc is not None:
            str = self.nc.__str__()
        else:
            str = "File not opened."

        return str

    def flush(self):
        """
        Flush data.
        """
        if self.nc is not None:
            if self.nc_finfo['mode'] in ['w', 'r+']:
                self.nc.sync()

    def close(self):
        """
        Close file.
        """
        if self.nc is not None:
            self.flush()
            self.nc.close()
            self.nc = None

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

    def _create_dims(self, dims):
        """
        Create dimension for NetCDF file.

        Parameters
        ----------
        dims : dict
            NetCDF dimension.
        """
        for name, size in dims.iteritems():
            self.nc.createDimension(name, size=size)

    def _init_loc_var(self):
        """
        Initialize location information.
        """
        for var in self.var.itervalues():
            self.nc.createVariable(var['name'], var['dtype'],
                                   dimensions=var['dim'])
            self.nc.variables[var['name']].setncatts(var['attr'])

    def write(self, loc_id, data, **kwargs):
        """
        Write.

        Parameters
        ----------
        loc_id : int
            Location id.
        data : dict
            Dictionary containing variable names as keys and data as items.
        """
        if self.nc_finfo['mode'] in ['w', 'r+', 'a']:

            kwargs['dim'] = self.dim
            var_names = list(set(data.keys()) |
                             set(self.nc.variables.keys()))

            for var_name in var_names:
                if var_name in data:
                    if var_name not in self.nc.variables:

                        try:
                            dtype = data[var_name].dtype
                        except AttributeError:
                            dtype = type(data[var_name])

                        self.nc.createVariable(var_name, dtype,
                                               dimensions=self.dim.keys())

                    self.nc.variables[var_name][self.loc_idx] = data[var_name]

            self.nc.variables[var_name][self.loc_idx] = loc_id
            self.loc_idx += 1
        else:
            raise IOError("Write operations failed. "
                          "File not open for writing.")

    def read(self, loc_id):
        """
        reads variable from netCDF file

        Parameters
        ----------
        loc_id : int
            Location id.

        Returns
        -------
        data : dict
            Dictionary containing variable names as a key and data as items.
        """
        data = None

        if self.nc_finfo['mode'] in ['r', 'r+']:
            loc_id_var = self.nc.variables[self.var['loc_id']['name']][:]
            pos = np.where(loc_id_var == loc_id)[0]

            if pos.size > 0:
                data = {}
                for var_name in self.nc.variables.keys():
                    data[var_name] = self.nc.variables[var_name][pos]

        else:
            raise IOError("Read operations failed. "
                          "File not open for reading.")

        return data


import os
import unittest
from tempfile import mkdtemp


class NcPointDataTest(unittest.TestCase):

    def setUp(self):
        self.fn = os.path.join('/home', 'shahn', 'test.nc')

    def tearDown(self):
        pass
        # os.remove(self.fn)

    def test_read_write(self):

        with PointData(self.fn,  mode='w') as nc:
            for loc_id, data in zip(range(5), range(5, 10)):
                if loc_id == 1:
                    nc.write(loc_id, {'var1': data, 'var2': data})
                elif loc_id == 3:
                    nc.write(loc_id, {'var1': data, 'var3': data})
                else:
                    nc.write(loc_id, {'var1': data})

        with PointData(self.fn) as nc:
            print(nc.read(4)['var1'])
            print(nc.read(10))
            print(nc)


if __name__ == "__main__":
    unittest.main()
