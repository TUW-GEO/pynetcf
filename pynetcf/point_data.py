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

        # for attr_name in attr:
        #     attr_value = attr[attr_name]
        #     self.obj.data.variables[name].setncattr(attr_name, attr_value)

        var[idx] = value


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
        name : string
            name of the variable
        """
        if self.mode in ['r', 'r+']:
            return self.io(name, **kwargs)[:]

    def io(self, name, **kwargs):
        """
        Description.
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

        super(PointData, self).__init__(filename, **kwargs)

        self.n_obs = n_obs

        # dimension names
        self.obs_dim_name = obs_dim_name

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

        # variable to track write operations
        self.write_operations = 0
        self.write_offset = None

        if self.mode == 'w':
            self._init_dimensions()
            self._init_location_variables()
            self._init_location_id_and_time()
            self.global_attr['featureType'] = 'point'

        self.read_bulk = read_bulk

        # if read bulk is activated the arrays will be read into the
        # local variables dict if it is not activated the data will be read
        # from the netCDF variables
        if not self.read_bulk:
            self.variables = self.dataset.variables
        else:
            self.variables = {}

    def _init_dimensions(self):
        """
        Initializes the dimensions.
        """
        self.create_dim(self.obs_dim_name, self.n_obs)

    def _init_location_variables(self):
        """
        Initialize location information: longitude, latitude and altitude.
        """
        self.write_var(self.lon_var, data=None, dim=self.obs_dim_name,
                       attr={'standard_name': 'longitude',
                             'long_name': 'location longitude',
                             'units': 'degrees_east',
                             'valid_range': (-180.0, 180.0)},
                       dtype=np.float32)

        self.write_var(self.lat_var, data=None, dim=self.obs_dim_name,
                       attr={'standard_name': 'latitude',
                             'long_name': 'location latitude',
                             'units': 'degrees_north',
                             'valid_range': (-90.0, 90.0)},
                       dtype=np.float32)

        self.write_var(self.alt_var, data=None, dim=self.obs_dim_name,
                       attr={'standard_name': 'height',
                             'long_name': 'vertical distance above the '
                             'surface',
                             'units': 'm',
                             'positive': 'up',
                             'axis': 'Z'},
                       dtype=np.float32)

    def _check_var(self, var):
        """
        Checks variable

        Parameters
        ----------
        var : int, string or numpy.ndarray
            Variable.

        Returns
        -------
        var : numpy.ndarray
            Variable.
        """
        if var is not None:
            if type(var) != np.ndarray:
                var = np.array([var])

        # netCDF library can not handle arrays of length 1 that contain only a
        # None value
        if var.size == 1 and var[0] is None:
            var = None

        return var

    def _add_observation(self, loc_id, time, lon, lat, alt=None, loc_descr=None):
        """
        Add a new observation to the data set.

        Parameters
        ----------
        loc_id : int or numpy.ndarray
            Location id of observation.
        time : int or numpy.ndarray
            Time of observation.
        lon : float or numpy.ndarray
            Longitudes of observation.
        lat : float or numpy.ndarray
            Longitudes of observation.
        alt : float or numpy.ndarray
            Altitude of observation.
        loc_descr : string or numpy.ndarray
            Location description.
        """
        loc_id = self._check_var(loc_id)
        time = self._check_var(time)
        lon = self._check_var(lon)
        lat = self._check_var(lat)
        alt = self._check_var(alt)

        loc_ids_new = None
        index = None

        if time is not None:
            self.dataset.variables[self.time_var][index] = time[loc_ids_new]

        if lon is not None:
            self.dataset.variables[self.lon_var][index] = lon[loc_ids_new]

        if lat is not None:
            self.dataset.variables[self.lat_var][index] = lat[loc_ids_new]

        if alt is not None:
            self.dataset.variables[self.alt_var][index] = alt[loc_ids_new]

        if loc_descr is not None:
            pass

    def write(self, loc_id, data, dates, loc_descr=None, time=None, lon=None,
              lat=None, alt=None, fill_values=None, attributes=None,
              dates_direct=False):
        """
        Write.

        Parameters
        ----------

        """
        try:
            idx = self._get_loc_id_index(loc_id)
        except IOError:
            idx = self._add_location(loc_id, lon, lat, alt, loc_descr)

        # find out if attributes is a dict to be used for all variables or if
        # there is a dictionary of attributes for each variable
        unique_attr = False
        if attributes is not None:
            if sorted(data.keys()) == sorted(attributes.keys()):
                unique_attr = True

        for key in data:
            if data[key].size != dates.size:
                raise IOError("Timestamps and dataset {:} "
                              "must have the same size".format(key))

        for key in data:

            internal_attributes = {'name': key,
                                   'coordinates': 'time lat lon alt'}

            if type(fill_values) == dict:
                internal_attributes['_FillValue'] = fill_values[key]

            if attributes is not None:
                if unique_attr:
                    variable_attributes = attributes[key]
                else:
                    variable_attributes = attributes

                internal_attributes.update(variable_attributes)

            if self.unlim_chunksize is None:
                chunksizes = None
            else:
                chunksizes = [self.n_loc, self.unlim_chunksize[0]]

            self.write_var(key, data=None, dim=(self.obs_dim_name),
                           attr=internal_attributes,
                           dtype=data[key].dtype, chunksizes=chunksizes)

            if self.write_offset is None:
                # find start of elements that are not yet filled with values
                _slice_new = slice(self.length_before_extend, None, None)
                masked = np.where(
                    self.dataset.variables[key][idx, _slice_new].mask)[0]
                # all indexes already filled
                if len(masked) == 0:
                    raise IOError("No free data slots available")
                else:
                    self.write_offset = np.min(
                        masked) + self.length_before_extend

            _slice = slice(self.write_offset, None, None)
            # has to be reshaped to 2 dimensions because it is written
            # into 2d variable otherwise netCDF library gets confused,
            # might be a bug in netCDF?
            self.dataset.variables[key][idx, _slice] = \
                data[key].reshape(1, data[key].size)

    def read(self, name):
        """
        Read.

        Parameters
        ----------

        Returns
        -------

        """
        return self.read_var(name)


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

        with NcData(self.fn, dims, mode='w') as data:
            var = np.arange(10)
            data.write('myvar1', var, dim=('obs',))
            data.write('myvar2', var, dim=('obs',))
            data.io('myvar3', dim=('obs', ))[0:5] = np.arange(5)

        with NcData(self.fn) as data:
            print data.io('myvar1')[0:4]
            print data.io('myvar3')[:]


if __name__ == "__main__":
    unittest.main()
