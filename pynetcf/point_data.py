# Copyright (c) 2017, Vienna University of Technology, Department of
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
Module for reading and writing point data in NetCDF format according to
the Climate Forecast Metadata Conventions (http://cfconventions.org/).
"""

import os

import numpy as np
import datetime
import netCDF4

from pygeobase.io_base import GriddedBase


class PointData(object):

    """
    PointData class for reading and writing netCDF files following the
    CF conventions for point data.

    Parameters
    ----------
    filename : str
        Filename of netCDF file. If already exiting then it will be opened
        as read only unless the append keyword is set.
    mode : str, optional
        access mode. default 'r'
        'r' means read-only; no data can be modified.
        'w' means write; a new file is created, an existing file with the
            same name is deleted.
        'a' and 'r+' mean append (in analogy with serial files); an existing
            file is opened for reading and writing.
        Appending s to modes w, r+ or a will enable unbuffered shared access
        to NETCDF3_CLASSIC or NETCDF3_64BIT formatted files. Unbuffered
        access may be useful even if you don't need shared access, since it
        may be faster for programs that don't access data sequentially.
        This option is ignored for NETCDF4 and NETCDF4_CLASSIC
        formatted files.
    zlib : boolean, optional
        If set netCDF compression will be used. Default True
    complevel : int, optional
        Compression level used from 1(low compression) to 9(high compression).
        Default: 4
    n_obs : int, optional
        Number of observations. If None, unlimited dimension will be used.
        Default: None
    obs_dim : str, optional
        Observation dimension name. Default: 'obs'
    add_dims : dict, optional
        Additional dimensions. Default: None
    loc_id_var : str, optional
        Location id variable name. Default: 'location id'
    time_units : str, optional
        Time unit.
    time_var : str, optional
        Time variable name. Default 'time'
    lat_var : str, optional
        Latitude variable name. Default 'lat'
    lon_var : str, optional
        Longitude variable name. Default: 'lon'
    alt_var : str, optional
        Altitude variable name. Default: 'alt'
    """

    def __init__(self, filename, mode='r', file_format='NETCDF4', zlib=True,
                 complevel=4, n_obs=None, obs_dim='obs', add_dims=None,
                 loc_id_var='location_id',
                 time_units="days since 1900-01-01 00:00:00",
                 time_var='time', lat_var='lat', lon_var='lon', alt_var='alt',
                 **kwargs):

        self.nc_finfo = {'filename': filename, 'mode': mode,
                         'format': file_format}

        initial_mode = mode

        if mode == 'a' and not os.path.exists(filename):
            initial_mode = 'w'

        if initial_mode == 'w':
            path = os.path.dirname(filename)
            if not os.path.exists(path):
                os.makedirs(path)
        self.compression_info = {'zlib': zlib,
                                 'complevel': complevel}

        try:
            self.nc = netCDF4.Dataset(filename, format=file_format,
                                      mode=initial_mode)
        except RuntimeError:
            raise IOError("File {} does not exist.".format(filename))

        loc_id_attr = {'long_name': 'location_id'}

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

        self.obs_dim = obs_dim

        if add_dims is not None:
            self.dim = add_dims.copy()
            self.dim.update({obs_dim: n_obs})
        else:
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

        self.builtin_vars = [self.var[key]['name'] for key in self.var]

        if initial_mode == 'w':

            s = "%Y-%m-%d %H:%M:%S"
            attr = {'id': os.path.split(self.nc_finfo['filename'])[1],
                    'date_created': datetime.datetime.now().strftime(s),
                    'featureType': 'point'}

            self.nc.setncatts(attr)
            self._create_dims(self.dim)
            self._init_loc_var()

        # find next free position, i.e. next empty loc_id
        self.loc_idx = 0
        if initial_mode in ['r+', 'a']:
            loc_id = self.nc.variables[self.var['loc_id']['name']]
            if self.nc.dimensions[obs_dim].isunlimited():
                self.loc_idx = loc_id.shape[0]
            else:
                self.loc_idx = np.where(loc_id[:].mask)[0][0]

    def __str__(self):
        """
        String representation of class instance.
        """
        if self.nc is not None:
            str = self.nc.__str__()
        else:
            str = 'NetCDF file closed.'

        return str

    def flush(self):
        """
        Flush data.
        """
        if self.nc is not None:
            if self.nc_finfo['mode'] in ['w', 'r+', 'a']:
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
        return self

    def __exit__(self, value_type, value, traceback):
        self.close()

    def _create_dims(self, dims):
        """
        Create dimensions in NetCDF file.

        Parameters
        ----------
        dims : dict
            NetCDF dimension.
        """
        for name, size in dims.items():
            self.nc.createDimension(name, size=size)

    def _init_loc_var(self):
        """
        Initialize location information (lon, lat, etc.).
        """
        for k, var in self.var.items():
            self.nc.createVariable(var['name'], var['dtype'],
                                   dimensions=var['dim'],
                                   **self.compression_info)
            self.nc.variables[var['name']].setncatts(var['attr'])

    def write(self, loc_id, data, lon=None, lat=None, alt=None, time=None,
              **kwargs):
        """
        Write data for specified location ids.

        Parameters
        ----------
        loc_id : numpy.ndarray
            Location id.
        data : dict of numpy.ndarray or numpy.recarray
            Dictionary containing variable names as keys and data as items.
        lon : numpy.ndarray, optional
            Longitude information. Default: None
        lat : numpy.ndarray, optional
            Latitude information. Default: None
        alt : numpy.ndarray, optional
            Altitude information. Default: None
        time : numpy.ndarray, optional
            Time information. Default: None
        """
        if self.nc_finfo['mode'] in ['w', 'r+', 'a']:

            num = np.array(loc_id).size
            idx = slice(self.loc_idx, self.loc_idx + num)

            # convert dict to recarray
            if isinstance(data, dict):

                # collect metadata info
                sub_md_list = [v.dtype.metadata for v in data.values()]

                # collect dtype info
                dtype_list = [(str(k), data[k].dtype.str,
                               data[k].shape) for k in data.keys()]

                # merge metadata info into common dict
                md_dict = {}
                for md in sub_md_list:
                    if md is not None and 'dims' in md:
                        md_dict.update(md['dims'])

                # convert dict to recarray
                metadata = {'dims': md_dict}
                dtype = np.dtype(dtype_list, metadata=metadata)
                data = np.core.records.fromarrays(data.values(), dtype=dtype)

            for var_data in data.dtype.names:
                if var_data not in self.nc.variables:
                    dtype = data[var_data].dtype
                    dimensions = (self.obs_dim,)

                    # check if custom metadata is included
                    if data.dtype.metadata is not None:
                        metadata = data.dtype.metadata
                        if 'dims' in metadata and var_data in metadata['dims']:
                            dimensions = metadata['dims'][var_data]

                    self.nc.createVariable(var_data, dtype,
                                           dimensions=dimensions,
                                           **self.compression_info)

                self.nc.variables[var_data][idx] = data[var_data]

            var_loc_id = self.var['loc_id']['name']
            self.nc.variables[var_loc_id][idx] = loc_id

            if lon is not None:
                var_lon = self.var['lon']['name']
                self.nc.variables[var_lon][idx] = lon

            if lat is not None:
                var_lat = self.var['lat']['name']
                self.nc.variables[var_lat][idx] = lat

            if alt is not None:
                var_alt = self.var['alt']['name']
                self.nc.variables[var_alt][idx] = alt

            if time is not None:
                var_time = self.var['time']['name']
                self.nc.variables[var_time][idx] = time

            self.loc_idx += num
        else:
            raise IOError("Write operations failed. "
                          "File not open for writing.")

    def read(self, loc_id):
        """
        Read variable from netCDF file for given location id.

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

        if self.nc_finfo['mode'] in ['r', 'r+', 'a']:
            loc_id_var = self.nc.variables[self.var['loc_id']['name']][:]
            pos = np.where(loc_id_var == loc_id)[0]

            if pos.size > 0:
                data = {}
                for var_name in self.nc.variables.keys():
                    read_data = self.nc.variables[var_name][pos]
                    if var_name not in self.builtin_vars:
                        read_data = np.squeeze(read_data)

                    data[var_name] = read_data

        else:
            raise IOError("Read operations failed. "
                          "File not open for reading.")

        return data

    def __getitem__(self, item):
        """
        Accessing netCDF variable.

        Parameters
        ----------
        item : str
            Variable name.

        Returns
        -------
        var : netcdf4.variable
            NetCDF variable.
        """
        return self.nc.variables[item]


class GriddedPointData(GriddedBase):

    """
    GriddedPointData class using GriddedBase class as parent and
    PointData as i/o class.
    """

    def __init__(self, *args, **kwargs):
        kwargs['ioclass'] = PointData
        if 'fn_format' not in kwargs:
            kwargs['fn_format'] = '{:04d}.nc'
        super(GriddedPointData, self).__init__(*args, **kwargs)

    def to_point_data(self, filename, **kwargs):
        """
        Re-write gridded point data into single file.

        Parameters
        ----------
        filename : str
            File name.
        """
        with PointData(filename, mode='w', **kwargs) as nc:
            for data, gp in self.iter_gp():
                nc.write(gp, data, lon=data['lon'], lat=data['lat'],
                         alt=data['alt'], time=data['time'])
