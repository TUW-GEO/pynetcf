# Copyright (c) 2020, TU Wien, Department of Geodesy and Geoinformation.
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
Base classes for reading and writing time series and images in NetCDF files
according to the Climate Forecast Metadata Conventions
(http://cfconventions.org/).
"""

import os
import time
import numpy as np
import netCDF4
import datetime

import xarray


class DatasetError(Exception):
    pass


class BufferedIndexedRaggedTs(object):
    """
    NetCDF file wrapper class that makes some things easier

    Parameters
    ----------
    filename : string
        filename of netCDF file. If already exiting then it will be opened
        as read only unless the append keyword is set. if the overwrite
        keyword is set then the file will be overwritten
    name : string, optional
        will be written as a global attribute if the file is a new file
    file_format : string, optional
        file format
    mode : string, optional
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
        Default True
        if set netCDF compression will be used
    complevel : int, optional
        Default 4
        compression level used from 1(low compression) to 9(high compression)
    autoscale : bool, optional
        If disabled data will not be automatically scaled when reading and
        writing
    automask : bool, optional
        If disabled data will not be masked during reading.
        This means Fill Values will be used instead of NaN.
    """

    def __init__(self, filename, name=None, file_format="NETCDF4",
                 mode='r', zlib=True, complevel=4,
                 autoscale=True, automask=True, n_loc=None,
                 loc_dim_name='locations', loc_ids_name='location_id', loc_descr_name='location_description',
                 time_units="days since 1900-01-01 00:00:00",
                 time_var='time', lat_var='lat', lon_var='lon', alt_var='alt',
                 ):

        self.dataset_name = name
        self.filename = filename
        self.file = None
        self.file_format = file_format
        self.buf_len = 0
        self.global_attr = {}
        self.global_attr['id'] = os.path.split(self.filename)[1]
        s = "%Y-%m-%d %H:%M:%S"
        self.global_attr['date_created'] = datetime.datetime.now().strftime(s)
        if self.dataset_name is not None:
            self.global_attr['dataset_name'] = self.dataset_name
        self.zlib = zlib
        self.complevel = complevel
        self.mode = mode
        self.autoscale = autoscale
        self.automask = automask
        self.n_loc = n_loc
        # location ids, to be read upon first reading operation
        self.loc_ids_var = None
        # dimension names
        self.loc_dim_name = loc_dim_name
        # location names
        self.loc_ids_name = loc_ids_name
        self.loc_descr_name = loc_descr_name

        # time, time units and location
        self.time_var = time_var
        self.time_units = time_units
        self.lat_var = lat_var
        self.lon_var = lon_var
        self.alt_var = alt_var


        self.dataset = xarray.Dataset()

    # TODO: if new, create dir, if not, write to file/append dataset to existing, clear dataset data for further use
    def write_to_disk(self):
        if self.mode == "a" and not os.path.exists(self.filename):
            self.mode = "w"
        if self.mode == 'w':
            self._create_file_dir()

        #except RuntimeError:
        #raise IOError("File {} does not exist".format(self.filename))

    def _create_file_dir(self):
        """
        Create directory for file to sit in.
        Avoid race condition if multiple instances are
        writing files into the same directory.
        """
        path = os.path.dirname(self.filename)
        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except OSError:
                time.sleep(1)
                self._create_file_dir()

    def _set_global_attr(self):
        """
        Write global attributes to Dataset.
        """
        self.dataset.assign_attrs(self.global_attr)
        self.global_attr = {}

    # this is kinda? coords in xarray
    def create_dim(self, name, n):
        """
        Create dimension for NetCDF file.
        if it does not yet exist

        Parameters
        ----------
        name : str
            Name of the NetCDF dimension.
        n : int
            Size of the dimension.
        """
        # TODO: where does size go? is this even coords in xarray?
        if name not in self.dataset.coords:
            self.dataset.coords[name] = n

    def write_var(self, name, data=None, dim=None, attr={}, dtype=None,
                  zlib=None, complevel=None, chunksizes=None, **kwargs):
        """
        Create or overwrite values in a NetCDF variable. The data will be
        written to disk once flush or close is called

        Parameters
        ----------
        name : str
            Name of the NetCDF variable.
        data : np.ndarray, optional
            Array containing the data.
            if not given then the variable will be left empty
        dim : tuple, optional
            A tuple containing the dimension names.
        attr : dict, optional
            A dictionary containing the variable attributes.
        dtype: data type, string or numpy.dtype, optional
            if not given data.dtype will be used
        zlib: boolean, optional
            explicit compression for this variable
            if not given then global attribute is used
        complevel: int, optional
            explicit compression level for this variable
            if not given then global attribute is used
        chunksizes : tuple, optional
            chunksizes can be used to manually specify the
            HDF5 chunksizes for each dimension of the variable.
        """

        fill_value = None
        if '_FillValue' in attr:
            fill_value = attr.pop('_FillValue')

        if dtype is None:
            dtype = data.dtype

        if zlib is None:
            zlib = self.zlib
        if complevel is None:
            complevel = self.complevel

        if name in self.dataset.data_vars.keys():
            var = self.dataset.data_vars[name]
        else:
            # TODO: fill_value, zlib, complevel, chunksizes, kwargs go missing
            """var = self.dataset.createVariable(name, dtype,
                                              dim, fill_value=fill_value,
                                              zlib=zlib, complevel=complevel,
                                              chunksizes=chunksizes, **kwargs)
            """
            self.dataset.data_vars[name] = xarray.Variable(dim, np.array(dtype))
            var = self.dataset.data_vars[name]


        for attr_name in attr.keys():
            attr_value = attr[attr_name]
            var.attr[attr_name] = attr_value
        #TODO: is this saved in parent dataset?
        #var.set_auto_scale(self.autoscale)
        if data is not None:
            var.data = data

    def append_var(self, name, data, **kwargs):
        """
        append data along unlimited dimension(s) of variable

        Parameters
        ----------
        name : string
            Name of variable to append to.
        data : numpy.array
            Numpy array of correct dimension.

        Raises
        ------
        IOError
            if appending to variable without unlimited dimension
        """
        if name in self.dataset.data_vars.keys():
            var = self.dataset.data_vars[name]
            dim_unlimited = []
            key = []
            for index, dim in enumerate(var.dimensions):
                unlimited = dim.isunlimited()
                dim_unlimited.append(unlimited)
                if not unlimited:
                    # if the dimension is not unlimited set the slice to :
                    key.append(slice(None, None, None))
                else:
                    # if unlimited set slice of this dimension to
                    # append meaning
                    # [var.shape[index]:]
                    key.append(slice(var.shape[index], None, None))

            dim_unlimited = np.array(dim_unlimited)
            nr_unlimited = np.where(dim_unlimited)[0].size
            key = tuple(key)
            # if there are unlimited dimensions we can do an append
            if nr_unlimited > 0:
                var.data[key] = data
            else:
                raise IOError(''.join(('Cannot append to variable that ',
                                       'has no unlimited dimension')))
        else:
            self.write_var(name, data, **kwargs)

    def read_var(self, name):
        """
        reads variable from netCDF file

        Parameters
        ----------
        name : string
            name of the variable
        """
        #TODO: do i want only data here or metadata too?
        if self.mode in ['r', 'r+']:
            if name in self.dataset.data_vars.keys():
                return self.dataset.data_vars[name][:]

    def write_ts(self, loc_id, data, dates, loc_descr=None, lon=None,
                 lat=None, alt=None, fill_values=None, attributes=None,
                 dates_direct=False):
        """
        write time series data, if not yet existing also add location to file

        Parameters
        ----------
        loc_id : int or numpy.ndarray
            location id, if it is an array the location ids have to match the
            data in the data dictionary and in the dates array. In this way data for more than
            one point can be written into the file at once.
        data : dict or numpy.recarray
            dictionary with variable names as keys and numpy.arrays as values
        dates: numpy.array
            array of datetime objects
        attributes : dict, optional
            dictionary of attributes that should be added to the netCDF
            variables. can also be a dict of dicts for each variable name as
            in the data dict.
        dates_direct : boolean
            if true the dates are already converted into floating
            point number of correct magnitude
        """
        if type(data) == np.ndarray:
            field_names = data.dtype.names
        else:
            field_names = data.keys()

        # we always want to work with arrays
        loc_id = np.atleast_1d(loc_id)
        if len(loc_id) == 1:
            loc_id = loc_id.repeat(dates.size)

        (loc_ids_uniq,
         loc_ids_uniq_index,
         loc_ids_uniq_lookup) = np.unique(loc_id,
                                          return_index=True,
                                          return_inverse=True)
        lon = np.atleast_1d(lon)
        lon_uniq = lon[loc_ids_uniq_index]
        lat = np.atleast_1d(lat)
        lat_uniq = lat[loc_ids_uniq_index]
        if alt is not None:
            alt = np.atleast_1d(alt)
            alt_uniq = alt[loc_ids_uniq_index]
        else:
            alt_uniq = None
        if loc_descr is not None:
            loc_descr = np.atleast_1d(loc_descr)
            loc_descr_uniq = loc_descr[loc_ids_uniq_index]
        else:
            loc_descr_uniq = None
        try:
            idx = self._get_loc_id_index(loc_id)
        except IOError:
            idx = self._add_location(loc_ids_uniq,
                                     lon_uniq,
                                     lat_uniq,
                                     alt_uniq,
                                     loc_descr_uniq)
            idx = self._get_loc_id_index(loc_id)

        # find out if attributes is a dict to be used for all variables or if
        # there is a dictionary of attributes for each variable
        unique_attr = False
        if attributes is not None:
            if sorted(data.keys()) == sorted(attributes.keys()):
                unique_attr = True

        for key in field_names:
            if data[key].size != dates.size:
                raise DatasetError("".join(("timestamps and dataset %s ",
                                            "must have the same size" % key)))

        # add number of new elements to index_var
        indices = np.atleast_1d(idx)
        self.append_var(self.obs_loc_lut, indices)

        index = np.arange(len(self.variables[self.obs_loc_lut]))[
                len(self.variables[self.obs_loc_lut]) - len(indices):]

        for key in field_names:

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

            # does nothing if variable exists already
            self.write_var(key, data=None, dim=self.obs_dim_name,
                           attr=internal_attributes,
                           dtype=data[key].dtype,
                           chunksizes=self.unlim_chunksize)

            self.dataset.variables[key][index] = data[key]

        if dates_direct:
            self.dataset.variables[self.time_var][index] = dates
        else:
            units = self.dataset.variables[self.time_var].units
            self.dataset.variables[self.time_var][index] = \
                netCDF4.date2num(dates, units=units, calendar='standard')

    def _add_location(self, loc_id, lon, lat, alt=None, loc_descr=None):
        """
        add a new location to the dataset

        Paramters
        ---------
        loc_id : int or numpy.array
            location id
        lon : float or numpy.array
            longitudes of location
        lat : float or numpy.array
            longitudes of location
        alt : float or numpy.array
            altitude of location
        loc_descr : string or numpy.array
            location description
        """

        if type(loc_id) != np.ndarray:
            loc_id = np.array([loc_id])

        if type(lon) != np.ndarray:
            lon = np.array([lon])

        # netCDF library can not handle arrays of length 1 that contain only a
        # None value
        if lon.size == 1 and lon[0] is None:
            lon = None

        if type(lat) != np.ndarray:
            lat = np.array([lat])

        # netCDF library can not handle arrays of length 1 that contain only a
        # None value
        if lat.size == 1 and lat[0] is None:
            lat = None

        if alt is not None:
            if type(alt) != np.ndarray:
                alt = np.array([alt])
            # netCDF library can not handle arrays of length 1 that contain
            # only a None value
            if alt.size == 1 and alt[0] is None:
                alt = None

        # remove location id's that are already in the file
        locations = np.ma.compressed(
            self.dataset.variables[self.loc_ids_name][:])

        loc_count = len(locations)
        if loc_count > 0:
            loc_ids_new = np.invert(np.in1d(loc_id, locations))
            if len(np.nonzero(loc_ids_new)[0]) == 0:
                # no new locations to add
                return None
        else:
            loc_ids_new = slice(None, None, None)

        # find free index position for limited location dimension
        idx = self._find_free_index_pos()

        index = np.arange(len(loc_id[loc_ids_new])) + idx

        self.dataset.variables[self.loc_ids_name][index] = loc_id[loc_ids_new]

        if lon is not None:
            self.dataset.variables[self.lon_var][index] = lon[loc_ids_new]

        if lat is not None:
            self.dataset.variables[self.lat_var][index] = lat[loc_ids_new]

        if alt is not None:
            self.dataset.variables[self.alt_var][index] = alt[loc_ids_new]

        if loc_descr is not None:

            if type(loc_descr) != np.ndarray:
                loc_descr = np.array(loc_descr)

            if len(index) == 1:
                index = int(index[0])
                loc_ids_new = 0
                self.dataset.variables[self.loc_descr_name][
                    index] = str(loc_descr)
            else:
                self.dataset.variables[self.loc_descr_name][
                    index] = loc_descr[loc_ids_new].astype(object)

        # update location ids variable after adding location
        self._read_loc_ids(force=True)

    def _get_loc_id_index(self, loc_id):
        """
        Gets index of location id in location ids variable.

        Parameters
        ----------
        loc_id : int or numpy.ndarray
            Location id.

        Returns
        -------
        loc_id_index : int
            Location id index.
        """
        self._read_loc_ids()
        loc_id = np.atleast_1d(loc_id)
        # check if the location ids are all actually in the location id
        # variable
        in1d = np.in1d(loc_id, self.loc_ids_var.data, assume_unique=True)
        if loc_id[in1d].size != loc_id.size:
            raise IOError("Location not yet defined")
        loc_ids_sorted = np.argsort(self.loc_ids_var.data)
        ypos = np.searchsorted(self.loc_ids_var[loc_ids_sorted], loc_id)
        try:
            loc_id_index = loc_ids_sorted[ypos]
            # loc_id_index = np.where(loc_id == self.loc_ids_var[:, None])[0]
        except IndexError:
            # Location not yet defined:
            raise IOError('Location not yet defined')

        if loc_id_index.size != loc_id.size:
            raise IOError('Index problem {:} elements '
                          ' found for {:} locations'.format(loc_id_index.size,
                                                            loc_id.size))

        if loc_id.size == 1:
            loc_id_index = loc_id_index[0]
        return loc_id_index

    def _find_free_index_pos(self):
        """
        If the index is not yet filled completely this function
        gets the id of the first free position.

        This function depends on the masked array being used if no
        data is yet in the file.

        Returns
        -------
        idx : int
            First free index position.

        Raises
        ------
        IOError
            If no free index is found.
        """
        self._read_loc_ids()

        masked = np.where(self.loc_ids_var.mask)[0]

        # all indexes already filled
        if len(masked) == 0:
            if self.dataset.dimensions[self.loc_dim_name].isunlimited():
                idx = self.loc_ids_var.size
            else:
                raise IOError('No free index available')
        else:
            idx = np.min(masked)

        return idx

    def _read_loc_ids(self, force=False):
        """
        Load location ids.
        """
        if self.loc_ids_var is None or force:
            loc_ids_var = self.dataset.data_vars[self.loc_ids_name][:]
            self.loc_ids_var = np.ma.masked_array(loc_ids_var)

    def add_global_attr(self, name, value):
        self.global_attr[name] = value

    def flush(self):
        if self.dataset is not None:
            if self.mode in ['w', 'r+']:
                self._set_global_attr()
                self.write_to_disk()

    def close(self):
        if self.dataset is not None:
            self.flush()
            # TODO: close file somehow?

    def __enter__(self):
        return self

    def __exit__(self, value_type, value, traceback):
        self.close()
