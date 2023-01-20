# Copyright (c) 2023, TU Wien, Department of Geodesy and Geoinformation.
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
Abstract class providing an interface for reading and writing time series
in NetCDF files using Climate Forecast Metadata Conventions
(http://cfconventions.org/).
"""

import os
import warnings
from abc import ABC, abstractmethod

import netCDF4
import numpy as np
import pandas as pd
from pygeobase.io_base import GriddedTsBase
from pygeogrids import CellGrid

from pynetcf.base import Dataset
from pynetcf.base import DatasetError


class DatasetTs(Dataset, ABC):
    """
    Abstract class to store common methods for NetCDF time series such as
    OrthoMulti-, ContiguousRaggedArray- and IndexedRaggedArray-representation.
    Implemented according to the NetCDF CF-conventions 1.6.

    Parameters
    ----------
    filename : string
        filename of netCDF file. If already exiting then it will be opened
        as read only unless the append keyword is set. if the overwrite
        keyword is set then the file will be overwritten
    n_loc : int, optional
        number of locations that this netCDF file contains time series for
        only required for new file
    loc_dim_name : string, optional
        name of the location dimension
    obs_dim_name : string, optional
        name of the observations dimension
    loc_ids_name : string, optional
        name of variable that has the location id's stored
    loc_descr_name : string, optional
        name of variable that has additional location information
        stored
    time_units : string, optional
        units the time axis is given in.
        Default: "days since 1900-01-01 00:00:00"
    time_var : string, optional
        name of time variable
        Default: time
    lat_var : string, optional
        name of latitude variable
        Default: lat
    lon_var : string, optional
        name of longitude variable
        Default: lon
    alt_var : string, optional
        name of altitude variable
        Default: alt
    unlim_chunksize : int, optional
        chunksize to use along unlimited dimensions, other chunksizes
        will be calculated by the netCDF library
    read_bulk : boolean, optional
        if set to True the data of all locations is read into memory,
        and subsequent calls to read_ts read from the cache and not from disk
        this makes reading complete files faster#
    read_dates : boolean, optional
        if false dates will not be read automatically but only on specific
        request useable for bulk reading because currently the netCDF
        num2date routine is very slow for big datasets
    """

    def __init__(self,
                 filename,
                 n_loc=None,
                 loc_dim_name='locations',
                 obs_dim_name='time',
                 loc_ids_name='location_id',
                 loc_descr_name='location_description',
                 time_units="days since 1900-01-01 00:00:00",
                 time_var='time',
                 lat_var='lat',
                 lon_var='lon',
                 alt_var='alt',
                 unlim_chunksize=None,
                 read_bulk=False,
                 read_dates=True,
                 **kwargs):

        super(DatasetTs, self).__init__(filename, **kwargs)

        self.n_loc = n_loc

        # dimension names
        self.obs_dim_name = obs_dim_name
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

        self.unlim_chunksize = unlim_chunksize

        if unlim_chunksize is not None:
            self.unlim_chunksize = [unlim_chunksize]

        self.write_offset = None

        # variable which lists the variables that should not be
        # considered time series even if they have the correct dimension
        self.not_timeseries = [self.time_var]

        # location ids, to be read upon first reading operation
        self.loc_ids_var = None

        # date variables, for OrthogonalMulitTs it can be stored
        # since it is the same for all variables in a file
        self.constant_dates = True
        self.dates = None
        self.read_dates_auto = read_dates

        if self.mode == 'r':
            self.read_bulk = read_bulk
        else:
            self.read_bulk = False

        # cache location id during reading
        self.prev_loc_id = None

        # if read bulk is activated the arrays will  be read into the
        # local variables dict if it is not activated the data will be read
        # from the netCDF variables
        if not self.read_bulk:
            self.variables = self.dataset.variables
        else:
            self.variables = {}

    @abstractmethod
    def _init_dimensions(self):
        """
        Initializes the dimensions.
        """
        pass

    @abstractmethod
    def _init_lookup(self):
        """
        Initializes variables for the lookup between locations and entries in
        the time series.
        """
        pass

    def _init_location_variables(self):
        """
        Initialize location information: longitude, latitude and altitude.
        """
        self.write_var(self.lon_var,
                       data=None,
                       dim=self.loc_dim_name,
                       attr={
                           'standard_name': 'longitude',
                           'long_name': 'location longitude',
                           'units': 'degrees_east',
                           'valid_range': (-180.0, 180.0)
                       },
                       dtype=np.float32)

        self.write_var(self.lat_var,
                       data=None,
                       dim=self.loc_dim_name,
                       attr={
                           'standard_name': 'latitude',
                           'long_name': 'location latitude',
                           'units': 'degrees_north',
                           'valid_range': (-90.0, 90.0)
                       },
                       dtype=np.float32)

        self.write_var(self.alt_var,
                       data=None,
                       dim=self.loc_dim_name,
                       attr={
                           'standard_name': 'height',
                           'long_name': 'vertical distance above the '
                           'surface',
                           'units': 'm',
                           'positive': 'up',
                           'axis': 'Z'
                       },
                       dtype=np.float32)

    def _init_location_id_and_time(self):
        """
        Initialize the dimensions and variables that are the basis of
        the format.
        """
        # make variable that contains the location id
        self.write_var(self.loc_ids_name,
                       data=None,
                       dim=self.loc_dim_name,
                       dtype=np.int64)

        self.write_var(self.loc_descr_name,
                       data=None,
                       dim=self.loc_dim_name,
                       dtype=str)

        # initialize time variable
        self.write_var(self.time_var,
                       data=None,
                       dim=self.obs_dim_name,
                       attr={
                           'standard_name': 'time',
                           'long_name': 'time of measurement',
                           'units': self.time_units
                       },
                       dtype=np.float64,
                       chunksizes=self.unlim_chunksize)

    def _read_loc_ids(self, force=False):
        """
        Load location ids.
        """
        if self.loc_ids_var is None or force == True:
            loc_ids_var = self.dataset.variables[self.loc_ids_name][:]
            self.loc_ids_var = np.ma.masked_array(loc_ids_var)

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

        masked = np.where(np.atleast_1d(self.loc_ids_var.mask))[0]

        # all indexes already filled
        if len(masked) == 0:
            if self.dataset.dimensions[self.loc_dim_name].isunlimited():
                idx = self.loc_ids_var.size
            else:
                raise IOError('No free index available')
        else:
            idx = np.min(masked)

        return idx

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
                          ' found for {:} locations'.format(
                              loc_id_index.size, loc_id.size))

        if loc_id.size == 1:
            loc_id_index = loc_id_index[0]
        return loc_id_index

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
            # onla a None value
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
                self.dataset.variables[self.loc_descr_name][index] = str(
                    loc_descr)
            else:
                self.dataset.variables[self.loc_descr_name][index] = loc_descr[
                    loc_ids_new].astype(object)

        # update location ids variable after adding location
        self._read_loc_ids(force=True)

        return idx

    def _get_all_ts_variables(self):
        """
        Gets all variable names that have the self.obs_dim_name as only
        dimension indicating that they are time series observations. This
        does not include the self.time_var variable.

        Returns
        -------
        variables : list
            List of variable names.
        """
        ts_var = []

        for variable_name in self.dataset.variables:
            if variable_name not in self.not_timeseries:
                if self.obs_dim_name in \
                        self.dataset.variables[variable_name].dimensions:
                    ts_var.append(variable_name)

        return ts_var

    @abstractmethod
    def _get_index_of_ts(self, loc_id):
        """
        Get the index of a time series.

        Parameters
        ----------
        loc_id : int
            Location id.

        Returns
        -------
        loc_id_index : int
            Location id index.
        """
        pass

    @abstractmethod
    def _get_loc_ix_from_obs_ix(self, obs_ix):
        """
        Get location index from observation index. In case of OrthoMultiTs
        all measurements are taken at the same time and therefore all
        location id's are affected.

        Parameters
        ----------
        obs_ix : int
            Observation index.

        Returns
        -------
        loc_ix : int
            Location index.
        """
        pass

    def read_time(self, loc_id):
        """
        Read the time stamps for the given location id
        in this case the location id is irrelevant since they
        all have the same timestamps
        """
        return self.dataset.variables[self.time_var][:]

    def read_dates(self, loc_id):
        """
        Read time stamps and convert them.
        """
        self.dates = netCDF4.num2date(
            self.read_time(loc_id),
            units=self.dataset.variables[self.time_var].units,
            calendar='standard',
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True)

        return self.dates.astype('datetime64[ns]')

    def _read_var_ts(self, loc_id, var):
        """
        Read a time series of a variable at a given location id

        Parameters
        ----------
        loc_id : int
            id of location, can be a grid point id or some other id
        var : string
            name of variable to read
        """
        if self.prev_loc_id != loc_id:
            index = self._get_index_of_ts(loc_id)
            self.prev_loc_index = index
        else:
            index = self.prev_loc_index

        self.prev_loc_id = loc_id

        if self.read_bulk:
            if var not in self.variables.keys():
                self.variables[var] = self.dataset.variables[var][:]

        return self.variables[var][index]

    def read_ts(self, variables, loc_id, dates_direct=False):
        """
        reads time series of variables

        Parameters
        ----------
        variables : list or string
        loc_id : int
            location_id
        dates_direct : boolean, optional
            if True the dates are read directly from the netCDF file
            without conversion to datetime
        """
        if type(variables) != list:
            variables = [variables]

        ts = {}
        for variable in variables:
            data = self._read_var_ts(loc_id, variable)
            ts[variable] = data

        if not dates_direct:
            # only read dates if they should be read automatically
            if self.read_dates_auto:
                # only read dates if they have not been read
                # or if they are different for each location id which is
                # the case if self.constant_dates is set to False
                if self.dates is None:
                    self.read_dates(loc_id)
                if not self.constant_dates:
                    self.read_dates(loc_id)
            ts['time'] = self.dates
        else:
            if self.read_dates_auto:
                # only read dates if they have not been read
                # or if they are different for each location id which is
                # the case if self.constant_dates is set to False
                ts['time'] = self.read_time(loc_id)

        return ts

    def read_all_ts(self, loc_id, dates_direct=False):
        """
        read a time series of all time series variables at a given location id

        Parameters
        ----------
        loc_id : int
            id of location, can be a grid point id or some other id
        dates_direct : boolean, optional
            if True the dates are read directly from the netCDF file
            without conversion to datetime

        Returns
        -------
        time_series : dict
            keys of var and time with numpy.arrays as values
        """
        ts = self.read_ts(self._get_all_ts_variables(),
                          loc_id,
                          dates_direct=dates_direct)

        return ts

    def extend_time(self, dates, direct=False):
        """
        Extend the time dimension and variable by the given dates

        Parameters
        ----------
        dates : numpy.array of datetime objects or floats
            Timestamps.
        direct : boolean
            if true the dates are already converted into floating
            point number of correct magnitude
        """
        if direct:
            self.append_var(self.time_var, dates)
        else:
            units = self.dataset.variables[self.time_var].units
            self.append_var(
                self.time_var,
                netCDF4.date2num(dates, units=units, calendar='standard'))

    def get_time_variable_overlap(self, dates):
        """
        Figure out if a new date array has a overlap with the already
        existing time variable.

        Return the index of the existing time variable where the new dates
        should be located.

        At the moment this only handles cases where all dates are new or none
        are new.

        Parameters
        ----------
        dates : list
            List of datetime objects

        Returns
        -------
        indexes : numpy.ndarray
            Array of indexes that overlap
        """
        timevar = self.dataset.variables[self.time_var]
        if timevar.size == 0:
            indexes = np.array([0])
        else:
            try:
                indexes = netCDF4.date2index(dates, timevar)
            except ValueError:
                indexes = np.array([timevar.size])

        return indexes

    @abstractmethod
    def write_ts(self,
                 loc_id,
                 data,
                 dates,
                 loc_descr=None,
                 lon=None,
                 lat=None,
                 alt=None,
                 fill_values=None,
                 attributes=None,
                 dates_direct=False):
        """
        Write time series data, if not yet existing also add location to file
        for this data format it is assumed that in each write/append cycle
        the same amount of data is added.

        Parameters
        ----------
        loc_id : int
            Location id.
        data : dict
            Dictionary with variable names as keys and
            numpy.ndarrays as values.
        dates: numpy.ndarray
            Array of datetime objects.
        attributes : dict, optional
            Dictionary of attributes that should be added to the netCDF
            variables. can also be a dict of dicts for each variable name
            as in the data dict.
        dates_direct : boolean
            If true the dates are already converted into floating
            point number of correct magnitude.
        """
        pass

    def write_ts_all_loc(self,
                         loc_ids,
                         data,
                         dates,
                         loc_descrs=None,
                         lons=None,
                         lats=None,
                         alts=None,
                         fill_values=None,
                         attributes=None,
                         dates_direct=False):
        """
        Write time series data in bulk, for this the user has to provide
        a 2D array with dimensions (self.nloc, dates) that is filled with
        the time series of all grid points in the file.

        Parameters
        ----------
        loc_ids : numpy.ndarray
            location ids along the first axis of the data array
        data : dict
            dictionary with variable names as keys and 2D numpy.arrays as
            values
        dates: numpy.ndarray
            Array of datetime objects with same size as second dimension of
            data arrays.
        attributes : dict, optional
            Dictionary of attributes that should be added to the netCDF
            variables. can also be a dict of dicts for each variable name as
            in the data dict.
        dates_direct : boolean
            If true the dates are already converted into floating
            point number of correct magnitude
        """
        if self.n_loc != loc_ids.size:
            raise ValueError("loc_ids is not the same number of "
                             "locations in the file")
        for key in data:

            if data[key].shape[1] != dates.size:
                raise IOError("Timestamps and dataset second dimension "
                              " {:} must have the same size".format(key))

            if data[key].shape[0] != self.n_loc:
                raise IOError("Datasets first dimension {:} must have "
                              "the same size as number of locations "
                              "in the file".format(key))

        # make sure zip works even if one of the parameters is not given
        if lons is None:
            lons = np.repeat(None, self.n_loc)
        if lats is None:
            lats = np.repeat(None, self.n_loc)
        if alts is None:
            alts = np.repeat(None, self.n_loc)

        # find out if attributes is a dict to be used for all variables or if
        # there is a dictionary of attributes for each variable
        unique_attr = False
        if attributes is not None:
            if sorted(data.keys()) == sorted(attributes.keys()):
                unique_attr = True

        self._add_location(loc_ids, lons, lats, alts, loc_descrs)

        for key in data:

            internal_attributes = {'name': key, 'coordinates': 'lat lon alt'}

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
            self.write_var(key,
                           data=None,
                           dim=(self.loc_dim_name, self.obs_dim_name),
                           attr=internal_attributes,
                           dtype=data[key].dtype,
                           chunksizes=chunksizes)

            if self.write_offset is None:
                # current shape tells us how many elements are already
                # in the file
                self.write_offset = self.dataset.variables[key].shape[1]

            _slice = slice(self.write_offset, self.write_offset + dates.size,
                           None)
            self.dataset.variables[key][:, _slice] = data[key]

        # fill up time variable
        if dates_direct:
            self.dataset.variables[self.time_var][self.write_offset:] = dates
        else:
            units = self.dataset.variables[self.time_var].units
            self.dataset.variables[self.time_var][self.write_offset:] = \
                netCDF4.date2num(dates, units=units, calendar='standard')


class OrthoMultiTs(DatasetTs):
    """
    Implementation of the Orthogonal multidimensional array representation
    of time series according to the NetCDF CF-conventions 1.6.

    Parameters
    ----------
    filename : string
        filename of netCDF file. If already exiting then it will be opened
        as read only unless the append keyword is set. if the overwrite
        keyword is set then the file will be overwritten
    n_loc : int, optional
        number of locations that this netCDF file contains time series for
        only required for new file
    loc_dim_name : string, optional
        name of the location dimension
    obs_dim_name : string, optional
        name of the observations dimension
    loc_ids_name : string, optional
        name of variable that has the location id's stored
    loc_descr_name : string, optional
        name of variable that has additional location information
        stored
    time_units : string, optional
        units the time axis is given in.
        Default: "days since 1900-01-01 00:00:00"
    time_var : string, optional
        name of time variable
        Default: time
    lat_var : string, optional
        name of latitude variable
        Default: lat
    lon_var : string, optional
        name of longitude variable
        Default: lon
    alt_var : string, optional
        name of altitude variable
        Default: alt
    unlim_chunksize : int, optional
        chunksize to use along unlimited dimensions, other chunksizes
        will be calculated by the netCDF library
    read_bulk : boolean, optional
        if set to True the data of all locations is read into memory,
        and subsequent calls to read_ts read from the cache and not from disk
        this makes reading complete files faster#
    read_dates : boolean, optional
        if false dates will not be read automatically but only on specific
        request useable for bulk reading because currently the netCDF
        num2date routine is very slow for big datasets
    """

    def __init__(self,
                 filename,
                 n_loc=None,
                 loc_dim_name='locations',
                 obs_dim_name='time',
                 loc_ids_name='location_id',
                 loc_descr_name='location_description',
                 time_units="days since 1900-01-01 00:00:00",
                 time_var='time',
                 lat_var='lat',
                 lon_var='lon',
                 alt_var='alt',
                 unlim_chunksize=None,
                 read_bulk=False,
                 read_dates=True,
                 **kwargs):

        super(OrthoMultiTs,
              self).__init__(filename, n_loc, loc_dim_name, obs_dim_name,
                             loc_ids_name, loc_descr_name, time_units,
                             time_var, lat_var, lon_var, alt_var,
                             unlim_chunksize, read_bulk, read_dates, **kwargs)

        # initialize dimensions and index_variable
        if self.mode == 'w':
            self._init_dimensions()
            self._init_lookup()
            self._init_location_variables()
            self._init_location_id_and_time()

            self.global_attr['featureType'] = 'timeSeries'

    def _init_dimensions(self):
        """
        Initializes the dimensions.
        """
        if self.n_loc is None:
            raise ValueError('Number of locations have to be set for'
                             'new OrthoMultiTs file')

        self.create_dim(self.loc_dim_name, self.n_loc)
        self.create_dim(self.obs_dim_name, None)

    def _init_lookup(self):
        """
        Initializes variables for the lookup between locations and entries in
        the time series.
        """
        # nothing to be done for OrthoMultiTs
        pass

    def _get_index_of_ts(self, loc_id):
        """
        Get the index of a time series.

        Parameters
        ----------
        loc_id : int
            Location id.

        Returns
        -------
        loc_id_index : int
            Location id index.
        """
        try:
            loc_id_index = self._get_loc_id_index(loc_id)
        except IOError:
            msg = "Index for location id #{:} not found".format(loc_id)
            raise IOError(msg)

        _slice = (loc_id_index, slice(None, None, None))

        return _slice

    def _get_loc_ix_from_obs_ix(self, obs_ix):
        """
        Get location index from observation index. In case of OrthoMultiTs
        all measurements are taken at the same time and therefore all
        location id's are affected.

        Parameters
        ----------
        obs_ix : int
            Observation index.

        Returns
        -------
        loc_ix : int
            Location index.
        """
        return self.read_var(self.loc_ids_name)

    def write_ts(self,
                 loc_id,
                 data,
                 dates,
                 loc_descr=None,
                 lon=None,
                 lat=None,
                 alt=None,
                 fill_values=None,
                 attributes=None,
                 dates_direct=False):
        """
        Write time series data, if not yet existing also add location to file
        for this data format it is assumed that in each write/append cycle
        the same amount of data is added.

        Parameters
        ----------
        loc_id : int
            Location id.
        data : dict
            Dictionary with variable names as keys and
            numpy.ndarrays as values.
        dates: numpy.ndarray
            Array of datetime objects.
        attributes : dict, optional
            Dictionary of attributes that should be added to the netCDF
            variables. can also be a dict of dicts for each variable name
            as in the data dict.
        dates_direct : boolean
            If true the dates are already converted into floating
            point number of correct magnitude.
        """
        try:
            idx = self._get_loc_id_index(loc_id)
        except IOError:
            _ = self._add_location(loc_id, lon, lat, alt, loc_descr)
            idx = self._get_loc_id_index(loc_id)

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

        overlap_indexes = self.get_time_variable_overlap(dates)
        if len(dates) != len(overlap_indexes):
            self.extend_time(dates, direct=dates_direct)
            self.length_before_extend = overlap_indexes[-1]
        else:
            self.length_before_extend = 0

        for key in data:

            internal_attributes = {'name': key, 'coordinates': 'lat lon alt'}

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
            self.write_var(key,
                           data=None,
                           dim=(self.loc_dim_name, self.obs_dim_name),
                           attr=internal_attributes,
                           dtype=data[key].dtype,
                           chunksizes=chunksizes)

            if self.write_offset is None:
                # find start of elements that are not yet filled with values
                _slice_new = slice(self.length_before_extend, None, None)
                masked = \
                    np.where(
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


class ContiguousRaggedTs(DatasetTs):
    """
    Class that represents a Contiguous ragged array representation of
    time series according to NetCDF CF-conventions 1.6.

    Parameters
    ----------
    filename : string
        filename of netCDF file. If already exiting then it will be opened
        as read only unless the append keyword is set. if the overwrite
        keyword is set then the file will be overwritten
    n_loc : int, optional
        number of locations that this netCDF file contains time series for
        only required for new file
    n_obs : int, optional
        how many observations will be saved into this netCDF file in total
        only required for new file
    obs_loc_lut : string, optional
        variable name in the netCDF file that contains the lookup between
        observations and locations
    loc_dim_name : string, optional
        name of the location dimension
    obs_dim_name : string, optional
        name of the observations dimension
    loc_ids_name : string, optional
        name of variable that has the location id's stored
    loc_descr_name : string, optional
        name of variable that has additional location information
        stored
    time_units : string, optional
        units the time axis is given in.
        Default: "days since 1900-01-01 00:00:00"
    time_var : string, optional
        name of time variable
        Default: time
    lat_var : string, optional
        name of latitude variable
        Default: lat
    lon_var : string, optional
        name of longitude variable
        Default: lon
    alt_var : string, optional
        name of altitude variable
        Default: alt
    """

    def __init__(self,
                 filename,
                 n_loc=None,
                 n_obs=None,
                 obs_loc_lut='row_size',
                 obs_dim_name='obs',
                 **kwargs):

        self.n_obs = n_obs
        self.obs_loc_lut = obs_loc_lut

        super(ContiguousRaggedTs, self).__init__(filename,
                                                 n_loc=n_loc,
                                                 obs_dim_name=obs_dim_name,
                                                 **kwargs)

        if self.mode == 'w':
            self._init_dimensions()
            self._init_lookup()
            self._init_location_variables()
            self._init_location_id_and_time()

            self.global_attr['featureType'] = 'timeSeries'

        self.constant_dates = False

    def _init_dimensions(self):
        """
        Initializes the dimensions.
        """
        self.create_dim(self.loc_dim_name, self.n_loc)
        self.create_dim(self.obs_dim_name, self.n_obs)

    def _init_lookup(self):
        """
        Initializes variables for the lookup between locations and entries in
        the time series.
        """
        attr = {
            'long_name': 'number of observations at this location',
            'sample_dimension': self.obs_dim_name
        }

        self.write_var(self.obs_loc_lut,
                       data=None,
                       dim=self.loc_dim_name,
                       dtype=np.int64,
                       attr=attr,
                       chunksizes=self.unlim_chunksize)

    def _get_index_of_ts(self, loc_id):
        """
        Get slice object for time series at location loc_id.

        Parameters
        ----------
        loc_id : int
            Location id.

        Returns
        -------
        slice_obj : slice
            Slice object with start and end of time series.

        Raises
        ------
        ValueError
            If location id could not be found.
        """
        try:
            loc_id_index = self._get_loc_id_index(loc_id)
        except IOError:
            raise IOError("Index of time series for "
                          "location id #{:} not found".format(loc_id))

        if self.read_bulk and self.obs_loc_lut not in self.variables:
            self.variables[self.obs_loc_lut] = np.array(
                self.dataset.variables[self.obs_loc_lut][:])
        start = np.sum(self.variables[self.obs_loc_lut][:loc_id_index])
        end = np.sum(self.variables[self.obs_loc_lut][:loc_id_index + 1])

        return slice(start, end)

    def _get_loc_ix_from_obs_ix(self, obs_ix):
        """
        Get location index from observation index.

        Parameters
        ----------
        obs_ix : int
            Observation index.

        Returns
        -------
        loc_id_index : int
            Location id index.
        """
        bins = np.hstack((0, np.cumsum(self.variables[self.obs_loc_lut])))
        loc_id_index = np.digitize(obs_ix, bins) - 1

        return loc_id_index

    def read_time(self, loc_id):
        """
        Read the time stamps for the given location id in this case it
        works like a normal time series variable.

        Returns
        -------
        time_var : np.float64
            Time variable.
        """
        return self._read_var_ts(loc_id, self.time_var)

    def write_ts(self,
                 loc_id,
                 data,
                 dates,
                 loc_descr=None,
                 lon=None,
                 lat=None,
                 alt=None,
                 fill_values=None,
                 attributes=None,
                 dates_direct=False):
        """
        Write time series data, if not yet existing also add location to file.

        Parameters
        ----------
        loc_id : int
            Location id.
        data : dict
            Dictionary with variable names as keys and
            numpy.ndarrays as values.
        dates: numpy.array
            Array of datetime objects.
        attributes : dict, optional
            Dictionary of attributes that should be added to the netCDF
            variables. can also be a dict of dicts for each variable name
            as in the data dict.
        dates_direct : boolean
            If true the dates are already converted into floating
            point number of correct magnitude.
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
                raise IOError("Timestamps and dataset {:}",
                              "must have the same size".format(key))

        # add number of new elements to index_var
        self.dataset.variables[self.obs_loc_lut][idx] = dates.size

        index = self._get_index_of_ts(loc_id)
        for key in data:

            internal_attributes = {
                'name': key,
                'coordinates': 'time lat lon alt'
            }

            if type(fill_values) == dict:
                internal_attributes['_FillValue'] = fill_values[key]

            if attributes is not None:
                if unique_attr:
                    variable_attributes = attributes[key]
                else:
                    variable_attributes = attributes

                internal_attributes.update(variable_attributes)

            self.write_var(key,
                           data=None,
                           dim=self.obs_dim_name,
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


class IndexedRaggedTs(DatasetTs):
    """
    Class that represents a Indexed ragged array representation of time series
    according to NetCDF CF-conventions 1.6.
    """

    def __init__(self,
                 filename,
                 n_loc=None,
                 obs_loc_lut='locationIndex',
                 **kwargs):
        self.obs_loc_lut = obs_loc_lut
        super(IndexedRaggedTs, self).__init__(filename, n_loc=n_loc, **kwargs)

        if self.mode == 'w':
            self._init_dimensions()
            self._init_lookup()
            self._init_location_variables()
            self._init_location_id_and_time()

            self.global_attr['featureType'] = 'timeSeries'

        self.not_timeseries.append(self.obs_loc_lut)
        self.constant_dates = False

    def _init_dimensions(self):
        """
        Initializes the dimensions.
        """
        self.create_dim(self.loc_dim_name, self.n_loc)
        self.create_dim(self.obs_dim_name, None)

    def _init_lookup(self):
        """
        Initializes variables for the lookup between locations and entries
        in the time series.
        """
        attr = {
            'long_name': 'which location this observation is for',
            'instance_dimension': self.loc_dim_name
        }

        self.write_var(self.obs_loc_lut,
                       data=None,
                       dim=self.obs_dim_name,
                       dtype=np.int64,
                       attr=attr,
                       chunksizes=self.unlim_chunksize)

    def _get_index_of_ts(self, loc_id):
        """
        Parameters
        ----------
        loc_id: int
            Location index.

        Raises
        ------
        IOError
            if location id could not be found
        """
        try:
            loc_ix = self._get_loc_id_index(loc_id)
        except IOError:
            msg = "".join(("Time series for Location #", loc_id.__str__(),
                           " not found."))
            raise IOError(msg)

        if self.read_bulk and self.obs_loc_lut not in self.variables:
            self.variables[self.obs_loc_lut] = self.dataset.variables[
                self.obs_loc_lut][:]
        index = np.where(self.variables[self.obs_loc_lut] == loc_ix)[0]

        if len(index) == 0:
            msg = "".join(("Time series for Location #", loc_id.__str__(),
                           " not found."))
            raise IOError(msg)

        return index

    def _get_loc_ix_from_obs_ix(self, obs_ix):
        """
        Get location index from observation index.

        Parameters
        ----------
        obs_ix : int
            Observation index.

        Returns
        -------
        loc_ix : int
            Location index.
        """
        return self.variables[self.obs_loc_lut][obs_ix]

    def read_time(self, loc_id):
        """
        Read the time stamps for the given location id in this case it
        works like a normal time series variable.

        Returns
        -------
        time_var : np.float64
            Time variable.
        """
        return self._read_var_ts(loc_id, self.time_var)

    def write_ts(self,
                 loc_id,
                 data,
                 dates,
                 loc_descr=None,
                 lon=None,
                 lat=None,
                 alt=None,
                 fill_values=None,
                 attributes=None,
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

        (loc_ids_uniq, loc_ids_uniq_index,
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
            idx = self._add_location(loc_ids_uniq, lon_uniq, lat_uniq,
                                     alt_uniq, loc_descr_uniq)
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

        index = np.arange(len(self.variables[
            self.obs_loc_lut]))[len(self.variables[self.obs_loc_lut]) -
                                len(indices):]

        for key in field_names:

            internal_attributes = {
                'name': key,
                'coordinates': 'time lat lon alt'
            }

            if type(fill_values) == dict:
                internal_attributes['_FillValue'] = fill_values[key]

            if attributes is not None:
                if unique_attr:
                    variable_attributes = attributes[key]
                else:
                    variable_attributes = attributes

                internal_attributes.update(variable_attributes)

            # does nothing if variable exists already
            self.write_var(key,
                           data=None,
                           dim=self.obs_dim_name,
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


class GriddedNcTs(GriddedTsBase):

    def __init__(self, *args, **kwargs):

        self.parameters = None
        if 'parameters' in kwargs:
            self.parameters = kwargs.pop('parameters')

        self.offsets = None
        if 'offsets' in kwargs:
            self.offsets = kwargs.pop('offsets')

        self.scale_factors = None
        if 'scale_factors' in kwargs:
            self.scale_factors = kwargs.pop('scale_factors')

        self.dtypes = None
        if 'dtypes' in kwargs:
            self.dtypes = kwargs.pop('dtypes')

        self.autoscale = True
        if 'autoscale' in kwargs:
            self.autoscale = kwargs.pop('autoscale')

        self.automask = True
        if 'automask' in kwargs:
            self.automask = kwargs.pop('automask')

        super(GriddedNcTs, self).__init__(*args, **kwargs)

        self.ioclass_kws.update({
            'autoscale': self.autoscale,
            'automask': self.automask
        })
        self.dates = None

        if self.ioclass == OrthoMultiTs:
            self.read_dates = False
        else:
            self.read_dates = True

    def _open(self, gp):
        """
        Open file.

        Parameters
        ----------
        gp : int
            Grid point.

        Returns
        -------
        success : boolean
            Flag if opening the file was successful.
        """
        success = True
        cell = self.grid.gpi2cell(gp)
        filename = os.path.join(self.path,
                                '{:}.nc'.format(self.fn_format.format(cell)))

        if self.mode == 'r':
            if self.previous_cell != cell:
                self.close()

                try:
                    self.fid = self.ioclass(filename,
                                            mode=self.mode,
                                            **self.ioclass_kws)
                    self.previous_cell = cell
                except (IOError, RuntimeError):
                    success = False
                    self.fid = None
                    msg = "I/O error {:}".format(filename)
                    warnings.warn(msg, RuntimeWarning)

        if self.mode in ['w', 'a']:
            if self.previous_cell != cell:
                self.close()

                try:
                    if self.mode == 'w':
                        if 'n_loc' not in self.ioclass_kws:
                            n_loc = self.grid.grid_points_for_cell(
                                cell)[0].size
                            self.ioclass_kws['n_loc'] = n_loc
                    self.fid = self.ioclass(filename,
                                            mode=self.mode,
                                            **self.ioclass_kws)
                    self.previous_cell = cell
                    self.ioclass_kws.pop('n_loc', None)
                except (IOError, RuntimeError):
                    success = False
                    self.fid = None
                    msg = "I/O error {:}".format(filename)
                    warnings.warn(msg, RuntimeWarning)

        return success

    def _read_gp(self, gpi, period=None, **kwargs):
        """
        Method reads data for given gpi, additional keyword arguments
        are passed to ioclass.read_ts

        Parameters
        ----------
        gp : int
            Grid point.
        period : list
            2 element array containing datetimes [start, end]

        Returns
        -------
        ts : pandas.DataFrame
            Time series data.
        """
        if self.mode in ['w', 'a']:
            raise IOError("trying to read file is in 'write/append' mode")

        if not self._open(gpi):
            return None

        if self.parameters is None:
            data = self.fid.read_all_ts(gpi, **kwargs)
        else:
            data = self.fid.read_ts(self.parameters, gpi, **kwargs)

        if self.dates is None or self.read_dates:
            if "dates_direct" in kwargs:
                self.dates = self.fid.read_time(gpi)
            else:
                self.dates = self.fid.read_dates(gpi)

        time = self.dates

        # remove time column from dataframe, only index should contain time
        try:
            data.pop('time')
        except KeyError:
            # if the time value is not found then do nothing
            pass

        ts = pd.DataFrame(data, index=time)

        if period is not None:
            ts = ts[period[0]:period[1]]

        if self.dtypes is not None:
            for dtype_column in self.dtypes:
                if dtype_column in ts.columns:
                    try:
                        ts[dtype_column] = ts[dtype_column].astype(
                            self.dtypes[dtype_column])
                    except ValueError:
                        raise ValueError(
                            "Dtype conversion did not work. Try turning off automatic masking."
                        )

        if self.scale_factors is not None:
            for scale_column in self.scale_factors:
                if scale_column in ts.columns:
                    ts[scale_column] *= self.scale_factors[scale_column]

        if self.offsets is not None:
            for offset_column in self.offsets:
                if offset_column in ts.columns:
                    ts[offset_column] += self.offsets[offset_column]

        return ts

    def _write_gp(self, gp, data, **kwargs):
        """
        Method writing data for given gpi.

        Parameters
        ----------
        gp : int
            Grid point.
        data : pandas.DataFrame
            Time series data to write. Index has to be pandas.DateTimeIndex.
        """
        if self.mode == 'r':
            raise IOError("trying to write but file is in 'read' mode")

        self._open(gp)
        lon, lat = self.grid.gpi2lonlat(gp)

        ds = data.to_dict('series')

        for key in ds:
            ds[key] = ds[key].values

        self.fid.write_ts(gp,
                          ds,
                          data.index.to_pydatetime(),
                          lon=lon,
                          lat=lat,
                          **kwargs)


class GriddedNcOrthoMultiTs(GriddedNcTs):

    def __init__(self, *args, **kwargs):
        kwargs['ioclass'] = OrthoMultiTs
        super(GriddedNcOrthoMultiTs, self).__init__(*args, **kwargs)


class GriddedNcContiguousRaggedTs(GriddedNcTs):

    def __init__(self, *args, **kwargs):
        kwargs['ioclass'] = ContiguousRaggedTs
        super(GriddedNcContiguousRaggedTs, self).__init__(*args, **kwargs)


class GriddedNcIndexedRaggedTs(GriddedNcTs):

    def __init__(self, *args, **kwargs):
        kwargs['ioclass'] = IndexedRaggedTs
        super(GriddedNcIndexedRaggedTs, self).__init__(*args, **kwargs)

    def write_cell(self, cell, gpi, data, datefield):
        """
        Write complete data set into cell file.

        Parameters
        ----------
        cell : int
            Cell number.
        gpi : numpy.ndarray
            Location ids.
        data : dict or numpy record array
            dictionary with variable names as keys and numpy.arrays as values
        datefield: string
            field in the data dict that contains dates in correct format
        """
        if isinstance(self.grid, CellGrid) is False:
            raise TypeError("Associated grid is not of type "
                            "pygeogrids.CellGrid.")

        if self.mode != 'w':
            raise ValueError("File not opened in write mode.")

        tmp_cell = np.unique(self.grid.arrcell[gpi])

        if tmp_cell.size > 1 or tmp_cell != cell:
            raise ValueError("GPIs do not correspond to given cell.")

        lons = self.grid.arrlon[gpi]
        lats = self.grid.arrlat[gpi]

        filename = os.path.join(self.path,
                                '{:}.nc'.format(self.fn_format.format(cell)))

        if os.path.isfile(filename):
            mode = 'a'
        else:
            mode = 'w'

        if self.previous_cell != cell:
            self.close()
            self.previous_cell = cell
            if self.mode == 'w':
                if 'n_loc' not in self.ioclass_kws:
                    n_loc = self.grid.grid_points_for_cell(cell)[0].size
                    self.ioclass_kws['n_loc'] = n_loc
            self.fid = self.ioclass(filename, mode=mode, **self.ioclass_kws)
            self.ioclass_kws.pop('n_loc', None)

        if type(data) != dict:
            data = {key: data[key] for key in data.dtype.names}

        dates = data[datefield]
        del data[datefield]
        self.fid.write_ts(gpi,
                          data,
                          dates,
                          lon=lons,
                          lat=lats,
                          dates_direct=True)
        self.close()
