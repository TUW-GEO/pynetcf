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
Abstract class providing an interface for reading and writing time series in NetCDF files
according to the Climate Forecast Metadata Conventions
(http://cfconventions.org/).
"""

import numpy as np
import netCDF4

from pynetcf.base import Dataset
from abc import ABC, abstractmethod


class DatasetTs(Dataset, ABC):

    """
    Abstract class to store common methods for NetCDF time series such as OrthoMulti-, ContiguousRaggedArray- and
    IndexedRaggedArray-representation. Implemented according to the NetCDF CF-conventions 1.6.

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

    def __init__(self, filename, n_loc=None, loc_dim_name='locations',
                 obs_dim_name='time', loc_ids_name='location_id',
                 loc_descr_name='location_description',
                 time_units="days since 1900-01-01 00:00:00",
                 time_var='time', lat_var='lat', lon_var='lon', alt_var='alt',
                 unlim_chunksize=None, read_bulk=False, read_dates=True,
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
        self.write_var(self.lon_var, data=None, dim=self.loc_dim_name,
                       attr={'standard_name': 'longitude',
                             'long_name': 'location longitude',
                             'units': 'degrees_east',
                             'valid_range': (-180.0, 180.0)},
                       dtype=np.float32)

        self.write_var(self.lat_var, data=None, dim=self.loc_dim_name,
                       attr={'standard_name': 'latitude',
                             'long_name': 'location latitude',
                             'units': 'degrees_north',
                             'valid_range': (-90.0, 90.0)},
                       dtype=np.float32)

        self.write_var(self.alt_var, data=None, dim=self.loc_dim_name,
                       attr={'standard_name': 'height',
                             'long_name': 'vertical distance above the '
                             'surface',
                             'units': 'm',
                             'positive': 'up',
                             'axis': 'Z'},
                       dtype=np.float32)

    def _init_location_id_and_time(self):
        """
        Initialize the dimensions and variables that are the basis of
        the format.
        """
        # make variable that contains the location id
        self.write_var(self.loc_ids_name, data=None, dim=self.loc_dim_name,
                       dtype=np.int64)

        self.write_var(self.loc_descr_name, data=None, dim=self.loc_dim_name,
                       dtype=str)

        # initialize time variable
        self.write_var(self.time_var, data=None, dim=self.obs_dim_name,
                       attr={'standard_name': 'time',
                             'long_name': 'time of measurement',
                             'units': self.time_units},
                       dtype=np.float64, chunksizes=self.unlim_chunksize)

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
                self.dataset.variables[self.loc_descr_name][
                    index] = str(loc_descr)
            else:
                self.dataset.variables[self.loc_descr_name][
                    index] = loc_descr[loc_ids_new].astype(object)

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
            calendar='standard', only_use_cftime_datetimes=False,
            only_use_python_datetimes=True)

        return self.dates.astype('datetime64[ns]')

    def _read_var_ts(self, loc_id, var):
        """
        read a time series of a variable at a given location id

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
        ts = self.read_ts(
            self._get_all_ts_variables(), loc_id, dates_direct=dates_direct)

        return ts

    def extend_time(self, dates, direct=False):
        """
        Extend the time dimension and variable by the given dates

        Parameters
        ----------
        dates : numpy.array of datetime objects or floats
        direct : boolean
            if true the dates are already converted into floating
            point number of correct magnitude
        """
        if direct:
            self.append_var(self.time_var, dates)
        else:
            units = self.dataset.variables[self.time_var].units
            self.append_var(self.time_var, netCDF4.date2num(dates, units=units,
                                                            calendar='standard'))

    def get_time_variable_overlap(self, dates):
        """Figure out if a new date array has a overlap with the already existing time
        variable.

        Return the index of the existing time variable where the new dates
        should be located.

        At the moment this only handles cases where all dates are new or none
        are new.

        Parameters
        ----------
        dates: list
            list of datetime objects


        Returns
        -------
        indexes: np.ndarray
           Array of indexes that overlap

        """
        timevar = self.dataset.variables[self.time_var]
        if timevar.size == 0:
            indexes = np.array([0])
        else:
            try:
                indexes = netCDF4.date2index(
                    dates, timevar)
            except ValueError:
                indexes = np.array([timevar.size])

        return indexes

    @abstractmethod
    def write_ts(self, loc_id, data, dates,
                 loc_descr=None, lon=None, lat=None, alt=None,
                 fill_values=None, attributes=None, dates_direct=False):
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

    def write_ts_all_loc(self, loc_ids, data, dates, loc_descrs=None,
                         lons=None, lats=None, alts=None, fill_values=None,
                         attributes=None, dates_direct=False):
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

            internal_attributes = {'name': key,
                                   'coordinates': 'lat lon alt'}

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
            self.write_var(key, data=None, dim=(self.loc_dim_name,
                                                self.obs_dim_name),
                           attr=internal_attributes,
                           dtype=data[key].dtype, chunksizes=chunksizes)

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