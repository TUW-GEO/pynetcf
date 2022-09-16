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
Class for gridded time series.
"""
import os
import warnings

import pandas as pd
from pygeobase.io_base import GriddedTsBase
from pygeogrids import CellGrid

from pynetcf.time_series_concrete import OrthoMultiTs, ContiguousRaggedTs, IndexedRaggedTs


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

        self.ioclass_kws.update({'autoscale': self.autoscale,
                                 'automask': self.automask})
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
                    self.fid = self.ioclass(filename, mode=self.mode,
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
                            n_loc = self.grid.grid_points_for_cell(cell)[
                                0].size
                            self.ioclass_kws['n_loc'] = n_loc
                    self.fid = self.ioclass(filename, mode=self.mode,
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
                            "Dtype conversion did not work. Try turning off automatic masking.")

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

        self.fid.write_ts(gp, ds, data.index.to_pydatetime(),
                          lon=lon, lat=lat, **kwargs)


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
            self.fid = self.ioclass(filename, mode=mode,
                                    **self.ioclass_kws)
            self.ioclass_kws.pop('n_loc', None)

        if type(data) != dict:
            data = {key: data[key] for key in data.dtype.names}

        dates = data[datefield]
        del data[datefield]
        self.fid.write_ts(gpi, data, dates, lon=lons, lat=lats,
                          dates_direct=True)
        self.close()
