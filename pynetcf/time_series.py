# Copyright (c) 2015, Vienna University of Technology,
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

import os
import pandas as pd

from base import OrthoMultiTs
import pytesmo.io.dataset_base as dsbase


class NetCDFGriddedTS(dsbase.DatasetTSBase):

    """
    Base class for reading time series data on a cell grid
    written with one of the netCDF writers
    in general.io.netcdf

    Parameters
    ----------
    grid : grid object
        that implements find_nearest_gpi() and gpi2cell()
    read_bulk : boolean, optional
        if true read_bulk will be activated
    data_path : string, optional
        path to the data directory
    parameters : list, optional
        if given only parameters from this list will be read
    ioclass : class
        class to use to read the data
    offsets : dict, optional
        offset to apply to a variable, in addition to the offset
        specified in the netCDF file
    scale_factors : dict, optional
        scale factors to apply to a variable

    """

    def __init__(self, grid=None, read_bulk=False,
                 data_path=None, parameters=None,
                 ioclass=None, cell_filename_template='%04d.nc',
                 offsets=None, scale_factors=None):

        self.parameters = parameters
        self.ioclass = ioclass
        self.netcdf_obj = None
        self.cell_file_templ = cell_filename_template
        self.read_bulk = read_bulk
        self.previous_cell = None
        self.offsets = offsets
        self.scale_factors = scale_factors
        if self.ioclass == OrthoMultiTs:
            self.read_dates = False
        else:
            self.read_dates = True
        self.dates = None
        super(NetCDFGriddedTS, self).__init__(data_path, grid)

    def read_gp(self, gpi, period=None):
        """
        Method reads data for given gpi

        Parameters
        ----------
        gpi : int
            grid point index on dgg grid
        period : list
            2 element array containing datetimes [start,end]

        Returns
        -------
        ts : pandas.DataFrame
            time series
        """
        cell = self.grid.gpi2cell(gpi)
        filename = os.path.join(self.path, self.cell_file_templ % cell)
        if self.read_bulk:
            if self.previous_cell != cell:
                # print "Switching cell to %04d reading gpi %d" % (cell, gpi)
                if self.netcdf_obj is not None:
                    self.netcdf_obj.close()
                    self.netcdf_obj = None
                self.netcdf_obj = self.ioclass(filename,
                                               read_bulk=self.read_bulk,
                                               read_dates=self.read_dates)
                self.previous_cell = cell
        else:
            if self.netcdf_obj is not None:
                self.netcdf_obj.close()
                self.netcdf_obj = None
            self.netcdf_obj = self.ioclass(filename, read_bulk=self.read_bulk,
                                           read_dates=self.read_dates)

        if self.parameters is None:
            data = self.netcdf_obj.read_all_ts(gpi)
        else:
            data = self.netcdf_obj.read_ts(self.parameters, gpi)

        if self.dates is None or self.read_dates:
            self.dates = self.netcdf_obj.read_dates(gpi)
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

        if self.scale_factors is not None:
            for scale_column in self.scale_factors:
                if scale_column in ts.columns:
                    ts[scale_column] *= self.scale_factors[scale_column]

        if self.offsets is not None:
            for offset_column in self.offsets:
                if offset_column in ts.columns:
                    ts[offset_column] += self.offsets[offset_column]

        if not self.read_bulk:
            self.netcdf_obj.close()
            self.netcdf_obj = None
        return ts
