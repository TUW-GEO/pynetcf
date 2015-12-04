# Copyright (c) 2015, Vienna University of Technology,
# Department of Geodesy and Geoinformation
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
Testing time series class.
"""

import os
import unittest
from tempfile import mkdtemp

import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import numpy.testing as nptest

import pynetcf.time_series as nc
import pygeogrids.grids as grids

import sys
if sys.version_info < (3, 0):
    range = xrange


class OrthoMultiTest(unittest.TestCase):

    def setUp(self):
        self.testfilename = os.path.join(mkdtemp(), 'test.nc')

    def tearDown(self):
        os.remove(self.testfilename)

    def test_file_io_simple(self):

        with nc.OrthoMultiTs(self.testfilename, mode='w',
                             n_loc=3) as dataset:
            for n_data in [5]:
                for location in [1, 2, 3]:

                    data = {'test': np.arange(n_data)}
                    base = datetime(2007, 1, n_data)
                    dates = np.array([base + timedelta(hours=i)
                                      for i in range(n_data)])
                    dataset.write_ts(
                        location, data, dates, loc_descr='first station',
                        lon=0, lat=0, alt=5)

        with nc.OrthoMultiTs(self.testfilename) as dataset:
            data = dataset.read_all_ts(2)
            nptest.assert_array_equal(data['test'], np.arange(5))

            test_dates = []
            for n_data in [5]:
                base = datetime(2007, 1, n_data)
                test_dates.append(
                    np.array([base + timedelta(hours=i)
                              for i in range(n_data)]))
            dates = np.concatenate(test_dates)
            nptest.assert_array_equal(data['time'], dates)

    def test_file_io_2_steps(self):

        with nc.OrthoMultiTs(self.testfilename, n_loc=3,
                             mode='w') as dataset:
            for n_data in [5]:
                for location in [1, 2, 3]:

                    data = {'test': np.arange(n_data)}
                    base = datetime(2007, 1, n_data)
                    dates = np.array([base + timedelta(hours=i)
                                      for i in range(n_data)])
                    dataset.write_ts(
                        location, data, dates, loc_descr='first station',
                        lon=0, lat=0, alt=5, fill_values={'test': -1})

        with nc.OrthoMultiTs(self.testfilename, n_loc=3,
                             mode='a') as dataset:
            for n_data in [5]:
                for location in [1, 2, 3]:

                    data = {'test': np.arange(n_data) + n_data}
                    base = datetime(2007, 2, n_data)
                    dates = np.array([base + timedelta(hours=i)
                                      for i in range(n_data)])
                    dataset.write_ts(
                        location, data, dates, loc_descr='first station',
                        lon=0, lat=0, alt=5, fill_values={'test': -1})

        with nc.OrthoMultiTs(self.testfilename) as dataset:
            data = dataset.read_all_ts(2)
            nptest.assert_array_equal(data['test'], np.arange(10))

            test_dates = []
            for n_data in [5]:
                base = datetime(2007, 1, n_data)
                test_dates.append(
                    np.array([base + timedelta(hours=i)
                              for i in range(n_data)]))
                base = datetime(2007, 2, n_data)
                test_dates.append(
                    np.array([base + timedelta(hours=i)
                              for i in range(n_data)]))
            dates = np.concatenate(test_dates)
            nptest.assert_array_equal(data['time'], dates)

    def test_file_write_ts_all(self):

        with nc.OrthoMultiTs(self.testfilename, n_loc=3,
                             mode='w') as dataset:
            n_data = 5
            locations = np.array([1, 2, 3])
            data = {'test': np.arange(n_data * 3).reshape(3, n_data)}
            base = datetime(2007, 1, n_data)
            dates = np.array([base + timedelta(hours=i)
                              for i in range(n_data)])
            descriptions = np.repeat([str('station')], 3).tolist()

            dataset.write_ts_all_loc(locations, data, dates,
                                     loc_descrs=descriptions,
                                     lons=np.arange(3),
                                     lats=np.arange(3), alts=np.arange(3))

        with nc.OrthoMultiTs(self.testfilename) as dataset:
            data = dataset.read_all_ts(2)
            nptest.assert_array_equal(data['test'], np.arange(5) + 5)
            test_dates = []
            for n_data in [5]:
                base = datetime(2007, 1, n_data)
                test_dates.append(
                    np.array([base + timedelta(hours=i)
                              for i in range(n_data)]))
            dates = np.concatenate(test_dates)
            nptest.assert_array_equal(data['time'], dates)

    def test_file_write_ts_all_1_location(self):

        with nc.OrthoMultiTs(self.testfilename, n_loc=1,
                             mode='w') as dataset:
            n_data = 5
            locations = np.array([1])
            data = {'test': np.arange(n_data).reshape(1, n_data)}
            base = datetime(2007, 1, n_data)
            dates = np.array([base + timedelta(hours=i)
                              for i in range(n_data)])
            descriptions = np.repeat([str('station')], 1).tolist()

            dataset.write_ts_all_loc(locations, data, dates,
                                     loc_descrs=descriptions, lons=np.arange(
                                         1),
                                     lats=np.arange(1), alts=None)

        with nc.OrthoMultiTs(self.testfilename) as dataset:
            data = dataset.read_all_ts(1)
            nptest.assert_array_equal(data['test'], np.arange(5))
            test_dates = []
            for n_data in [5]:
                base = datetime(2007, 1, n_data)
                test_dates.append(
                    np.array([base + timedelta(hours=i)
                              for i in range(n_data)]))
            dates = np.concatenate(test_dates)
            nptest.assert_array_equal(data['time'], dates)

    def test_file_write_ts_all_attributes(self):

        with nc.OrthoMultiTs(self.testfilename, n_loc=3,
                             mode='w') as dataset:
            n_data = 5
            locations = np.array([1, 2, 3])
            data = {'test': np.arange(n_data * 3).reshape(3, n_data),
                    'test2': np.arange(n_data * 3).reshape(3, n_data)}
            base = datetime(2007, 1, n_data)
            dates = np.array([base + timedelta(hours=i)
                              for i in range(n_data)])
            descriptions = np.repeat([str('station')], 3).tolist()

            dataset.write_ts_all_loc(locations, data, dates,
                                     loc_descrs=descriptions,
                                     lons=np.arange(3),
                                     lats=np.arange(3), alts=np.arange(3),
                                     attributes={'testattribute':
                                                 'teststring'})

        with nc.OrthoMultiTs(self.testfilename) as dataset:
            data = dataset.read_all_ts(2)
            nptest.assert_array_equal(data['test'], np.arange(5) + 5)
            assert dataset.dataset.variables[
                'test'].testattribute == 'teststring'
            assert dataset.dataset.variables[
                'test2'].testattribute == 'teststring'
            test_dates = []
            for n_data in [5]:
                base = datetime(2007, 1, n_data)
                test_dates.append(
                    np.array([base + timedelta(hours=i)
                              for i in range(n_data)]))
            dates = np.concatenate(test_dates)
            nptest.assert_array_equal(data['time'], dates)

    def test_file_write_ts_attributes_for_each(self):
        """
        test writing two datasets with attributes for each dataset
        """

        with nc.OrthoMultiTs(self.testfilename, n_loc=3,
                             mode='w') as dataset:
            n_data = 5
            locations = np.array([1, 2, 3])
            data = {'test': np.arange(n_data * 3).reshape(3, n_data),
                    'test2': np.arange(n_data * 3).reshape(3, n_data)}
            base = datetime(2007, 1, n_data)
            dates = np.array([base + timedelta(hours=i)
                              for i in range(n_data)])
            descriptions = np.repeat([str('station')], 3).tolist()

            dataset.write_ts_all_loc(locations, data, dates,
                                     loc_descrs=descriptions,
                                     lons=np.arange(3),
                                     lats=np.arange(3), alts=np.arange(3),
                                     attributes={'test':
                                                 {'testattribute':
                                                  'teststring'},
                                                 'test2': {'testattribute2':
                                                           'teststring2'}})

        with nc.OrthoMultiTs(self.testfilename) as dataset:
            data = dataset.read_all_ts(2)
            nptest.assert_array_equal(data['test'], np.arange(5) + 5)
            assert dataset.dataset.variables[
                'test'].testattribute == 'teststring'
            assert dataset.dataset.variables[
                'test2'].testattribute2 == 'teststring2'
            test_dates = []
            for n_data in [5]:
                base = datetime(2007, 1, n_data)
                test_dates.append(
                    np.array([base + timedelta(hours=i)
                              for i in range(n_data)]))
            dates = np.concatenate(test_dates)
            nptest.assert_array_equal(data['time'], dates)


class DatasetContiguousTest(unittest.TestCase):

    def setUp(self):
        self.testfilename = os.path.join(mkdtemp(), 'test.nc')

    def tearDown(self):
        os.remove(self.testfilename)

    def test_file_writing(self):

        dates = np.array([datetime(2007, 1, 1), datetime(2007, 2, 1),
                datetime(2007, 3, 1)])

        with nc.ContiguousRaggedTs(self.testfilename,
                                   n_loc=3, n_obs=9, mode='w') as dataset:
            data = {'test': np.arange(3)}
            dataset.write_ts(1, data, dates, loc_descr='first station',
                             lon=1, lat=1, alt=1)
            dataset.write_ts(2, data, dates, loc_descr='second station',
                             lon=2, lat=2, alt=2)
            dataset.write_ts(3, data, dates, loc_descr='third station',
                             lon=3, lat=3, alt=3)

        with nc.ContiguousRaggedTs(self.testfilename) as dataset:
            data = dataset.read_all_ts(1)
            nptest.assert_array_equal(data['test'], np.arange(3))
            nptest.assert_array_equal(data['time'], dates)

    def test_unlim_obs_file_writing(self):

        dates = np.array([datetime(2007, 1, 1), datetime(2007, 2, 1),
                 datetime(2007, 3, 1)])

        with nc.ContiguousRaggedTs(self.testfilename,
                                   n_loc=3, mode='w') as dataset:
            data = {'test': np.arange(3)}
            dataset.write_ts(1, data, dates, loc_descr='first station',
                             lon=1, lat=1, alt=1)
            dataset.write_ts(2, data, dates, loc_descr='second station',
                             lon=2, lat=2, alt=2)
            dataset.write_ts(3, data, dates, loc_descr='third station',
                             lon=3, lat=3, alt=3)

        with nc.ContiguousRaggedTs(self.testfilename) as dataset:
            data = dataset.read_all_ts(1)
            nptest.assert_array_equal(data['test'], np.arange(3))
            nptest.assert_array_equal(data['time'], dates)

    def test_unlim_loc_file_writing(self):

        dates = np.array([datetime(2007, 1, 1), datetime(2007, 2, 1),
                 datetime(2007, 3, 1)])

        with nc.ContiguousRaggedTs(self.testfilename, mode='w') as dataset:
            data = {'test': np.arange(3)}
            dataset.write_ts(1, data, dates, loc_descr='first station',
                             lon=1, lat=1, alt=1)
            dataset.write_ts(2, data, dates, loc_descr='second station',
                             lon=2, lat=2, alt=2)
            dataset.write_ts(3, data, dates, loc_descr='third station',
                             lon=3, lat=3, alt=3)

        with nc.ContiguousRaggedTs(self.testfilename) as dataset:
            data = dataset.read_all_ts(1)
            nptest.assert_array_equal(data['test'], np.arange(3))
            nptest.assert_array_equal(data['time'], dates)


class DatasetIndexedTest(unittest.TestCase):

    def setUp(self):
        self.testfilename = os.path.join(mkdtemp(), 'test.nc')

    def tearDown(self):
        os.remove(self.testfilename)

    def test_file_writing(self):

        with nc.IndexedRaggedTs(self.testfilename, n_loc=3,
                                mode='w') as dataset:
            for n_data in [2, 5, 6]:
                for location in [1, 2, 3]:
                    data = {'test': np.arange(n_data)}
                    base = datetime(2007, 1, n_data)
                    dates = np.array([base + timedelta(hours=i)
                                      for i in range(n_data)])
                    dataset.write_ts(location, data, dates,
                                     loc_descr='first station',
                                     lon=location, lat=location,
                                     alt=location)

        with nc.IndexedRaggedTs(self.testfilename) as dataset:
            data = dataset.read_all_ts(1)
            nptest.assert_array_equal(
                data['test'], np.concatenate([np.arange(2), np.arange(5),
                                              np.arange(6)]))
            test_dates = []
            for n_data in [2, 5, 6]:
                base = datetime(2007, 1, n_data)
                test_dates.append(
                    np.array([base + timedelta(hours=i)
                              for i in range(n_data)]))
            dates = np.concatenate(test_dates)
            nptest.assert_array_equal(data['time'], dates)

    def test_unlim_loc_file_writing(self):

        with nc.IndexedRaggedTs(self.testfilename, mode='w') as dataset:
            for n_data in [2, 5, 6]:
                for location in [1, 2, 3]:
                    data = {'test': np.arange(n_data)}
                    base = datetime(2007, 1, n_data)
                    dates = np.array([base + timedelta(hours=i)
                                      for i in range(n_data)])
                    dataset.write_ts(location, data, dates,
                                     loc_descr='first station',
                                     lon=location, lat=location,
                                     alt=location)

        with nc.IndexedRaggedTs(self.testfilename) as dataset:
            data = dataset.read_all_ts(1)
            nptest.assert_array_equal(
                data['test'], np.concatenate([np.arange(2), np.arange(5),
                                              np.arange(6)]))
            test_dates = []
            for n_data in [2, 5, 6]:
                base = datetime(2007, 1, n_data)
                test_dates.append(
                    np.array([base + timedelta(hours=i)
                              for i in range(n_data)]))
            dates = np.concatenate(test_dates)
            nptest.assert_array_equal(data['time'], dates)

    def test_file_writing_with_attributes(self):

        with nc.IndexedRaggedTs(self.testfilename, n_loc=3,
                                mode='w') as dataset:
            for n_data in [2, 5, 6]:
                for location in [1, 2, 3]:

                    data = {'test': np.arange(n_data)}
                    base = datetime(2007, 1, n_data)
                    dates = np.array([base + timedelta(hours=i)
                                      for i in range(n_data)])
                    dataset.write_ts(location, data, dates,
                                     loc_descr='first station', lon=0, lat=0,
                                     alt=5,
                                     attributes={'testattribute':
                                                 'teststring'})

        with nc.IndexedRaggedTs(self.testfilename) as dataset:
            data = dataset.read_all_ts(1)
            assert dataset.dataset.variables[
                'test'].testattribute == 'teststring'
            nptest.assert_array_equal(
                data['test'], np.concatenate([np.arange(2), np.arange(5),
                                              np.arange(6)]))

            test_dates = []
            for n_data in [2, 5, 6]:
                base = datetime(2007, 1, n_data)
                test_dates.append(
                    np.array([base + timedelta(hours=i)
                              for i in range(n_data)]))
            dates = np.concatenate(test_dates)
            nptest.assert_array_equal(data['time'], dates)


class DatasetGriddedTsTests(unittest.TestCase):

    def setUp(self):
        self.testdatapath = os.path.join(mkdtemp())
        self.testfilename = os.path.join(self.testdatapath, '0107.nc')
        self.grid = grids.genreg_grid().to_cell_grid()

    def tearDown(self):
        os.remove(self.testfilename)

    def _test_writing_with_attributes(self, ioclass):

        dates = pd.date_range(start='2007-01-01', end='2007-02-01')

        ts = pd.DataFrame({'var1': np.arange(len(dates)),
                           'var2': np.arange(len(dates))}, index=dates)

        attributes = {'var1': {'testattribute': 'teststring'},
                      'var2': {'testattribute2': 'teststring2'}}

        dataset = ioclass(self.testdatapath, nc.IndexedRaggedTs,
                          mode='w', grid=self.grid)
        for gpi in [10, 11, 12]:
            dataset.write_gp(gpi, ts, attributes=attributes)

        dataset = ioclass(self.testdatapath, nc.IndexedRaggedTs,
                          mode='a', grid=self.grid)
        for gpi in [13, 10]:
            dataset.write_gp(gpi, ts)

        dataset = ioclass(self.testdatapath, nc.IndexedRaggedTs,
                          grid=self.grid)
        for gpi in [11, 12]:
            ts = dataset.read_gp(gpi)
            nptest.assert_array_equal(ts['var1'], np.arange(len(dates)))
            nptest.assert_array_equal(ts['var2'], np.arange(len(dates)))

    def _test_writing_with_attributes_prepared_classes(self, ioclass):

        dates = pd.date_range(start='2007-01-01', end='2007-02-01')

        ts = pd.DataFrame({'var1': np.arange(len(dates)),
                           'var2': np.arange(len(dates))}, index=dates)

        attributes = {'var1': {'testattribute': 'teststring'},
                      'var2': {'testattribute2': 'teststring2'}}

        dataset = ioclass(self.testdatapath, self.grid, mode='w')
        for gpi in [10, 11, 12]:
            dataset.write(gpi, ts, attributes=attributes)

        dataset = ioclass(self.testdatapath, self.grid, mode='a')
        for gpi in [13, 10]:
            dataset.write(gpi, ts)

        dataset = ioclass(self.testdatapath, self.grid, mode='r')
        for gpi in [11, 12]:
            ts = dataset.read(gpi)
            nptest.assert_array_equal(ts['var1'], np.arange(len(dates)))
            nptest.assert_array_equal(ts['var2'], np.arange(len(dates)))

    def test_writing_with_attributes_GriddedTs(self):
        self._test_writing_with_attributes(nc.GriddedTs)

    def test_writing_with_attributes_GriddedContigious(self):
        self._test_writing_with_attributes_prepared_classes(
            nc.GriddedNcContiguousRaggedTs)

    def test_writing_with_attributes_GriddedIndexed(self):
        self._test_writing_with_attributes_prepared_classes(
            nc.GriddedNcIndexedRaggedTs)
    def test_rw_dates_direct(self):

        dates = pd.date_range(start='2007-01-01', end='2007-02-01')

        ts = pd.DataFrame({'var1': np.arange(len(dates)),
                           'var2': np.arange(len(dates))}, index=dates)

        attributes = {'var1': {'testattribute': 'teststring'},
                      'var2': {'testattribute2': 'teststring2'}}

        dataset = nc.GriddedTs(self.testdatapath, nc.IndexedRaggedTs,
                               mode='w', grid=self.grid)
        for gpi in [10, 11, 12]:
            dataset.write_gp(gpi, ts, attributes=attributes)

        dataset = nc.GriddedTs(self.testdatapath, nc.IndexedRaggedTs,
                               mode='a', grid=self.grid)
        for gpi in [13, 10]:
            dataset.write_gp(gpi, ts)

        dataset = nc.GriddedTs(self.testdatapath, nc.IndexedRaggedTs,
                               grid=self.grid)
        for gpi in [11, 12]:
            ts = dataset.read_gp(gpi, dates_direct=True)
            nptest.assert_array_equal(ts['var1'], np.arange(len(dates)))
            nptest.assert_array_equal(ts['var2'], np.arange(len(dates)))
            nptest.assert_array_equal(
                ts.index.values, np.arange(len(dates)) + 39081)


if __name__ == "__main__":
    unittest.main()
