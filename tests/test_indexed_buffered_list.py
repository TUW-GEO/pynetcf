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
Testing indexed ragged array implementation, buffered in a list.
"""

import os
import unittest
from tempfile import mkdtemp

from datetime import datetime, timedelta
import numpy as np
import numpy.testing as nptest

from pynetcf.time_series_buffered import BufferedIndexedRaggedTs


class DatasetIndexedTest(unittest.TestCase):

    def setUp(self):
        self.testfilename = os.path.join(mkdtemp(), 'test.nc')

    def tearDown(self):
        os.remove(self.testfilename)

    def test_file_writing(self):

        with BufferedIndexedRaggedTs(self.testfilename, n_loc=3,
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

        with BufferedIndexedRaggedTs(self.testfilename) as dataset:
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

    def test_file_writing_multiple_points_at_once(self):
        """
        Write multiple points at once. This means that we can have multiple
        locations more than once. Dates and data must be in the same order as
        locations. This mean we only need to translate from locations to index
        and can write the data as is.
        """
        with BufferedIndexedRaggedTs(self.testfilename, n_loc=3,
                             mode='w') as dataset:
            locations = np.array([1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3])
            data = {'test': np.concatenate([np.arange(2),
                                            np.arange(5),
                                            np.arange(6)])}
            dates = []
            for n_data in [2, 5, 6]:
                base = datetime(2007, 1, n_data)
                dates.append(np.array([base + timedelta(hours=i)
                                       for i in range(n_data)]))
            dates = np.concatenate(dates)
            dataset.write_ts(locations, data, dates,
                             loc_descr=['first station'] * 13,
                             lon=locations, lat=locations,
                             alt=locations)

        with BufferedIndexedRaggedTs(self.testfilename) as dataset:
            for gpis, n_data in zip([1, 2, 3], [2, 5, 6]):
                data = dataset.read_all_ts(gpis)
                nptest.assert_array_equal(data['test'], np.arange(n_data))
                test_dates = []
                base = datetime(2007, 1, n_data)
                test_dates.append(
                    np.array([base + timedelta(hours=i)
                              for i in range(n_data)]))
                dates = np.concatenate(test_dates)
                nptest.assert_array_equal(data['time'], dates)

    def test_file_writing_multiple_points_at_once_two_steps(self):
        """
        Write multiple points at once. Add more during an append.
        """
        with BufferedIndexedRaggedTs(self.testfilename, n_loc=4,
                             mode='w') as dataset:
            locations = np.array([1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3])
            data = {'test': np.concatenate([np.arange(2),
                                            np.arange(5),
                                            np.arange(6)])}
            dates = []
            for n_data in [2, 5, 6]:
                base = datetime(2007, 1, n_data)
                dates.append(np.array([base + timedelta(hours=i)
                                       for i in range(n_data)]))
            dates = np.concatenate(dates)
            dataset.write_ts(locations, data, dates,
                             loc_descr=['first station'] * 13,
                             lon=locations, lat=locations,
                             alt=locations)

        with BufferedIndexedRaggedTs(self.testfilename, n_loc=4,
                             mode='a') as dataset:
            locations = np.array([1, 1, 4, 4])
            data = {'test': np.concatenate([np.arange(2),
                                            np.arange(2)])}
            dates = []
            for n_data in [2, 2]:
                base = datetime(2007, 2, n_data)
                dates.append(np.array([base + timedelta(hours=i)
                                       for i in range(n_data)]))
            dates = np.concatenate(dates)
            dataset.write_ts(locations, data, dates,
                             loc_descr=['first station'] * 4,
                             lon=locations, lat=locations,
                             alt=locations)

        with BufferedIndexedRaggedTs(self.testfilename) as dataset:
            for gpis, n_data, base_month in zip([1, 2, 3, 4],
                                                [2, 5, 6, 2],
                                                [1, 1, 1, 2]):
                data = dataset.read_all_ts(gpis)
                if gpis == 1:
                    nptest.assert_array_equal(
                        data['test'], np.concatenate([np.arange(n_data),
                                                      np.arange(n_data)]))
                else:
                    nptest.assert_array_equal(data['test'], np.arange(n_data))
                test_dates = []
                base = datetime(2007, base_month, n_data)
                test_dates.append(
                    np.array([base + timedelta(hours=i)
                              for i in range(n_data)]))
                if gpis == 1:
                    base = datetime(2007, 2, n_data)
                    test_dates.append(
                        np.array([base + timedelta(hours=i)
                                  for i in range(n_data)]))

                dates = np.concatenate(test_dates)
                nptest.assert_array_equal(data['time'], dates)

    def test_file_writing_multiple_points_at_once_two_steps_recarray_input(self):
        """
        Write multiple points at once. Add more during an append. Use record arrays as input.
        """
        with BufferedIndexedRaggedTs(self.testfilename, n_loc=4,
                             mode='w') as dataset:
            locations = np.array([1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3])
            data = {'test': np.concatenate([np.arange(2),
                                            np.arange(5),
                                            np.arange(6)])}
            dates = []
            for n_data in [2, 5, 6]:
                base = datetime(2007, 1, n_data)
                dates.append(np.array([base + timedelta(hours=i)
                                       for i in range(n_data)]))
            dates = np.concatenate(dates)
            data = np.array(data['test'], dtype={'names': ['test'],
                                                 'formats': ['i8']})
            dataset.write_ts(locations, data, dates,
                             loc_descr=['first station'] * 13,
                             lon=locations, lat=locations,
                             alt=locations)

        with BufferedIndexedRaggedTs(self.testfilename, n_loc=4,
                             mode='a') as dataset:
            locations = np.array([1, 1, 4, 4])
            data = {'test': np.concatenate([np.arange(2),
                                            np.arange(2)])}
            dates = []
            for n_data in [2, 2]:
                base = datetime(2007, 2, n_data)
                dates.append(np.array([base + timedelta(hours=i)
                                       for i in range(n_data)]))
            dates = np.concatenate(dates)

            data = np.array(data['test'], dtype={'names': ['test'],
                                                 'formats': ['i8']})
            dataset.write_ts(locations, data, dates,
                             loc_descr=['first station'] * 4,
                             lon=locations, lat=locations,
                             alt=locations)

        with BufferedIndexedRaggedTs(self.testfilename) as dataset:
            for gpis, n_data, base_month in zip([1, 2, 3, 4],
                                                [2, 5, 6, 2],
                                                [1, 1, 1, 2]):
                data = dataset.read_all_ts(gpis)
                if gpis == 1:
                    nptest.assert_array_equal(
                        data['test'], np.concatenate([np.arange(n_data),
                                                      np.arange(n_data)]))
                else:
                    nptest.assert_array_equal(data['test'], np.arange(n_data))
                test_dates = []
                base = datetime(2007, base_month, n_data)
                test_dates.append(
                    np.array([base + timedelta(hours=i)
                              for i in range(n_data)]))
                if gpis == 1:
                    base = datetime(2007, 2, n_data)
                    test_dates.append(
                        np.array([base + timedelta(hours=i)
                                  for i in range(n_data)]))

                dates = np.concatenate(test_dates)
                nptest.assert_array_equal(data['time'], dates)

    def test_unlim_loc_file_writing(self):

        with BufferedIndexedRaggedTs(self.testfilename, mode='w') as dataset:
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

        with BufferedIndexedRaggedTs(self.testfilename) as dataset:
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

        with BufferedIndexedRaggedTs(self.testfilename, n_loc=3,
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

        with BufferedIndexedRaggedTs(self.testfilename) as dataset:
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