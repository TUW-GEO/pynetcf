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

import os
import pytest
import unittest
from tempfile import mkdtemp

import numpy as np
import numpy.testing as nptest
import pygeogrids.grids as grids

from pynetcf.point_data import PointData, GriddedPointData


class PointDataReadWriteTest(unittest.TestCase):

    def setUp(self):
        """
        Define test file.
        """
        self.fn = os.path.join(mkdtemp(), 'test.nc')
        self.fn_not_written = os.path.join(mkdtemp(), 'test_not_written.nc')

    def tearDown(self):
        """
        Delete test file.
        """
        try:
            os.remove(self.fn)
        except OSError:
            pass

    def test_io(self):
        """
        Write/read test.
        """
        loc_ids = np.arange(0, 5)
        data1 = np.arange(5, 10)
        data2 = np.arange(10, 15)

        with PointData(self.fn, mode='w', n_obs=5) as nc:
            for loc_id, d1, d2 in zip(loc_ids, data1, data2):
                nc.write(loc_id, {'var1': np.array(d1), 'var2': np.array(d2)})

        with PointData(self.fn) as nc:
            nptest.assert_array_equal(nc['var1'], range(5, 10))
            assert nc.nc['var1'].filters()['zlib'] == True
            nptest.assert_array_equal(nc['var2'][1], np.array([11]))

    def test_ioerror(self):
        """
        Should raise IOError if the files does not exist.
        """
        with pytest.raises(IOError):
            PointData(self.fn_not_written)


class PointDataAppendTest(unittest.TestCase):

    def setUp(self):
        """
        Define test file.
        """
        self.fn = os.path.join(mkdtemp(), 'test.nc')

    def tearDown(self):
        """
        Delete test file.
        """
        os.remove(self.fn)

    def test_append(self):
        """
        Test appending data.
        """
        with PointData(self.fn, mode='w', n_obs=10) as nc:
            for loc_id, data in zip(range(5), range(5, 10)):
                data = np.array(data)
                if loc_id == 1:
                    nc.write(loc_id, {'var1': data, 'var2': data})
                elif loc_id == 4:
                    nc.write(loc_id, {'var1': data, 'var3': data})
                else:
                    nc.write(loc_id, {'var1': data})

        with PointData(self.fn, mode='a') as nc:
            for loc_id, data in zip(range(5), range(5, 10)):
                data = np.array(data)
                if loc_id == 1:
                    nc.write(loc_id, {'var1': data, 'var2': data})
                elif loc_id == 4:
                    nc.write(loc_id, {'var1': data, 'var3': data})
                else:
                    nc.write(loc_id, {'var1': data})

        with PointData(self.fn) as nc:
            x = np.tile(range(5, 10), 2)
            nptest.assert_array_equal(nc['var1'], x)
            nptest.assert_array_equal(nc.read(1)['var2'], np.array([6, 6]))
            nptest.assert_array_equal(nc.read(4)['var3'], np.array([9, 9]))


class PointDataAppendUnlimTest(unittest.TestCase):

    def setUp(self):
        """
        Define test file.
        """
        self.fn = os.path.join(mkdtemp(), 'test.nc')

    def tearDown(self):
        """
        Delete test file.
        """
        os.remove(self.fn)

    def test_append_unlim(self):
        """
        Test appending to pre - existing point data file with
        unlimited observation dimension.
        """
        with PointData(self.fn, mode='a') as nc:
            for loc_id, data in zip(range(5), range(5, 10)):
                data = np.array(data)
                if loc_id == 1:
                    nc.write(loc_id, {'var1': data, 'var2': data})
                elif loc_id == 4:
                    nc.write(loc_id, {'var1': data, 'var3': data})
                else:
                    nc.write(loc_id, {'var1': data})

        with PointData(self.fn, mode='a') as nc:
            for loc_id, data in zip(range(5), range(5, 10)):
                data = np.array(data)
                if loc_id == 1:
                    nc.write(loc_id, {'var1': data, 'var2': data})
                elif loc_id == 4:
                    nc.write(loc_id, {'var1': data, 'var3': data})
                else:
                    nc.write(loc_id, {'var1': data})

        with PointData(self.fn) as nc:
            x = np.tile(range(5, 10), 2)
            nptest.assert_array_equal(nc['var1'], x)
            nptest.assert_array_equal(nc.read(1)['var2'], np.array([6, 6]))
            nptest.assert_array_equal(nc.read(4)['var3'], np.array([9, 9]))


class PointDataMultiDimRecarrayTest(unittest.TestCase):

    def setUp(self):
        """
        Define test file.
        """
        self.fn = os.path.join(mkdtemp(), 'test.nc')

    def tearDown(self):
        """
        Delete test file.
        """
        os.remove(self.fn)

    def test_io_multi_dim_recarray(self):
        """
        Test support of multi - dimensional arrays using
        numpy.dtype.metadata field from recarray.
        """
        dim_info = {
            'dims': {
                'var': ('obs', 'coef', 'config'),
                'var2': ('obs', 'coef', 'doy')
            }
        }

        data = np.zeros(4,
                        dtype=np.dtype([('var', np.float32, (3, 13)),
                                        ('var2', np.int32, (3, 366))],
                                       metadata=dim_info))

        add_dims = {'coef': 3, 'config': 13, 'doy': 366}

        with PointData(self.fn, mode='w', add_dims=add_dims) as nc:
            nc.write(np.arange(4), data)

        with PointData(self.fn) as nc:
            for loc_id in range(4):
                nptest.assert_array_equal(
                    nc.read(loc_id)['var'], data['var'][0])
                nptest.assert_array_equal(
                    nc.read(loc_id)['var2'], data['var2'][0])


class PointDataMultiDimDictTest(unittest.TestCase):

    def setUp(self):
        """
        Define test file.
        """
        self.fn = os.path.join(mkdtemp(), 'test.nc')

    def tearDown(self):
        """
        Delete test file.
        """
        os.remove(self.fn)

    def test_io_multi_dim_dict(self):
        """
        Test support of multi - dimensional arrays using
        numpy.dtype.metadata field from dictionary.
        """
        dim_info = {'dims': {'var1': ('obs', 'coef', 'config')}}
        data_var = np.zeros((4, 3, 13),
                            dtype=np.dtype(np.float32, metadata=dim_info))

        dim_info = {'dims': {'var2': ('obs', 'coef', 'doy')}}
        data_var2 = np.zeros((4, 3, 366),
                             dtype=np.dtype(np.int32, metadata=dim_info))

        add_dims = {'coef': 3, 'config': 13, 'doy': 366}

        data = {'var1': data_var, 'var2': data_var2}

        with PointData(self.fn, mode='w', add_dims=add_dims) as nc:
            nc.write(np.arange(4), data)


class GriddedPointDataReadWriteTest(unittest.TestCase):

    def setUp(self):
        """
        Define test file.
        """
        self.testdatapath = os.path.join(mkdtemp())
        self.testfilename = os.path.join(self.testdatapath, '0107.nc')
        self.grid = grids.genreg_grid().to_cell_grid()

    def tearDown(self):
        """
        Delete test file.
        """
        os.remove(self.testfilename)

    def test_read_write(self):
        """
        Test writing and reading of gridded PointData.
        """
        nc = GriddedPointData(self.testdatapath,
                              mode='w',
                              grid=self.grid,
                              fn_format='{:04d}.nc')

        loc_ids = [10, 11, 12]
        dataset = [1, 2, 3]

        for loc_id, data in zip(loc_ids, dataset):
            data = np.array(data)
            nc.write(loc_id, {'var1': data})

        nc.close()

        nc = GriddedPointData(self.testdatapath,
                              grid=self.grid,
                              fn_format='{:04d}.nc')

        for i, loc_id in enumerate(loc_ids):
            data = nc.read(loc_id)
            nptest.assert_equal(data['var1'], i + 1)

        nc.close()


class GriddedPointData2PointDataTest(unittest.TestCase):

    def setUp(self):
        """
        Define test file, grid and grid points.
        """
        self.gpis = [10, 11, 12, 10000, 10001, 10002, 20000, 20001, 20002]
        self.grid = grids.genreg_grid().to_cell_grid().subgrid_from_gpis(
            self.gpis)
        self.path = mkdtemp()
        self.fn_global = os.path.join(self.path, 'global.nc')

    def tearDown(self):
        """
        Delete test files.
        """
        os.remove(os.path.join(self.path, '0107.nc'))
        os.remove(os.path.join(self.path, '1464.nc'))
        os.remove(os.path.join(self.path, '2046.nc'))

    def test_read_write(self):
        """
        Test re - writing gridded PointData into single file.
        """
        loc_ids = self.gpis

        with GriddedPointData(self.path,
                              mode='w',
                              grid=self.grid,
                              fn_format='{:04d}.nc') as nc:

            for loc_id in loc_ids:
                nc.write(loc_id, {'var1': np.array(loc_id)})

        with GriddedPointData(self.path, grid=self.grid,
                              fn_format='{:04d}.nc') as nc:

            nc.to_point_data(self.fn_global)

        with PointData(self.fn_global) as nc:
            nptest.assert_equal(nc['var1'][:].sort(), loc_ids.sort())


if __name__ == "__main__":
    unittest.main()
