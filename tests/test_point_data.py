import os
import unittest
from tempfile import mkdtemp

import numpy as np
import numpy.testing as nptest

from pynetcf.point_data import PointData, GriddedPointData
import pygeogrids.grids as grids


class PointDataReadWriteTest(unittest.TestCase):

    """
    Test writing and reading PointData.
    """

    def setUp(self):
        self.fn = os.path.join(mkdtemp(), 'test.nc')

    def tearDown(self):
        os.remove(self.fn)

    def test_read_write(self):

        with PointData(self.fn, mode='w', n_obs=5) as nc:
            for loc_id, data in zip(range(5), range(5, 10)):
                if loc_id == 1:
                    nc.write(loc_id, {'var1': data, 'var2': data})
                elif loc_id == 4:
                    nc.write(loc_id, {'var1': data, 'var3': data})
                else:
                    nc.write(loc_id, {'var1': data})

        with PointData(self.fn) as nc:
            nptest.assert_array_equal(nc['var1'], range(5, 10))
            nptest.assert_array_equal(nc['var2'][1], np.array([6]))
            nptest.assert_array_equal(nc['var3'][4], np.array([9]))


class PointDataAppendTest(unittest.TestCase):

    """
    Test appending to pre-existing point data file.
    """

    def setUp(self):
        self.fn = os.path.join(mkdtemp(), 'test.nc')

    def tearDown(self):
        os.remove(self.fn)

    def test_append(self):

        with PointData(self.fn, mode='w', n_obs=10) as nc:
            for loc_id, data in zip(range(5), range(5, 10)):
                if loc_id == 1:
                    nc.write(loc_id, {'var1': data, 'var2': data})
                elif loc_id == 4:
                    nc.write(loc_id, {'var1': data, 'var3': data})
                else:
                    nc.write(loc_id, {'var1': data})

        with PointData(self.fn, mode='a') as nc:
            for loc_id, data in zip(range(5), range(5, 10)):
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

    """
    Test appending to pre-existing point data file with
    unlimited observation dimension.
    """

    def setUp(self):
        self.fn = os.path.join(mkdtemp(), 'test.nc')

    def tearDown(self):
        os.remove(self.fn)

    def test_append_unlim(self):

        with PointData(self.fn, mode='w') as nc:
            for loc_id, data in zip(range(5), range(5, 10)):
                if loc_id == 1:
                    nc.write(loc_id, {'var1': data, 'var2': data})
                elif loc_id == 4:
                    nc.write(loc_id, {'var1': data, 'var3': data})
                else:
                    nc.write(loc_id, {'var1': data})

        with PointData(self.fn, mode='a') as nc:
            for loc_id, data in zip(range(5), range(5, 10)):
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


class PointDataMultiDimTest(unittest.TestCase):

    """
    Test support of multi-dimensional arrays using numpy.dtype.metadata field.
    """

    def setUp(self):
        self.fn = os.path.join(mkdtemp(), 'test.nc')

    def tearDown(self):
        os.remove(self.fn)

    def test_read_write_multi_dim(self):

        dims = {'dims': ('obs', 'coef', 'config')}
        data = np.ones((1, 3, 13), dtype=np.dtype(np.float32, metadata=dims))

        add_dims = {'coef': 3, 'config': None}

        with PointData(self.fn, mode='w', add_dims=add_dims) as nc:
            for loc_id in range(5):
                nc.write(loc_id, {'var1': data, 'var2': 5})

        with PointData(self.fn) as nc:
            for loc_id in range(5):
                nptest.assert_array_equal(nc.read(loc_id)['var1'], data)
                nptest.assert_array_equal(nc.read(loc_id)['var2'], 5)


class GriddedPointDataReadWriteTest(unittest.TestCase):

    """
    Test writing and reading of gridded PointData.
    """

    def setUp(self):
        self.testdatapath = os.path.join(mkdtemp())
        self.testfilename = os.path.join(self.testdatapath, '0107.nc')
        self.grid = grids.genreg_grid().to_cell_grid()

    def tearDown(self):
        os.remove(self.testfilename)

    def test_read_write(self):

        nc = GriddedPointData(self.testdatapath, mode='w', grid=self.grid,
                              fn_format='{:04d}.nc')

        loc_ids = [10, 11, 12]
        dataset = [1, 2, 3]

        for loc_id, data in zip(loc_ids, dataset):
            nc.write(loc_id, {'var1': data})

        nc.close()

        nc = GriddedPointData(self.testdatapath, grid=self.grid,
                              fn_format='{:04d}.nc')

        for i, loc_id in enumerate(loc_ids):
            data = nc.read(loc_id)
            nptest.assert_equal(data['var1'], i + 1)

        nc.close()


class GriddedPointData2PointDataTest(unittest.TestCase):

    """
    Test re-writing gridded PointData into single file.
    """

    def setUp(self):
        self.gpis = [10, 11, 12, 10000, 10001, 10002, 20000, 20001, 20002]
        self.grid = grids.genreg_grid().to_cell_grid().\
            subgrid_from_gpis(self.gpis)
        self.path = mkdtemp()
        self.fn_global = os.path.join(self.path, 'global.nc')

    def tearDown(self):
        os.remove(os.path.join(self.path, '0107.nc'))
        os.remove(os.path.join(self.path, '1464.nc'))
        os.remove(os.path.join(self.path, '2046.nc'))

    def test_read_write(self):

        loc_ids = self.gpis

        with GriddedPointData(self.path, mode='w', grid=self.grid,
                              fn_format='{:04d}.nc') as nc:

            for loc_id in loc_ids:
                nc.write(loc_id, {'var1': loc_id})

        with GriddedPointData(self.path, grid=self.grid,
                              fn_format='{:04d}.nc') as nc:

            nc.to_point_data(self.fn_global)

        with PointData(self.fn_global) as nc:
            nptest.assert_equal(nc['var1'][:].sort(), loc_ids.sort())

if __name__ == "__main__":
    unittest.main()
