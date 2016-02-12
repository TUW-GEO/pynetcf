import os
import unittest
from tempfile import mkdtemp

import numpy as np
import numpy.testing as nptest

from pynetcf.point_data import PointData, GriddedPointData
import pygeogrids.grids as grids


class NcPointDataTest(unittest.TestCase):

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


class GriddedNcPointDataTest(unittest.TestCase):

    def setUp(self):
        # self.testdatapath = os.path.join(mkdtemp())
        self.testdatapath = os.path.join('/home', 'shahn')
        self.testfilename = os.path.join(self.testdatapath, '0107.nc')
        self.grid = grids.genreg_grid().to_cell_grid()

    def tearDown(self):
        # os.remove(self.testfilename)
        pass

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


if __name__ == "__main__":
    unittest.main()
