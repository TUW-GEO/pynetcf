"""
Testing image classes of pynetcf.
"""

import os
import unittest

import numpy as np
from datetime import datetime

import pynetcf.image as ncdata
import pygeogrids.grids as grids


def curpath():
    pth, _ = os.path.split(os.path.abspath(__file__))
    return pth


class ImageStackTests(unittest.TestCase):

    def setUp(self):
        self.testfilename = os.path.join(curpath(), 'data', 'test.nc')
        self.grid = grids.genreg_grid()

    def tearDown(self):
        os.remove(self.testfilename)

    def test_writing(self):
        with ncdata.ImageStack(self.testfilename, self.grid,
                               [datetime(2007, 1, 1),
                                datetime(2007, 1, 2)], mode="w") as nc:
            nc[14] = {'variable': [141, 142]}
            nc.write_ts([22, 23], {'variable': [[221, 222], [231, 232]]})

        with ncdata.ImageStack(self.testfilename, self.grid) as nc:
            data = nc[14]
            assert list(data['variable'].values) == [141, 142]
            data = nc[22]
            assert list(data['variable'].values) == [221, 222]


class ArrayStackTests(unittest.TestCase):

    def setUp(self):
        self.testfilename = os.path.join(curpath(), 'data', 'test.nc')
        self.grid = grids.BasicGrid(np.arange(180), np.arange(180) - 90)

    def tearDown(self):
        # os.remove(self.testfilename)
        pass

    def test_writing(self):
        with ncdata.ArrayStack(self.testfilename, self.grid,
                               [datetime(2007, 1, 1),
                                datetime(2007, 1, 2)], mode="w") as nc:
            nc[14] = {'variable': [141, 142]}
            nc.write_ts([22, 23], {'variable': [[221, 222], [231, 232]]})

        with ncdata.ArrayStack(self.testfilename, self.grid,
                               [datetime(2007, 1, 1),
                                datetime(2007, 1, 2)]) as nc:
            data = nc[14]
            assert list(data['variable'].values) == [141, 142]
            data = nc[22]
            assert list(data['variable'].values) == [221, 222]
            data = nc[datetime(2007, 1, 1)]

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
