"""
Testing base classes of pynetcf.
"""

import os
import unittest

import numpy as np
import numpy.testing as nptest
from datetime import datetime, timedelta

from pynetcf.time_series import NcTsBase as ncbase


def curpath():
    pth, _ = os.path.split(os.path.abspath(__file__))
    return pth


class DatasetTest(unittest.TestCase):

    def setUp(self):
        self.testfilename = os.path.join(curpath(), 'data', 'test.nc')

    def tearDown(self):
        os.remove(self.testfilename)

    def test_write_append_read_1D(self):

        with ncbase(self.testfilename, file_format='NETCDF4', mode='w') as self.dataset:
            # create unlimited Dimension
            self.dataset.create_dim('dim', None)
            self.dataset.write_var('test', np.arange(15), dim=('dim'))

        with ncbase(self.testfilename) as self.dataset:
            data = self.dataset.read_var('test')
            nptest.assert_array_equal(data, np.arange(15))

        with ncbase(self.testfilename, mode='a') as self.dataset:
            self.dataset.append_var('test', np.arange(15))

        with ncbase(self.testfilename) as self.dataset:
            data = self.dataset.read_var('test')
            nptest.assert_array_equal(
                data, np.concatenate([np.arange(15), np.arange(15)]))

    def test_write_read_2D(self):

        with ncbase(self.testfilename,
                    file_format='NETCDF4', mode='w') as self.dataset:
            self.dataset.create_dim('dim1', 15)
            self.dataset.create_dim('dim2', 15)
            self.dataset.write_var(
                'test', np.arange(15 * 15).reshape((15, 15)),
                dim=('dim1', 'dim2'))

        with ncbase(self.testfilename) as self.dataset:
            data = self.dataset.read_var('test')
            nptest.assert_array_equal(
                data, np.arange(15 * 15).reshape((15, 15)))

    def test_write_append_2D(self):

        with ncbase(self.testfilename, file_format='NETCDF4', mode='w') as self.dataset:
            self.dataset.create_dim('dim1', 15)
            self.dataset.create_dim('dim2', None)
            self.dataset.write_var(
                'test', np.arange(15 * 15).reshape((15, 15)),
                dim=('dim1', 'dim2'))

        with ncbase(self.testfilename) as self.dataset:
            data = self.dataset.read_var('test')
            nptest.assert_array_equal(
                data, np.arange(15 * 15).reshape((15, 15)))

        with ncbase(self.testfilename, mode='a') as self.dataset:
            self.dataset.append_var('test', np.arange(15).reshape((15, 1)))

        with ncbase(self.testfilename) as self.dataset:
            data = self.dataset.read_var('test')
            nptest.assert_array_equal(data, np.hstack(
                [np.arange(15 * 15).reshape((15, 15)),
                 np.arange(15).reshape((15, 1))]))


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
