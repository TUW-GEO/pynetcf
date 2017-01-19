# Copyright (c) 2017, Vienna University of Technology,
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
Testing base classes of pynetcf.
"""

import os
import unittest
from tempfile import mkdtemp

import numpy as np
import numpy.testing as nptest

import pynetcf.base as ncbase


class DatasetTest(unittest.TestCase):

    def setUp(self):
        self.testfilename = os.path.join(mkdtemp(), 'test.nc')

    def tearDown(self):
        os.remove(self.testfilename)

    def test_write_append_read_1D(self):

        with ncbase.Dataset(self.testfilename,
                            file_format='NETCDF4', mode='w') as self.dataset:
            # create unlimited Dimension
            self.dataset.create_dim('dim', None)
            self.dataset.write_var('test', np.arange(15), dim=('dim'))

        with ncbase.Dataset(self.testfilename) as self.dataset:
            data = self.dataset.read_var('test')
            nptest.assert_array_equal(data, np.arange(15))

        with ncbase.Dataset(self.testfilename, mode='a') as self.dataset:
            self.dataset.append_var('test', np.arange(15))

        with ncbase.Dataset(self.testfilename) as self.dataset:
            data = self.dataset.read_var('test')
            nptest.assert_array_equal(
                data, np.concatenate([np.arange(15), np.arange(15)]))

    def test_write_append_read_1D_autmaskscale_enabled(self):

        with ncbase.Dataset(self.testfilename,
                            file_format='NETCDF4', mode='w') as self.dataset:
            # create unlimited Dimension
            self.dataset.create_dim('dim', None)
            self.dataset.write_var('test', np.arange(15), dim=('dim'),
                                   attr={"scale_factor": 0.5})

        with ncbase.Dataset(self.testfilename, autoscale=False) as self.dataset:
            data = self.dataset.read_var('test')
            nptest.assert_array_equal(data, np.arange(15) * 2)

        with ncbase.Dataset(self.testfilename, mode='a') as self.dataset:
            self.dataset.append_var('test', np.arange(15))

        with ncbase.Dataset(self.testfilename) as self.dataset:
            data = self.dataset.read_var('test')
            nptest.assert_array_equal(
                data, np.concatenate([np.arange(15), np.arange(15)]))

    def test_write_append_read_1D_autmaskscale_disabled(self):

        with ncbase.Dataset(self.testfilename,
                            file_format='NETCDF4', mode='w',
                            autoscale=False) as self.dataset:
            # create unlimited Dimension
            self.dataset.create_dim('dim', None)
            self.dataset.write_var('test', np.arange(15), dim=('dim'),
                                   attr={"scale_factor": 0.5})

        with ncbase.Dataset(self.testfilename, autoscale=False) as self.dataset:
            data = self.dataset.read_var('test')
            nptest.assert_array_equal(data, np.arange(15))

        with ncbase.Dataset(self.testfilename, mode='a', autoscale=False) as self.dataset:
            self.dataset.append_var('test', np.arange(15))

        with ncbase.Dataset(self.testfilename, autoscale=False) as self.dataset:
            data = self.dataset.read_var('test')
            nptest.assert_array_equal(
                data, np.concatenate([np.arange(15), np.arange(15)]))

    def test_write_read_2D(self):

        with ncbase.Dataset(self.testfilename,
                            file_format='NETCDF4', mode='w') as self.dataset:
            self.dataset.create_dim('dim1', 15)
            self.dataset.create_dim('dim2', 15)
            self.dataset.write_var(
                'test', np.arange(15 * 15).reshape((15, 15)),
                dim=('dim1', 'dim2'))

        with ncbase.Dataset(self.testfilename) as self.dataset:
            data = self.dataset.read_var('test')
            nptest.assert_array_equal(
                data, np.arange(15 * 15).reshape((15, 15)))

    def test_write_append_2D(self):

        with ncbase.Dataset(self.testfilename,
                            file_format='NETCDF4', mode='w') as self.dataset:
            self.dataset.create_dim('dim1', 15)
            self.dataset.create_dim('dim2', None)
            self.dataset.write_var(
                'test', np.arange(15 * 15).reshape((15, 15)),
                dim=('dim1', 'dim2'))

        with ncbase.Dataset(self.testfilename) as self.dataset:
            data = self.dataset.read_var('test')
            nptest.assert_array_equal(
                data, np.arange(15 * 15).reshape((15, 15)))

        with ncbase.Dataset(self.testfilename, mode='a') as self.dataset:
            self.dataset.append_var('test', np.arange(15).reshape((15, 1)))

        with ncbase.Dataset(self.testfilename) as self.dataset:
            data = self.dataset.read_var('test')
            nptest.assert_array_equal(data, np.hstack(
                [np.arange(15 * 15).reshape((15, 15)),
                 np.arange(15).reshape((15, 1))]))


if __name__ == "__main__":
    unittest.main()
