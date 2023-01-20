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
"""
Testing image classes of pynetcf.
"""

import os
import unittest
from datetime import datetime
from tempfile import mkdtemp

import numpy as np
import pynetcf.image as ncdata
import pygeogrids.grids as grids


class ImageStackTests(unittest.TestCase):

    def setUp(self):
        """
        Define test file.
        """
        self.testfilename = os.path.join(mkdtemp(), 'test.nc')
        self.grid = grids.genreg_grid()

    def tearDown(self):
        """
        Delete test file.
        """
        os.remove(self.testfilename)

    def test_io(self):
        """
        Write/read test.
        """
        with ncdata.ImageStack(
                self.testfilename,
                self.grid,
            [datetime(2007, 1, 1), datetime(2007, 1, 2)],
                mode="w") as nc:
            nc[14] = {'variable': [141, 142]}
            nc.write_ts([22, 23], {'variable': [[221, 222], [231, 232]]})

        with ncdata.ImageStack(self.testfilename, self.grid) as nc:
            data = nc[14]
            assert list(data['variable'].values) == [141, 142]
            data = nc[22]
            assert list(data['variable'].values) == [221, 222]


class ArrayStackTests(unittest.TestCase):

    def setUp(self):
        """
        Define test file.
        """
        self.testfilename = os.path.join(mkdtemp(), 'test.nc')
        self.grid = grids.BasicGrid(np.arange(180), np.arange(180) - 90)

    def tearDown(self):
        """
        Delete test file.
        """
        os.remove(self.testfilename)

    def test_io(self):
        """
        Write/read test.
        """
        with ncdata.ArrayStack(
                self.testfilename,
                self.grid,
            [datetime(2007, 1, 1), datetime(2007, 1, 2)],
                mode="w") as nc:
            nc[14] = {'variable': [141, 142]}
            nc.write_ts([22, 23], {'variable': [[221, 222], [231, 232]]})

        with ncdata.ArrayStack(
                self.testfilename, self.grid,
            [datetime(2007, 1, 1), datetime(2007, 1, 2)]) as nc:
            data = nc[14]
            assert list(data['variable'].values) == [141, 142]
            data = nc[22]
            assert list(data['variable'].values) == [221, 222]


if __name__ == "__main__":
    unittest.main()
