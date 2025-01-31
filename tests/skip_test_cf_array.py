# Copyright (c) 2025, TU Wien
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
Test module with array representation of time series defined by CF conventions.
"""

import unittest

import numpy as np
import numpy.testing as nptest
import xarray as xr

from pynetcf.cf_array import vrange
from pynetcf.cf_array import pad_to_2d

from pynetcf.cf_array import IndexedRaggedArray
from pynetcf.cf_array import ContiguousRaggedArray
from pynetcf.cf_array import OrthoMultiArray


def test_vrange():
    """
    Test vrange function.
    """
    starts = [3, 4, 3]
    stops = [5, 6, 4]
    ranges = vrange(starts, stops)
    exp_ranges = np.array([3, 4, 4, 5, 3])

    nptest.assert_array_equal(ranges, exp_ranges)


def test_pad_to_2d():
    """
    Test pad_to_2d function.
    """
    ds = xr.DataArray(data=np.arange(12))

    # e.g. number of locations, max time series length
    shape = (3, 10)
    x = [0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2]
    y = [0, 1, 2, 3, 0, 1, 0, 1, 2, 3, 4, 5]
    padded = pad_to_2d(ds, x, y, shape)

    exp_padded = np.array([[0, 1, 2, 3, 0, 0, 0, 0, 0, 0],
                           [4, 5, 0, 0, 0, 0, 0, 0, 0, 0],
                           [6, 7, 8, 9, 10, 11, 0, 0, 0, 0]])

    nptest.assert_array_equal(padded, exp_padded)


class IndexedRaggedArrayTests(unittest.TestCase):
    """
    Test IndexRaggedArray class.
    """

    def setUp(self):
        """
        Initialize tests.
        """
        pass

    def tearDown(self):
        """
        Clean up tests.
        """
        pass


class ContiguousRaggedArrayTests(unittest.TestCase):
    """
    Test ContiguousRaggedArray class.
    """

    def setUp(self):
        """
        Initialize tests.
        """
        pass

    def tearDown(self):
        """
        Clean up tests.
        """
        pass


class OrthoMultiArrayTests(unittest.TestCase):
    """
    Test OrthoMultiArray class.
    """

    def setUp(self):
        """
        Initialize tests.
        """
        pass

    def tearDown(self):
        """
        Clean up tests.
        """
        pass


if __name__ == "__main__":
    # unittest.main()
    # test_vrange()
    test_pad_to_2d()
