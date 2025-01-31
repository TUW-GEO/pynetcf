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
Array representation of time series defined by CF conventions.
https://cfconventions.org
"""

from typing import Sequence
from pathlib import Path

import numpy as np
import xarray as xr


def verify_indexed_ragged(ds: xr.Dataset, index_var: str,
                          sample_dim: str) -> None:
    """
    Verify dataset follows indexed ragged array CF defintion.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset.
    index_var : str
        The index variable can be identified by having an attribute with
        name of instance_dimension whose value is the instance dimension.
    sample_dim : str
        Name of the sample dimension.
    """
    if index_var not in ds:
        raise RuntimeError(f"Index variable is missing: {index_var}")

    if ds[index_var].dims != (sample_dim, ):
        raise RuntimeError(f"Index variable ({index_var}) must have the "
                           f"sample dimension ({sample_dim}) as its "
                           "single dimension")

    if "instance_dimension" not in ds[index_var].attrs:
        raise RuntimeError(f"Index variable {index_var} has no "
                           "instance_dimension attribute")


def verify_contiguous_ragged(ds: xr.Dataset, count_var: str,
                             instance_dim: str) -> None:
    """
    Verify dataset follows contiguous ragged array CF defintion.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to be verified.
    count_var : str
        Name of the count variable.
        The count variable contains the length of each time series feature.
        It is identified by having an attribute with name 'sample_dimension'
        whose value is name of the sample dimension. The count variable
        implicitly partitions into individual instances all variables that
        have the sample dimension.

    Returns
    -------
    sample_dimension : str
        Name of the sample dimension.
    """
    if count_var not in ds:
        raise RuntimeError(f"Count variable is missing: {count_var}")

    if "sample_dimension" not in ds[count_var].attrs:
        raise RuntimeError(f"Count variable ({count_var}) has no "
                           "sample_dimension attribute")

    if ds[count_var].dims != (instance_dim, ):
        raise RuntimeError(f"Count variable ({count_var}) must have the "
                           f"instance dimension ({instance_dim}) as its "
                           "single dimension")


def verify_ortho_multi(ds: xr.Dataset, instance_dim: str,
                       element_dim: str) -> None:
    """
    Verify dataset follows contiguous ragged array CF defintion.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to be verified.
    instance_dim : str
        Name of the instance dimension.
    element_dim : str
        Name of the element dimension.

    Returns
    -------
    sample_dimension : str
        Name of the sample dimension.
    """
    if instance_dim not in ds.dims:
        raise RuntimeError(f"Instance dimension is missing: {instance_dim}")

    if element_dim not in ds.dims:
        raise RuntimeError(f"Element dimension is missing: {element_dim}")


def verify_point_array(ds: xr.Dataset) -> None:
    """
    Verify dataset follows point array CF defintion.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset.
    """
    pass


class IndexedRaggedArray:
    """
    Indexed ragged array representation (CF convention).

    In an indexed ragged array representation, the dataset is structured
    to store variable-length data (e.g., time series with varying lengths)
    compactly. To achieve this, auxiliary indexing variables that map the
    flat array storage to meaningful groups (e.g. locations).

    If the instance dimension exists as a variable, it is assumed that the
    values represent the identfiers for each instance otherwise they counting
    upwards from 0.

    Attributes
    ----------
    index_var : str
        The indexed ragged array representation must contain an index
        variable, which must be an integer type, and must have the sample
        dimension as its single dimension.
        The index variable can be identified by having an attribute
        'instance_dimension' whose value is the instance dimension.
    sample_dim : str
        Name of the sample dimension. The sample dimension indicates the
        number of instances (e.g. stations, locations).
    instance_dim : str
        The name of the instance dimension. The value is defined by
        the 'instance_dimension' attribute, which must be present on the
        index variable. All variables having the instance dimension are
        instance variables, i.e. variables holding time series data.
    ds : xarray.Dataset
        Indexed ragged array dataset.
    instance_variables : list
        List of instance variables.
    instance_ids : list
        List of instance ids.

    Methods
    -------
    sel_instance(i)
        Read single time series returning a xr.Dataset.
    sel_instances(i)
        Read multiple time series returing a new IndexedRaggedArray.
    iter()
        Yield time series for each instance.
    save(filename)
        Save dataset on disk.
    from_file(filename, count_var, instance_dim)
        Read dataset from disk.
    to_contiguous()
        Convert indexed ragged array to contiguous ragged array representation.
    to_orthomulti()
        Convert indexed ragged array to incomplete orthogonal multidimensional
        representation.
    to_point_data()
        Convert indexed ragged array to point data array.
    apply(func)
        Apply function on each instance.
    append(ds)
        Append indexed ragged array to existing indexed ragged array.
    """

    def __init__(self, ds: xr.Dataset, index_var: str,
                 sample_dim: str) -> None:
        """
        Initialize.

        Parameters
        ----------
        ds : xr.Dataset
            Data in indexed ragged array structure.
        index_var : str
            Index variable name.
        sample_dim : str
            Sample dimension name.
        """
        verify_indexed_ragged(ds, index_var, sample_dim)

        self.index_var = index_var
        self.sample_dim = sample_dim
        self.instance_dim = ds[index_var].attrs["instance_dimension"]

        self._data = ds.set_coords(self.index_var).set_xindex(self.index_var)
        self._instance_lut = None
        self._lut = None

        self._set_instance_lut()

    def __repr__(self) -> str:
        """"""
        return self.ds.__repr__()

    def _set_instance_lut(self) -> None:
        """
        Set instance lookup-table.
        """
        if self.instance_dim in self.ds:
            instance_ids = self.ds[self.instance_dim].to_numpy()
            self._instance_lut = np.zeros(instance_ids.max() + 1,
                                          dtype=np.int64) - 1
            self._instance_lut[instance_ids] = np.arange(instance_ids.size)
        else:
            instance_ids = np.unique(self.ds[self.index_var])
            self._instance_lut = np.arange(instance_ids.size)

        self._lut = np.zeros(instance_ids.max() + 1, dtype=bool)

    @classmethod
    def from_file(cls, filename: str, index_var: str, sample_dim: str):
        """
        Read data from file.

        Parameters
        ----------
        filename : str
            Filename.
        index_var : str
            Index variable name.
        sample_dim : str
            Sample dimension name.

        Returns
        -------
        data : IndexRaggedArray
            IndexRaggedArray object loaded from a file.
        """
        ds = xr.open_dataset(filename)
        verify_indexed_ragged(ds, index_var, sample_dim)

        return cls(ds, sample_dim, index_var)

    def save(self, filename: str) -> None:
        """
        Write data to file.

        Parameters
        ----------
        filename : str
            Filename.
        """
        suffix = Path(filename).suffix

        if suffix == ".nc":
            self.ds.to_netcdf(filename)
        elif suffix == ".zarr":
            self.ds.to_zarr(filename)
        else:
            raise ValueError(f"Unknown file suffix '{suffix}' "
                             "(.nc and .zarr supported)")

    @property
    def ds(self) -> xr.Dataset:
        """
        Dataset.

        Returns
        -------
        ds : xr.Dataset
            Indexed ragged array dataset.
        """
        return self._data

    @property
    def size(self) -> list:
        """
        Number of instances.

        Returns
        -------
        instance_ids : int
            Number of instance.
        """
        return self.instance_ids.size

    @property
    def instance_ids(self) -> list:
        """
        Instance ids.

        Returns
        -------
        instance_ids : list of int
            Instance ids.
        """
        return self.ds[self.instance_dim].values

    @property
    def instance_variables(self) -> list:
        """
        Instance variables.

        Returns
        -------
        instance_variables : list of str
            Instance variables.
        """
        return [
            var for var in self.ds.variables if (self.ds[var].dims == (
                self.sample_dim, )) and (var != self.sample_dim)
        ]

    def sel_instance(self, i: int) -> xr.Dataset:
        """
        Read time series.

        Parameters
        ----------
        i : int
            Instance identifier.

        Returns
        -------
        ds : xr.Dataset
            Time series for instance.
        """
        data = self.ds.sel({
            self.index_var: self._instance_lut[i],
            self.instance_dim: i
        })

        # reset index variable or drop index variable (my preference)?
        data[self.index_var] = (self.sample_dim,
                                np.zeros(data[self.index_var].size, dtype=int))

        return data

    def sel_instances(self, i: np.array, ignore_missing: bool = True):
        """
        Select multiple instances (time series).

        Parameters
        ----------
        i : numpy.array
            Instance identifier.

        Returns
        -------
        ds : xr.Dataset
            Time series for instance.
        """
        # check for duplicates in instance identifier?
        i = np.asarray(i)

        # check if any missing instance ids have been selected
        if ignore_missing:
            valid = np.where(self._instance_lut[i] != -1)[0]
            if valid.size == 0:
                raise ValueError("No valid instances selected")
        else:
            if np.any(self._instance_lut[np.asarray(i)] == -1):
                raise ValueError("Missing instances selected")
            else:
                valid = np.ones(i.size, dtype=bool)

        i = i[valid]

        # initialize look-up table
        self._lut[self._instance_lut[i]] = True

        data = self.ds.sel(
            {self.index_var: self._lut[self.ds[self.index_var]]})

        # reset index variable
        index = data[self.instance_dim][data[self.index_var]].to_numpy()
        lut = np.zeros(index.max() + 1, dtype=np.int64)
        # n_inst = self.ds.sel({self.instance_dim: i}).sizes[self.instance_dim]
        n_inst = i.size
        lut[i] = np.arange(n_inst)
        data[self.index_var] = (self.sample_dim, lut[index])
        data = data.sel({self.instance_dim: i})

        # copy attributes
        data[self.index_var].attrs = self.ds[self.index_var].attrs

        # reset look-up table
        self._lut[:] = False

        return IndexedRaggedArray(data, self.index_var, self.sample_dim)

    def __iter__(self) -> xr.Dataset:
        """
        Iterator over instances.

        Returns
        -------
        ds : xr.Dataset
            Time series for instance.
        """
        for i in self.instance_ids:
            yield self.sel_instance(i)

    def iter(self) -> xr.Dataset:
        """
        Explicit iterator method.

        Returns
        -------
        ds : xr.Dataset
            Time series for instance.
        """
        return self.__iter__()

    def to_contiguous(self, count_var: str = "row_size"):
        """
        Convert to contiguous ragged array.

        Parameters
        ----------
        count_var : str, optional
            Count variable (default: "row_size").

        Returns
        -------
        data : ContiguousRaggedArray
            Contiguous ragged array time series.
        """
        ds = indexed_to_contiguous(self.ds, self.sample_dim, self.instance_dim,
                                   count_var, self.index_var)

        return ContiguousRaggedArray(ds, self.instance_dim, self.instance_dim,
                                     count_var)

    def to_orthomulti(self):
        """
        Convert to orthogonal multidimensional array.

        Returns
        -------
        data : OrthoMultiArray
            Orthogonal multidimensional array time series.
        """
        ds = self.ds.sortby([self.index_var])
        _, row_size = np.unique(ds[self.index_var], return_counts=True)

        # size defined by longest time series
        element_len = row_size.max()
        instance_len = ds.sizes[self.instance_dim]

        # compute matrix coordinates
        x = np.arange(row_size.size).repeat(row_size)
        y = vrange(np.zeros_like(row_size), row_size)
        shape = (instance_len, element_len)

        reshaped_ds = xr.Dataset(
            {
                var: ([self.instance_dim, self.sample_dim
                       ], pad_to_2d(ds[var], x, y, shape))
                for var in ds.data_vars
            },
            coords={
                self.instance_dim: ds[self.instance_dim],
            },
        )

        return OrthoMultiArray(reshaped_ds, self.instance_dim, self.sample_dim)

    def to_point_data(self):
        """
        Convert to point data.

        Returns
        -------
        data : PointData
            Point data.
        """
        pass

    def apply(self, func):
        """
        Apply function on each instance.

        Parameters
        ----------
        func : Function
            Function to be mapped on each instance.

        Returns
        -------
        data : xr.Dataset
            Result.
        """
        return self.ds.groupby(self.sample_dim).map(func)

    def append(self, ds: xr.Dataset) -> None:
        """
        Append indexed ragged array time series to existing time series.

        Parameters
        ----------
        ds : xarray.Dataset
            Indexed ragged array time series.
        """
        verify_indexed_ragged(ds, self.index_var, self.sample_dim)

        # use instance_id in index variable if instance dimension is a variable
        if self.instance_dim in self.ds:
            self.ds[self.index_var] = self.ds[self.instance_dim][self.ds[
                self.index_var]]

        if self.instance_dim in ds:
            # assume this has been set or not?
            ds = ds.set_coords(self.index_var).set_xindex(self.index_var)
            ds[self.index_var] = (
                self.sample_dim,
                ds[self.instance_dim].values[ds[self.index_var]])

        self._data = xr.combine_nested([self._data, ds], self.sample_dim)

        if self.instance_dim in self.ds:
            instance_ids = self.ds[self.instance_dim].to_numpy()
            lut = np.zeros(instance_ids.max() + 1, dtype=np.int64)
            lut[instance_ids] = np.arange(self.ds[self.instance_dim].size)
            self.ds[self.index_var] = (self.sample_dim,
                                       lut[self.ds[self.index_var]])

        self._set_instance_lut()


class ContiguousRaggedArray:
    """
    Contiguous ragged array representation (CF convention).

    In an contiguous ragged array representation, the dataset for all time
    series are stored in a single 1D array. Additional variables or
    dimensions provide the metadata needed to map these values back to their
    respective time series.

    The contiguous ragged array representation can be used only if the size
    of each instance is known at the time it is created.
    In this representation, the data for each instance will be stored
    contiguously on disk.

    If the instance dimension exists as a variable (e.g. location_id/gpi),
    it is assumed that the values represent the identfiers for each instance
    otherwise the index is a simple counter starting at 0.

    Attributes
    ----------
    instance_dim : str
        Name of the instance dimension.
    sample_dim : str
        Name of the sample dimension. The variable bearing the
        sample_dimension attribute (i.e. count_var) must have the instance
        dimension as its single dimension, and must have an integer type.
    count_var : str
        Name of the count variable. The count variable must be an integer
        type and must have the instance dimension as its sole dimension.
        The count variable are identifiable by the presence of an attribute,
        sample_dimension, found on the count variable, which names the sample
        dimension being counted.
    ds : xarray.Dataset
        Contiguous ragged array dataset.
    instance_variables : list
        List of instance variables.
    instance_ids : list
        List of instance ids.

    Methods
    -------
    sel_instance(i)
        Read single time series returning a xr.Dataset.
    sel_instances(i)
        Read multiple time series returing a new IndexedRaggedArray.
    iter()
        Yield time series for each instance.
    save(filename)
        Save dataset on disk.
    from_file(filename, count_var, instance_dim)
        Read dataset from disk.
    to_indexed()
        Convert contiguous ragged array to indexed ragged array representation.
    to_orthomulti()
        Convert contiguous ragged array to incomplete orthogonal multidimensional
        representation.
    to_point_data()
        Convert contiguous ragged array to point data array.
    apply(func)
        Apply function on each instance.
    """

    def __init__(self, ds: xr.Dataset, count_var: str,
                 instance_dim: str) -> None:
        """
        Initialize.

        Parameters
        ----------
        ds : xr.Dataset
            Data stored in contiguous ragged array foramt.
        count_var : str
            Count variable name.
        instance_dim : str
            Instance dimension name.
        """
        verify_contiguous_ragged(ds, count_var, instance_dim)

        self.count_var = count_var
        self.instance_dim = instance_dim
        self.sample_dim = ds[count_var].attrs["sample_dimension"]
        self._data = ds
        self._lut = None

        # cache row_size and instance_ids data
        self._row_size = self.ds[self.count_var].to_numpy()
        self._instance_ids = self.ds[self.instance_dim].to_numpy()
        self._set_instance_lut()

    def _set_instance_lut(self) -> None:
        """
        Set instance lookup-table.
        """
        self._instance_lookup = np.zeros(
            self._instance_ids.max() + 1,
            dtype=np.int64) + self._instance_ids.size
        self._instance_lookup[self._instance_ids] = np.arange(
            self._instance_ids.size)

        self._lut = np.zeros(self._instance_ids.max() + 1, dtype=bool)

    @classmethod
    def from_file(cls, filename: str, count_var: str, instance_dim: str):
        """
        Load data from file.

        Parameters
        ----------
        filename : str
            Filename.
        count_var : str
            Count variable name.
        instance_dim : str
            Instance dimension name.

        Returns
        -------
        data : ContiguousRaggedArray
            ContiguousRaggedArray object loaded from a file.
        """
        ds = xr.open_dataset(filename)
        verify_contiguous_ragged(ds, count_var, instance_dim)

        return cls(ds, count_var, instance_dim)

    @property
    def ds(self):
        """
        Dataset.

        Returns
        -------
        ds : xr.Dataset
            Contiguous ragged array dataset.
        """
        return self._data

    @property
    def size(self) -> list:
        """
        Number of instances.

        Returns
        -------
        instance_ids : int
            Number of instance.
        """
        return self._instance_ids.size

    @property
    def instance_ids(self) -> list:
        """
        Instance ids

        Returns
        -------
        instance_ids : list of int
            Instance ids.
        """
        return self._instance_ids

    @property
    def instance_variables(self) -> list:
        """
        Instance variables.

        Returns
        -------
        instance_variables : list of str
            Instance variables.
        """
        return [
            var for var in self.ds.variables if (self.ds[var].dims == (
                self.sample_dim, )) and (var != self.sample_dim)
        ]

    def sel_instance(self, i: int) -> xr.Dataset:
        """Read time series"""
        try:
            idx = self._instance_lookup[i]
        except IndexError:
            data = None
        else:
            if idx == self._instance_ids.max() + 1:
                data = None
            else:
                start = self._row_size[:idx].sum()
                end = start + self._row_size[idx]
                data = self.ds.isel({
                    self.sample_dim: slice(start, end),
                    self.instance_dim: idx
                })

        return data

    def sel_instances(self, i: np.array):
        """
        Read time series.

        Parameters
        ----------
        i : numpy.array
            Instance identifier.

        Returns
        -------
        ds : xr.Dataset
            Time series for instance.
        """
        obs = np.repeat(np.arange(self.ds[self.count_var].size),
                        self.ds[self.count_var])

        self._lut[self._instance_lookup[np.asarray(i)]] = True
        data = self.ds.sel({
            self.sample_dim: self._lut[obs],
            self.instance_dim: i
        })
        self._lut[:] = False

        return data

    def __iter__(self) -> xr.Dataset:
        """
        Iterator over instances.

        Returns
        -------
        ds : xr.Dataset
            Time series for instance.
        """
        for i in self.instance_ids:
            yield self.sel_instance(i)

    def iter(self) -> xr.Dataset:
        """
        Explicit iterator method.

        Returns
        -------
        ds : xr.Dataset
            Time series for instance.
        """
        return self.__iter__()

    def to_indexed(self):
        """
        Convert to indexed ragged array.

        Returns
        -------
        data : IndexedRaggedArray
            Indexed ragged array time series.
        """
        ds = contiguous_to_indexed(self.ds, self.sample_dim, self.instance_dim,
                                   self.count_var, self.sample_dim)

        return IndexedRaggedArray(ds, self.sample_dim, self.sample_dim)

    def to_orthomulti(self):
        """
        Convert to orthogonal multidimensional array.

        Returns
        -------
        data : OrthoMultiArray
            Orthogonal multidimensional array time series.
        """
        pass

    def to_point_data(self):
        """
        Convert to point data.

        Returns
        -------
        data : PointData
            Point data.
        """
        pass

    def apply(self, func):
        """
        Apply function on each instance.

        Parameters
        ----------
        func : Function
            Function to be mapped on each instance.

        Returns
        -------
        data : xr.Dataset
            Result.
        """
        return self.ds.groupby(self.sample_dim).map(func)


class OrthoMultiArray:
    """
    Orthogonal multidimensional array.
    """

    def __init__(self,
                 ds: xr.Dataset,
                 instance_dim: str = "loc",
                 element_dim: str = "time") -> None:

        verify_ortho_multi(ds, instance_dim, element_dim)

        self.instance_dim = instance_dim
        self.element_dim = element_dim
        self._data = ds

        # if instance_dim exists as a variable, it is assumed these are there
        # instance identifiers used as keys/reading
        if self.instance_dim in ds:
            self._instances = ds[self.instance_dim].to_numpy()
            self._instance_lookup = np.zeros(
                self._instances.max() + 1,
                dtype=np.int64) + self._instances.size
            self._instance_lookup[self._instances] = np.arange(
                self._instances.size)
        else:
            self._instances = np.arange(ds.sizes[instance_dim])
            self._instance_lookup = np.arange(self._instances.size)

    @classmethod
    def from_file(cls, filename: str, instance_dim: str, element_dim: str):
        """
        Load data from file.

        Parameters
        ----------
        filename : str
            Filename.
        instance_dim : str
            Instance variable name.
        element_dim : str
            Element dimension name.

        Returns
        -------
        data : ContiguousRaggedArray
            ContiguousRaggedArray object loaded from a file.
        """
        ds = xr.open_dataset(filename)
        verify_ortho_multi(ds, instance_dim, element_dim)
        return cls(ds, instance_dim, element_dim)

    @property
    def ds(self):
        return self._data

    def sel_instance(self, instance_id: int):
        """Read time series"""
        return self.ds.sel({self.instance_dim: instance_id})

    def __iter__(self):
        """
        Iterator over instances.

        Returns
        -------
        ds : xr.Dataset
            Time series for instance.
        """
        for instance in self.ds[self.instance_dim]:
            yield self.sel_instance(instance)

    def iter(self):
        """
        Explicit iterator method.

        Returns
        -------
        ds : xr.Dataset
            Time series for instance.
        """
        return self.__iter__()

    def to_point_data(self):
        pass

    def to_indexed(self):
        pass

    def to_contiguous(self):
        pass

    def apply(self, func):
        pass


class PointData:
    """
    Point data represent scattered locations and times with no implied
    relationship among of coordinate positions, both data and coordinates must
    share the same (sample) instance dimension.
    """

    def __init__(self, ds: xr.Dataset) -> None:
        """
        Initialize.
        """
        verify_point_array(ds)
        self._data = ds

    @property
    def ds(self) -> xr.Dataset:
        return self._data

    def to_indexed(self):
        pass

    def to_contiguous(self):
        pass

    def to_ortho_multi(self):
        pass


def vrange(starts: np.array, stops: np.array) -> np.array:
    """
    Create concatenated ranges of integers for multiple start/stop values.

    Parameters
    ----------
    starts : list or numpy.ndarray
        Starts for each range.
    stops : list or numpy.ndarray
        Stops for each range (same shape as starts).

    Returns
    -------
    ranges : numpy.ndarray
        Concatenated ranges.

    Example
    -------
        >>> starts = [3, 4]
        >>> stops  = [5, 7]
        >>> vrange(starts, stops)
        array([3, 4, 4, 5, 6])
    """
    starts = np.asarray(starts)
    stops = np.asarray(stops)
    l = stops - starts  # lengths of each range
    return np.repeat(stops - l.cumsum(), l) + np.arange(l.sum())


def pad_to_2d(var: xr.DataArray, x: np.array, y: np.array,
              shape: tuple) -> np.array:
    """
    Pad each time series

    Parameters
    ----------
    var : xarray.DataArray
        1d array to be converted into 2d array.
    x : np.array
        Row indices.
    y : np.array
        Column indices.
    shape : tuple
        Array shape.

    Returns
    -------
    padded : numpy.array
        Padded 2d array.
    """
    padded = np.zeros(shape, dtype=var.dtype)
    padded[x, y] = var.values

    return padded


def indexed_to_contiguous(
        ds: xr.Dataset,
        sample_dim: str,
        instance_dim: str,
        count_var: str,
        index_var: str,
        sort_vars: Sequence[str] | None = None) -> xr.Dataset:
    """
    Convert an indexed ragged array dataset to a contiguous ragged array
    dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset in indexed ragged array representation.
    sample_dim : str
        Name of the sample dimension.
    instance_dim : str
        The name of the instance dimension.
    count_var : str
        Name of the count variable.
    index_var : str
        Index variable.
    sort_vars : list of str, optional
        Variable sort order.

    Returns
    -------
    ds : xarray.Dataset
        Dataset in contiguous ragged array representation.
    """
    sort_vars = sort_vars or []

    ds = ds.sortby([index_var, *sort_vars])
    idxs, sizes = np.unique(ds[index_var], return_counts=True)

    row_size = np.zeros_like(ds[instance_dim].data)
    row_size[idxs] = sizes
    ds = ds.assign({
        count_var: (instance_dim, row_size, {
            "sample_dimension": sample_dim
        })
    }).drop_vars([index_var])

    return ds


def indexed_to_point(ds: xr.Dataset, sample_dim: str, instance_dim: str,
                     index_var: str) -> xr.Dataset:
    """
    Convert indexed ragged array to point data array.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset in indexed ragged array representation.
    sample_dim : str
        Name of the sample dimension.
    instance_dim : str
        The name of the instance dimension.
    index_var : str
        Index variable.

    Returns
    -------
    ds : xarray.Dataset
        Dataset in point data array representation.
    """
    instance_vars = [
        var for var in ds.variables if instance_dim in ds[var].dims
    ]
    for instance_var in instance_vars:
        ds = ds.assign({
            instance_var: (
                sample_dim,
                ds[instance_var][ds[index_var]].data,
                ds[instance_var].attrs,
            )
        })
    ds = ds.drop_vars([index_var]).assign_attrs({"featureType": "point"})

    return ds


def contiguous_to_indexed(ds: xr.Dataset, sample_dim: str, instance_dim: str,
                          count_var: str, index_var: str) -> xr.Dataset:
    """
    Convert contiguous ragged array dataset to indexed ragged array dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset in contiguous ragged array representation.
    sample_dim : str
        Name of the sample dimension.
    instance_dim : str
        The name of the instance dimension.
    count_var : str
        Name of the count variable.
    index_var : str
        Index variable.

    Returns
    -------
    ds : xarray.Dataset
        Dataset in indexed ragged array representation.
    """
    row_size = np.where(ds[count_var].data > 0, ds[count_var].data, 0)
    locationIndex = np.repeat(np.arange(row_size.size), row_size)

    ds = ds.assign({
        index_var: (
            sample_dim,
            locationIndex,
            {"instance_dimension": instance_dim},
        )
    }).drop_vars([count_var])

    # put locationIndex as first var
    ds = ds[[index_var] + [var for var in ds.variables if var != index_var]]

    return ds


def contiguous_to_point(ds: xr.Dataset, sample_dim: str, instance_dim: str,
                        count_var: str) -> xr.Dataset:
    """
    Convert indexed ragged array to point data array.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset in indexed ragged array representation.
    sample_dim : str
        Name of the sample dimension.
    instance_dim : str
        The name of the instance dimension.
    count_var : str
        Count variable.

    Returns
    -------
    ds : xarray.Dataset
        Dataset in point data array representation.
    """
    row_size = ds[count_var].values
    ds = ds.drop_vars(count_var)
    instance_vars = [
        var for var in ds.variables if instance_dim in ds[var].dims
    ]
    for instance_var in instance_vars:
        ds = ds.assign({
            instance_var: (
                sample_dim,
                np.repeat(ds[instance_var].values, row_size),
                ds[instance_var].attrs,
            )
        })
    ds = ds.assign_attrs({"featureType": "point"})

    return ds
