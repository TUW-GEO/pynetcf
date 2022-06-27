# -*- coding: utf-8 -*-

"""
Module description
"""
# TODO:
#   (+) 
#---------
# NOTES:
#   -
import numpy as np
from smecv_grid.grid import SMECV_Grid_v052
import pandas as pd
from .results_manager import NcResultsManager
from pygeogrids.grids import CellGrid


def cci_subgrid_from_cells(cells):
    """
    Generate a subgrid for given cells from the smecv_grid

    Parameters
    ----------
    cells : int, numpy.ndarray
        Cell numbers.

    Returns
    -------
    grid : CellGrid
        Subgrid.
    """
    grid = SMECV_Grid_v052(None)
    subgpis, sublons, sublats = grid.grid_points_for_cell(cells)
    subcells = grid.gpi2cell(subgpis)

    return CellGrid(sublons, sublats, subcells, subgpis,
                    geodatum=grid.geodatum.name, shape=(20,20))

def cell_data_add(cell_data:dict, new_data:dict):
    for k in new_data.keys():
        if k not in cell_data.keys():
            cell_data[k] = np.array([new_data[k]])
        else:
            cell_data[k] = np.vstack([cell_data[k], new_data[k]])
    return cell_data

def create_random_data(ts_template, include_nan=True, cols=2):
    for i in range(cols):
        data = np.random.rand(ts_template.index.size)
        if include_nan:
            mask = np.random.choice([1, 0], data.shape).astype(bool)
            data[mask] = np.nan
        ts_template['var_{}'.format(i)] = data
    return ts_template


vars = ['var_1', 'var_2']
grid = SMECV_Grid_v052('land')
ts_path = r'C:\tmp\ts' # this is what we create and fill

ts_template = pd.DataFrame(index=pd.date_range('1904-01-01', '1904-12-31', freq='D'))
# make sure that all the time series cover the same time period
datetimes = pd.to_datetime(ts_template.index).to_pydatetime()



#for cell in grid.get_cells():
cells=[777]
for cell in cells:
    cgpis, clons, clats = cci_subgrid_from_cells(cell).get_grid_points()
    store_gpis = []

    celldata = {var : np.full([20, 20, 366], np.nan) for var in vars}

    for gpi, lon, lat in zip(cgpis, clons, clats):
        print(gpi)
        (crow, ccol) = grid.subgrid_from_cells(cell).gpi2rowcol(gpi)
        gpidata = create_random_data(ts_template, cols=len(vars))
        #data_dict = gpidata.to_dict('list')
        #cell_data = cell_data_add(cell_data, new_data=data_dict)
        #store_gpis.append(gpi)
        for col in gpidata.columns:
            assert gpidata[col].values.size == 366
            celldata[col][crow, ccol, :] = np.float32(gpidata[col].values)

    writer = NcResultsManager(r"C:\Temp\test_compress\celldata")
    for i in range(366):
        print(i)
        writer.add(str(i), celldata, time=)
    print('done')
    ts_dataset.close()
