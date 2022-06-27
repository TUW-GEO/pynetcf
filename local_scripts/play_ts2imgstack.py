# -*- coding: utf-8 -*-

"""
Module description
"""
# TODO:
#   (+) 
#---------
# NOTES:
#   -

from pynetcf.image import ImageStack
from smecv_grid.grid import SMECV_Grid_v052
import pandas as pd
import numpy as np

def cell_data_add(cell_data:dict, new_data:dict):
    for k in new_data.keys():
        if k not in cell_data.keys():
            cell_data[k] = np.array([new_data[k]])
        else:
            cell_data[k] = np.vstack([cell_data[k], new_data[k]])
    return cell_data

def create_random_data(ts_template, include_nan=True, cols=10):
    for i in range(cols):
        data = np.random.rand(ts_template.index.size)
        if include_nan:
            mask = np.random.choice([1, 0], data.shape).astype(bool)
            data[mask] = np.nan
        ts_template['var_{}'.format(i)] = data
    return ts_template

grid = SMECV_Grid_v052('land')
stackfile = r'C:\tmp\imgstacktest.nc' # this is what we create and fill

ts_template = pd.DataFrame(index=pd.date_range('2000-01-01', '2000-12-31', freq='D'))
# make sure that all the time series cover the same time period
datetimes = pd.to_datetime(ts_template.index).to_pydatetime()
stack = ImageStack(filename=stackfile, grid=grid, times=datetimes,
                   mode='w', name='TESTFILE')

#for cell in grid.get_cells():
cell=777
gpis, lons, lats = grid.grid_points_for_cell(cell)
cell_data = {}
store_gpis = []

for gpi, lon, lat in zip(gpis, lons, lats):
    print(gpi)
    gpidata = create_random_data(ts_template)
    data_dict = gpidata.to_dict('list')
    cell_data = cell_data_add(cell_data, new_data=data_dict)
    store_gpis.append(gpi)

stack.write_ts(store_gpis, cell_data)
print('done')
stack.close()