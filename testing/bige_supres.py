import argparse
import numpy as np
import os
import re
import sys
import time
from osgeo import gdal, osr
from collections import defaultdict
from supres import Solver #DSen2_20, DSen2_60

from mlib import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

class BigEarthReader():
    def __init__(self, root):
        self.root = root

        # Checks the existence of required python packages
        self.gdal_existed = self.rasterio_existed = self.georasters_existed = False
        try:
            from osgeo import gdal
            self.gdal_existed = True
            print('INFO: GDAL package will be used to read GeoTIFF files')
        except ImportError:
            try:
                import rasterio
                self.rasterio_existed = True
                print('INFO: rasterio package will be used to read GeoTIFF files')
            except ImportError:
                print('ERROR: please install either GDAL or rasterio package to read GeoTIFF files')

        # Spectral band names to read related GeoTIFF files
        self.band_names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 
                           'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']

    def reset_root(self, root):
        self.root = root

    def check_bands(self, patch_name):
        self._read(patch_name, ret=False)

    def read_file(self, patch_name):
        return self._read(patch_name, ret=True)

    def _read(self, patch_name, ret=False):
        # Reads spectral bands of all patches whose folder names are populated before
        bands = []
        for band_name in self.band_names:
            # First finds related GeoTIFF path and reads values as an array
            band_path = os.path.join(
                self.root, patch_name, patch_name + '_' + band_name + '.tif')
            if self.gdal_existed:
                band_ds = gdal.Open(band_path,  gdal.GA_ReadOnly)
                raster_band = band_ds.GetRasterBand(1)
                band_data = raster_band.ReadAsArray()
            elif self.rasterio_existed:
                band_ds = rasterio.open(band_path)
                band_data = band_ds.read(1)
            if ret:
                bands.append(band_data)
            else:
                # band_data keeps the values of band band_name for the patch patch_name
                print('INFO: band', band_name, 'of patch', patch_name,
                        'is ready with size', band_data.shape)
        if ret:
            return bands

def split_bands(bands):
    d10 = np.array([bands[i] for i in (1,2,3,7)])
    d20 = np.array([bands[i] for i in (4,5,6,8,10,11)])
    d60 = np.array([bands[i] for i in (0,9)])
    return d10.transpose(1,2,0), d20.transpose(1,2,0), d60.transpose(1,2,0)

def ssr(name, breader, solver):
    raster = breader.read_file(name)
    d10, d20, d60 = split_bands(raster)
    sr20 = solver.predict20(d10, d20)
    sr60 = solver.predict60(d10, d20, d60)
    # sr20 = DSen2_20(d10, d20, deep=False)
    # sr60 = DSen2_60(d10, d20, d60, deep=False)
    output = np.vstack((d10.transpose(2,0,1), sr20.transpose(2,0,1), sr60.transpose(2,0,1)))
    output = output[(10,0,1,2,4,5,6,3,7,11,8,9),:,:]
    return output

if __name__ == "__main__":
    in_root = r'/media/miracle/AICO/Satellite/BigEarthNet-v1.0'
    out_root = r'/media/miracle/AICO/Satellite/SRBigEarth'

    breader = BigEarthReader(in_root)
    ori_dir = Path(breader.root)
    folders = glob.glob(breader.root+'/*')
    print('Folder Number:', len(folders))

    solver = Solver()

    with Timer("All Pairs"):
        subset = folders[2480:]
        for idx, folder in enumerate(subset):
            try:
                print(f"\nNumber: {idx}/{len(subset)} Processing...")
                name = Path(folder).name
                result = ssr(name, breader, solver)
                mio.write(result, Path(out_root)/f'{name}.npy')
            except:
                with open('error_log.txt', 'a+') as f:
                    f.write(str(idx)+'\n')