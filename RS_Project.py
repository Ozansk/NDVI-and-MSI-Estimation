import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import rasterio
import rasterio._shim
import rasterio.crs
import rasterio.control
import rasterio.sample
import rasterio.vrt
import rasterio._features
from rasterio import plot
import glob
import os

temp = 0


def NDVI():
    global e
    global temp
    filepath = e.get()
    os.chdir(filepath)
    rasters = glob.glob('*tif')

    dsRed = rasterio.open(rasters[5])
    bandRed = dsRed.read(1).astype('float32')

    dsNir = rasterio.open(rasters[6])
    bandNir = dsNir.read(1).astype('float32')

    ndvi = np.zeros(dsRed.shape, dtype=rasterio.float32)

    np.seterr(divide='ignore', invalid='ignore')
    ndvi = (bandNir.astype(float) - bandRed.astype(float)) / (bandNir + bandRed)

    kwargs = dsRed.meta
    kwargs.update(
        dtype=rasterio.float32,
        count=1,
        compress='lzw')

    if temp == 0:
        os.mkdir('output')
        temp = 1

    filepath = filepath + '\\output'
    os.chdir(filepath)
    with rasterio.open('NDVI.tif', 'w', **kwargs) as dst:
        dst.write_band(1, ndvi.astype(rasterio.float32))

    filepath = filepath + '\\NDVI.tif'
    NdviBand = rasterio.open(filepath)
    ndvi_band = NdviBand.read(1)

    negatives = -4
    positives = 4

    bounds_min = np.linspace(negatives, 0, 129)
    bounds_max = np.linspace(0, positives, 129)[1:]

    bounds = np.concatenate((bounds_min, bounds_max), axis=None)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)

    num_neg_colors = int(256 / (positives - negatives) * (-negatives))
    num_pos_colors = 256 - num_neg_colors
    cmap_BuRd = plt.cm.jet

    colors_2neg_4pos = [cmap_BuRd(0.5 * c / num_neg_colors) for c in range(num_neg_colors)] + \
                       [cmap_BuRd(1 - 0.5 * c / num_pos_colors) for c in range(num_pos_colors)][::-1]
    cmap_2neg_4pos = colors.LinearSegmentedColormap.from_list('cmap_2neg_4pos', colors_2neg_4pos, N=256)

    ticks = np.append(np.arange(-2.0, 0, 0.20), np.arange(0, 4.001, 0.20))

    plt.figure(figsize=(10, 6))
    plt.axis("off")
    plt.imshow(ndvi_band, cmap=cmap_2neg_4pos)
    plt.colorbar(ticks=ticks)
    plt.show()


def MSI():
    global e
    global temp
    filepath = e.get()
    os.chdir(filepath)
    rasters = glob.glob('*tif')

    dsNir = rasterio.open(rasters[6])
    bandNir = dsNir.read(1).astype('float32')

    dsSix = rasterio.open(rasters[7])
    bandSix = dsSix.read(1).astype('float32')

    msi = np.zeros(dsNir.shape, dtype=rasterio.float32)

    np.seterr(divide='ignore', invalid='ignore')
    msi = np.where(
        bandNir == 0.,
        0.,
        bandSix / bandNir)

    max_value = np.max(msi)

    modified_msi = np.where(
        bandSix / bandNir >= max_value,
        max_value,
        bandSix / bandNir)

    kwargs = dsNir.meta
    kwargs.update(
        dtype=rasterio.float32,
        count=1,
        compress='lzw')

    if temp == 0:
        os.mkdir('output')
        temp = 1

    filepath = filepath + '\\output'
    os.chdir(filepath)
    with rasterio.open('MSI.tif', 'w', **kwargs) as dst:
        dst.write_band(1, modified_msi.astype(rasterio.float32))

    filepath = filepath + '\\MSI.tif'
    MSIBand = rasterio.open(filepath)
    msi_band = MSIBand.read(1)

    negatives = -4
    positives = 4

    bounds_min = np.linspace(negatives, 0, 129)
    bounds_max = np.linspace(0, positives, 129)[1:]

    bounds = np.concatenate((bounds_min, bounds_max), axis=None)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)

    num_neg_colors = int(256 / (positives - negatives) * (-negatives))
    num_pos_colors = 256 - num_neg_colors
    cmap_BuRd = plt.cm.jet

    colors_2neg_4pos = [cmap_BuRd(0.5 * c / num_neg_colors) for c in range(num_neg_colors)] + \
                       [cmap_BuRd(1 - 0.5 * c / num_pos_colors) for c in range(num_pos_colors)][::-1]
    cmap_2neg_4pos = colors.LinearSegmentedColormap.from_list('cmap_2neg_4pos', colors_2neg_4pos, N=256)

    ticks = np.append(np.arange(-2.0, 0, 0.50), np.arange(0, max_value, 0.50))

    plt.figure(figsize=(10, 6))
    plt.axis("off")
    plt.imshow(msi_band, cmap=cmap_2neg_4pos)
    plt.colorbar(ticks=ticks)
    plt.show()


from tkinter import *

root = Tk()
root.geometry("300x250")
root.title('Estimate NDVI and MSI')
Label(root, text="Please enter path of the Landsat-8 directory").pack()

e = Entry(root)
e.pack()
e.focus_set()

Label(text="").pack()

Button(text="NDVI", height="2", width="30", command=NDVI).pack()
Label(text="").pack()
Button(text="MSI", height="2", width="30", command=MSI).pack()

root.mainloop()