"""script to calculate vegeation indices from 
mutlispectral ortho
"""

import numpy as np
from PIL import Image
from extract_data import get_clip_ms
import click


def get_ndvi(clip_list):
    r = clip_list[5]  # Red2
    nir = clip_list[9]  # NIR
    ndvi = (nir - r)/ (nir+r)
    return ndvi

# normalized difference red-edge index
def get_ndre(clip_list):
    re = clip_list[7]  # RedEdge2
    nir = clip_list[9]  # NIR
    ndre = (nir - re)/ (nir+re)
    return ndre

# enhanced vegetation index
def get_evi(clip_list):
    nir = clip_list[9]  # NIR
    r = clip_list[5]  # Red2
    b = clip_list[1]

    num = 2.5 * (nir-r)
    denom = (nir + 6*r -7.5*b) + 1
    evi = num / denom
    return evi

# optimised soil adjusted vegetation index 
def get_osavi(clip_list):
    r = clip_list[5]  # Red2
    nir = clip_list[9]  # NIR
    num  = nir - r
    denom = nir + r + 0.16
    osavi = 1.6 * (num / denom)
    return osavi

def save_vi(np_arr, fn):
    np_max = np_arr.max()
    np_min = np_arr.min()
    np_arr = (np_arr-np_min) / (np_max- np_min) * 255
    pil_img = Image.fromarray(np_arr.astype(np.uint8))
    pil_img.save(fn)
    
@click.command()
@click.option('--ms_ortho_fp', "-p", required=True, help="path to the multispectral raster")
@click.option('-x', required=True, help="UTM x location", type=float)
@click.option('-y', required=True, help="UTM y location", type=float)
@click.option('--clip_size', "-c", default=0.5, help="clip size")
def main(ms_ortho_fp, x, y, clip_size):
    location = (x, y)
    
    clip_list = get_clip_ms(ms_ortho_fp, location, clip_size, return_pil=False)

    ndvi = get_ndvi(clip_list)
    save_vi(ndvi, "ndvi.png")

    ndre = get_ndre(clip_list)
    save_vi(ndre, "ndre.png")

    evi = get_evi(clip_list)
    save_vi(evi, "evi.png")

    osavi = get_osavi(clip_list)
    save_vi(osavi, "osavi.png")

if __name__ == '__main__':
    main()
