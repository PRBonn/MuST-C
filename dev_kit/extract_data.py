"""A script to generate the motivation picture from the data set
"""

import os

import click
import pandas as pd
import rasterio
from rasterio.windows import Window
from PIL import Image
import numpy as np
import laspy
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon


def get_clip_rgb(ortho_fp, location, clip_size):
    c_x, c_y = location
    min_x = c_x - clip_size
    min_y = c_y - clip_size

    max_x = c_x + clip_size
    max_y = c_y + clip_size
    with rasterio.open(ortho_fp) as dataset:
        row1, col1 = dataset.index(min_x, min_y)
        row2, col2 = dataset.index(max_x, max_y)
        min_row = min(row1, row2)
        max_row = max(row1, row2)
        min_col = min(col1, col2)
        max_col = max(col1, col2)
        window = Window(min_col, min_row, max_col - min_col, max_row - min_row)
        clip_array = dataset.read(window=window)
        clip_array = np.transpose(clip_array, (1, 2, 0))
        clip = Image.fromarray(clip_array)
    return clip

def get_clip_ms(ortho_fp, location, clip_size, return_pil=True):
    clip_list = []
    c_x, c_y = location
    min_x = c_x - clip_size
    min_y = c_y - clip_size

    max_x = c_x + clip_size
    max_y = c_y + clip_size
    with rasterio.open(ortho_fp) as dataset:
        image_array = dataset.read()
        row1, col1 = dataset.index(min_x, min_y)
        row2, col2 = dataset.index(max_x, max_y)
        min_row = min(row1, row2)
        max_row = max(row1, row2)
        min_col = min(col1, col2)
        max_col = max(col1, col2)
        window = Window(min_col, min_row, max_col - min_col, max_row - min_row)

        for b in range(dataset.count):
            clip_array = dataset.read(b+1, window=window)

            if return_pil:
                # these are reflectance values so we need to normalise to image space
                clip_array = (clip_array - clip_array.min() ) / clip_array.max() * 255
                clip = Image.fromarray(clip_array.astype(np.uint8))
                clip_list.append(clip)
            else:
                clip_list.append(clip_array)
    return clip_list


def get_date(name):
    if "UGVLMI" in name:
        date_str = name.split("_")[-2]
        side_str = name.split("_")[-1]
        date_str = date_str + "_" + side_str
    else:
        date_str = name.split("_")[0]
    date_str = name.split(".")[0]
    return date_str


def get_orthos(parent_dir, data_pkg_str):
    rasters_dir = os.path.join(parent_dir, "raster_data", data_pkg_str)
    raster_fp_list = []
    date_str_list = []

    for raster_name in os.listdir(rasters_dir):
        if "ortho" in raster_name:
            raster_fp_list.append(os.path.join(rasters_dir, raster_name))
            date_str_list.append(get_date(raster_name))

    return raster_fp_list, date_str_list


def get_plot(location, parent_dir, shp_fn="metadata/md_FieldSHP/md_FieldSHP.shp"):
    shp = gpd.read_file(os.path.join(parent_dir, shp_fn))
    s_p = Point(location)
    for idx, row in shp.iterrows():
        if row['plot_ID'] == 0:
            continue  # large geometry demarking the full field trial
        line_coords = list(row['geometry'].coords)
        closed_line = LineString(line_coords + [line_coords[0]])
        polygon = Polygon(closed_line)
        if polygon.contains(s_p):
            plot = row['plot_ID']
    return plot

def get_pointclouds(parent_dir, data_pkg_str, location=(0,0), shp_fn="metadata/md_FieldSHP/md_FieldSHP.shp"):
    pc_fp_list = []
    date_str_list = []
    pcs_dir = os.path.join(parent_dir, "point_clouds", data_pkg_str)

    if data_pkg_str == "UAV2-Lidar":
        for pc_name in os.listdir(pcs_dir):
            pc_fp_list.append(os.path.join(pcs_dir, pc_name))
            date_str_list.append(get_date(pc_name))
    elif data_pkg_str == "UGV-LMI" or data_pkg_str == "UGV-Ouster":
        # first we need to figure out which plot the location
        plot = get_plot(location, parent_dir, shp_fn)
        for pc_name in os.listdir(pcs_dir):
            if str(plot) in pc_name:
                for mew in os.listdir(os.path.join(pcs_dir, pc_name)):
                    pc_fp_list.append(os.path.join(pcs_dir, pc_name, mew))
                    date_str_list.append(get_date(mew))
    elif data_pkg_str == "UAV1-RGB":
        x, y = location
        # look for the point cloud with the point inside
        for date_str in os.listdir(pcs_dir):
            date_pc_dir = os.path.join(pcs_dir, date_str)
            for las_fn in os.listdir(date_pc_dir):
                with laspy.open(os.path.join(date_pc_dir, las_fn)) as las_file:
                    if las_file.header.point_count > 0:
                        min_x , min_y, min_z = las_file.header.min
                        max_x, max_y, max_z = las_file.header.max

                        if x > min_x and x < max_x and y > min_y and y < max_y:
                            pc_fp_list.append(os.path.join(date_pc_dir, las_fn))
                            date_str_list.append(date_str)
    else:
        print(f"Invalid value: unknown data package for point clouds: {data_pkg_str}")
    return pc_fp_list, date_str_list

def get_clip_las(pc_fp, loc, clip_size, delta_z=0):
    las = laspy.read(pc_fp)
    cx, cy = loc
    min_x = cx-clip_size 
    min_y = cy-clip_size 
    max_x = cx+clip_size 
    max_y = cy+clip_size 
    las = las[las.x > min_x]
    las = las[las.y > min_y]
    las = las[las.x < max_x]
    las = las[las.y < max_y]

    las.z+=delta_z

    return las

def get_clips(parent_dir, POIs_csv, clip_size, output_dir, data_pkg_list, data_type_list):
    POIs_df = pd.read_csv(POIs_csv)

    for data_type, data_pkg_str in zip(data_type_list, data_pkg_list):
        # for raster image data
        if data_type == "raster":
            raster_fp_list, date_str_list = get_orthos(parent_dir, data_pkg_str)
            for raster_fp, date_str in zip(raster_fp_list, date_str_list):
                for index, row in POIs_df.iterrows():
                    if "RGB" in data_pkg_str:
                        img_pil = get_clip_rgb(raster_fp, (row["X"], row["Y"]), clip_size)
                        img_pil.save(os.path.join(output_dir, str(row["crop"])+str(row["ID"])+data_pkg_str+"_"+date_str+".png"))
                    elif "MS" in data_pkg_str:
                        img_pil_list = get_clip_ms(raster_fp, (row["X"], row["Y"]), clip_size)
                        for c, img_pil in enumerate(img_pil_list):
                            img_pil.save(os.path.join(output_dir, str(row["crop"])+str(row["ID"])+data_pkg_str+str(c)+"_"+date_str+".png"))
        # for point cloud data 
        elif data_type == "pointcloud":
            for index, row in POIs_df.iterrows():
                # retreive the correct point cloud based on location
                pc_fp_list, date_str_list = get_pointclouds(parent_dir, data_pkg_str, location=(row["X"], row["Y"]))

                for date_str, pc_fp in zip(date_str_list, pc_fp_list,):
                    if data_pkg_str == "UAV1-RGB":
                        las = get_clip_las(pc_fp, (row["X"], row["Y"]), clip_size, delta_z= 47.5247) #47.4820)  # convert from EGM96 to WGS84
                    else:
                        las = get_clip_las(pc_fp, (row["X"], row["Y"]), clip_size)
                    las.write(os.path.join(output_dir, f"{row['crop']}_{str(row['ID'])}_{data_pkg_str}_{date_str}.las"))
        else:
            print(f"ERROR: Invalid data type: {data_type}. Check the data.csv")


@click.command()
@click.option('--parent_dir', "-p", required=True, help="path to parent directory of the MuST-C dataset")
@click.option('--output_dir', "-o", default=".", help="path to output dir. must exist")
@click.option('--pois_csv', "-i", default="./poi.csv", help="path to csv with center points of the regions of interest ")
@click.option("--clip_size", "-c", default=0.5, help="size of region of interest in meters")
@click.option("--data_csv", "-d", default="./data.csv", help="path to data ")
def main(parent_dir, output_dir, pois_csv, clip_size, data_csv):
    data_df = pd.read_csv(data_csv)
    data_pkg_list = data_df["Package"].to_list()
    data_type_list = data_df["Type"].to_list()
    get_clips(parent_dir, pois_csv, clip_size, output_dir, data_pkg_list, data_type_list)

if __name__ == '__main__':
    main()
