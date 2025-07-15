"""Script to get images which contain a particular plot
"""

import csv
import shutil
import os

import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon


def get_plot_poly(shp_fn, plot_id):
    shp = gpd.read_file(shp_fn)
    for idx, row in shp.iterrows():
        if row['plot_ID'] != plot_id:
            continue
        line_coords = list(row['geometry'].coords)
        closed_line = LineString(line_coords + [line_coords[0]])
        polygon = Polygon(closed_line)
    return polygon


def get_img_fps(calip_fp, shp_fp, plot_id, d_size_m=1):
    img_list = []

    plot_poly = get_plot_poly(shp_fp, plot_id)

    with open(calip_fp, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        for r_i, row in enumerate(reader):
            if r_i <2:
                continue
            x = row[1]
            y = row[2]

            s_p = Point((x,y))
            circle = s_p.buffer(d_size_m)

            if plot_poly.intersects(circle):
                img_list.append(row[0])

    return img_list

if __name__ == '__main__':
    calip_fp = "MuST-C/sample_data/MuST-C/images/UAV3-RGB/230615/cam_params.txt"
    shp_fp = "MuST-C/metadata/md_FieldSHP/md_FieldSHP.shp"

    img_list = get_img_fps(calip_fp, shp_fp, 198)
    print(img_list)

    in_dir = "MuST-C/sample_data/MuST-C/images/UAV3-RGB/230615"
    out_dir = "MuST-C/sample_data/MuST-C/images/UAV3-RGB/230615_new"
    for img_fn in img_list: 
        shutil.copy2(
                os.path.join(in_dir, img_fn + ".JPG"),
                os.path.join(out_dir, img_fn + ".JPG"),
                )


