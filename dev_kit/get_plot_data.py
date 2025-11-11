"""Script to extract data of specified plots
"""

import os 
import shutil

import click
from PIL import Image
import rasterio
from rasterio.mask import mask
from rasterio.windows import Window
import numpy as np
import tifffile
from shapely.geometry import mapping, Polygon
from shapely.affinity import scale
from shapely import contains, points
import laspy
import gc
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
import open3d as o3d


from get_images_for_a_plot import get_plot_poly, get_img_fps


def get_pc_list(dir_fp, plot_poly):
    pc_list = []
    for pc in os.listdir(dir_fp):
        if pc.split(".")[-1] != "las":
            print(f"skipping {pc} because it is not a las file")
            continue
        with laspy.open(os.path.join(dir_fp, pc)) as pc_las:
            if pc_las.header.point_count == 0:
                continue

            min_x, min_y, _ = pc_las.header.min
            max_x, max_y, _ = pc_las.header.max

        las_poly = Polygon([
            (min_x, min_y),
            (min_x, max_y),
            (max_x, max_y),
            (max_x, min_y),
            ])

        if las_poly.intersects(plot_poly):
            pc_list.append(pc)

    return pc_list 


def get_clip_plot_ms(ortho_fp, plot_poly, out_fp, d_buffer_r=1.1):
    # plot_poly = plot_poly.buffer(d_buffer_m)
    plot_poly = scale(plot_poly, xfact=d_buffer_r, yfact=d_buffer_r, origin='center')
    with rasterio.open(ortho_fp) as dataset:
        geoms = [mapping(plot_poly)]
        out_image, out_transform = mask(dataset, geoms, crop=True)
        out_meta = dataset.meta.copy()

    # Update metadata
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })

    # TODO if the raster is empty, dont write it 
    with rasterio.open(out_fp, "w", **out_meta) as dest:
        dest.write(out_image)


def get_clip_plot_rgb(ortho_fp, plot_poly, out_fp):
    min_x, min_y, max_x, max_y = plot_poly.bounds
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
        if np.issubdtype(clip_array.dtype, np.integer):
            clip = Image.fromarray(clip_array)
            clip.save(out_fp)
        else:  # DEM
            tifffile.imwrite(out_fp, clip_array)


# filter out images which contains the plot
def one_date_images(calip_fp, shp_fp, plot_id, file_ext, data_str, date, parent_dir, output_dir):
    os.makedirs(os.path.join(output_dir, date, f"plot{plot_id}", "images", data_str), exist_ok=True)
    img_list = get_img_fps(calip_fp, shp_fp, plot_id, d_size_m=1) 
    for img_fn in img_list:
        img_fn = img_fn + file_ext
        source_fp = os.path.join(parent_dir, "images", data_str, date, img_fn)
        if os.path.isfile(source_fp):
            shutil.copy2(
                    source_fp,
                    os.path.join(output_dir, f"plot{plot_id}", date, "images", data_str, img_fn)
                )
        else:
            print(f"Skipping {source_fp} since file does not exist")


def check_dir(parent_dir, *args, **kwargs):
    fp = os.path.join(parent_dir, *args)
    is_ok = os.path.isdir(fp)
    if not is_ok:
        print(f"check_dir: skipping {os.path.join(*args)} because the necessary files do not exist in {parent_dir}")
    else:
        if 'output_dir' in kwargs:
            os.makedirs(os.path.join(kwargs['output_dir'], *args), exist_ok=True)
            print(f"Processing {args}")
    return is_ok


@click.command()
@click.option(
    "--parent_dir",
    type=str,
    help="path to the MuST-C dataset",
    required=True
)
@click.option(
    "--output_dir",
    type=str,
    help="path to the generated plot-wise version of the dataset. creates the directory if it does not exist",
    required=True
)
@click.option(
    "--plot_id",
    type=int,
    help="plot id of the plot of interest",
    default=-1,
    required=False
)
@click.option('--uav1_rgb', is_flag=True)
@click.option('--uav2_rgb', is_flag=True)
@click.option('--uav3_rgb', is_flag=True)
@click.option('--uav3_ms', is_flag=True)
@click.option('--ugv_rgb', is_flag=True)
@click.option('--uav2_lidar', is_flag=True)
@click.option('--ugv_lmi', is_flag=True)
@click.option('--ugv_ouster', is_flag=True)
def main(
            parent_dir, 
            output_dir, 
            plot_id, 
            uav1_rgb,
            uav2_rgb,
            uav3_rgb,
            uav3_ms,
            ugv_rgb,
            uav2_lidar,
            ugv_lmi,
            ugv_ouster
        ):
        
        if plot_id == -1:  # no plot specfied, do for all plots 
            items = range(162,242)

            with ThreadPoolExecutor(max_workers=6) as executor:
                executor.map(lambda plot_id: 
                        one_plot(
                                parent_dir, 
                                output_dir, 
                                plot_id, 
                                uav1_rgb,
                                uav2_rgb,
                                uav3_rgb,
                                uav3_ms,
                                ugv_rgb,
                                uav2_lidar,
                                ugv_lmi,
                                ugv_ouster
                            ),
                        items
                        )
            """
            for plot_id in tqdm(range(162,242)):
                one_plot(
                    parent_dir, 
                    output_dir, 
                    plot_id, 
                    uav1_rgb,
                    uav2_rgb,
                    uav3_rgb,
                    uav3_ms,
                    ugv_rgb,
                    uav2_lidar,
                    ugv_lmi,
                    ugv_ouster
                           )
            """
        else:
            one_plot(
                parent_dir, 
                output_dir, 
                plot_id, 
                uav1_rgb,
                uav2_rgb,
                uav3_rgb,
                uav3_ms,
                ugv_rgb,
                uav2_lidar,
                ugv_lmi,
                ugv_ouster
                       )

def one_plot(
            parent_dir, 
            output_dir, 
            plot_id, 
            uav1_rgb,
            uav2_rgb,
            uav3_rgb,
            uav3_ms,
            ugv_rgb,
            uav2_lidar,
            ugv_lmi,
            ugv_ouster,
            is_generate_uav1_rgb = False  # True only for generating plotwise data for the first time in uav1rgb pcs 
        ):
    if not (uav1_rgb or uav2_rgb or uav3_rgb or uav3_ms or ugv_rgb or uav2_lidar or ugv_lmi or ugv_ouster):
        # if no flag specified, then run all
        uav1_rgb = True
        uav2_rgb = True
        uav3_rgb = True 
        uav3_ms = True 
        ugv_rgb = True 
        uav2_lidar = True 
        ugv_lmi = True
        ugv_ouster = True 

    # make the new dir struct 
    os.makedirs(output_dir, exist_ok=True)

    shp_fp = os.path.join(parent_dir, "LAI_biomass_and_metadata", "md_FieldSHP", "md_FieldSHP.shp")
    assert os.path.isfile(shp_fp), f"md_FieldSHP shapefile not found at {shp_fp}"
    # images 
    if check_dir(parent_dir, "images"):
        if uav1_rgb:
            if check_dir(parent_dir, "images", "UAV1-RGB", output_dir=output_dir):
                for date in os.listdir(os.path.join(parent_dir, "images", "UAV1-RGB")):
                    if date == "cam_params":
                        shutil.copytree(
                            os.path.join(parent_dir, "images", "UAV1-RGB", date),
                            os.path.join(output_dir, "images", "UAV1-RGB", date),
                            copy_function=shutil.copy2,
                            dirs_exist_ok=True
                            )
                        continue

                    calip_fp = os.path.join(parent_dir, "images", "UAV1-RGB", "cam_params", date, "cam_params.txt")
                    one_date_images(calip_fp, shp_fp, plot_id, ".tiff", "UAV1-RGB", date, parent_dir, output_dir)

        if uav2_rgb:
            if check_dir(parent_dir, "images", "UAV2-RGB", output_dir=output_dir):
                for date in os.listdir(os.path.join(parent_dir, "images", "UAV2-RGB")):
                    calip_fp = os.path.join(parent_dir, "images", "UAV2-RGB", date, "cam_params.txt")
                    one_date_images(calip_fp, shp_fp, plot_id, ".JPG", "UAV2-RGB", date, parent_dir, output_dir)

                    shutil.copy2(
                            os.path.join(parent_dir, "images", "UAV2-RGB", date, "cam_params.txt"),
                            os.path.join(output_dir, "images", "UAV2-RGB", date, "cam_params.txt"),
                            )

                    shutil.copy2(
                            os.path.join(parent_dir, "images", "UAV2-RGB", date, "cam_params.xml"),
                            os.path.join(output_dir, "images", "UAV2-RGB", date, "cam_params.xml"),
                            )

        if uav3_rgb:
            if check_dir(parent_dir, "images", "UAV3-RGB", output_dir=output_dir):
                for date in os.listdir(os.path.join(parent_dir, "images", "UAV3-RGB")):
                    calip_fp = os.path.join(parent_dir, "images", "UAV3-RGB", date, "cam_params.txt")
                    one_date_images(calip_fp, shp_fp, plot_id, ".JPG", "UAV3-RGB", date, parent_dir, output_dir)

                    shutil.copy2(
                            os.path.join(parent_dir, "images", "UAV3-RGB", date, "cam_params.txt"),
                            os.path.join(output_dir, "images", "UAV3-RGB", date, "cam_params.txt"),
                            )

                    shutil.copy2(
                            os.path.join(parent_dir, "images", "UAV3-RGB", date, "cam_params.xml"),
                            os.path.join(output_dir, "images", "UAV3-RGB", date, "cam_params.xml"),
                            )

        if uav3_ms:
            if check_dir(parent_dir, "images", "UAV3-MS", output_dir=output_dir):
                for date in os.listdir(os.path.join(parent_dir, "images", "UAV3-MS")):
                    if date == "reference_panels":
                        continue
                    calip_fp = os.path.join(parent_dir, "images", "UAV3-MS", date, "cam_params.txt")
                    one_date_images(calip_fp, shp_fp, plot_id, ".tif", "UAV3-MS", date, parent_dir, output_dir)

                    shutil.copy2(
                            os.path.join(parent_dir, "images", "UAV3-MS", date, "cam_params.txt"),
                            os.path.join(output_dir, "images", "UAV3-MS", date, "cam_params.txt"),
                            )

                    shutil.copy2(
                            os.path.join(parent_dir, "images", "UAV3-MS", date, "cam_params.xml"),
                            os.path.join(output_dir, "images", "UAV3-MS", date, "cam_params.xml"),
                            )

        if ugv_rgb:
            if check_dir(parent_dir, "images", "UGV-RGB", output_dir=output_dir):
                for date in os.listdir(os.path.join(parent_dir, "images", "UGV-RGB")):
                    os.makedirs(os.path.join(output_dir, "images", "UGV-RGB", date), exist_ok=True)
                    calib_fp = os.path.join(parent_dir, "images", "UGV-RGB", date, "cam_params.xml")
                    if os.path.isfile(calib_fp):
                        shutil.copy2(
                                calib_fp,
                                os.path.join(output_dir, "images", "UGV-RGB", date, "cam_params.xml")
                                )
                    if os.path.isdir(os.path.join(parent_dir, "images", "UGV-RGB", date, f"plot{plot_id}")):
                        shutil.copytree(
                            os.path.join(parent_dir, "images", "UGV-RGB", date, f"plot{plot_id}"),
                            os.path.join(output_dir, f"plot{plot_id}", date, "images", "UGV-RGB"),
                            copy_function=shutil.copy2,
                            dirs_exist_ok=True
                            )

    # raster 
    if check_dir(parent_dir, "raster_data"):
        plot_poly = get_plot_poly(shp_fp, plot_id)
        if uav1_rgb:
            if check_dir(parent_dir, "raster_data", "UAV1-RGB", output_dir=output_dir):
                for ortho_fn in os.listdir(os.path.join(parent_dir, "raster_data", "UAV1-RGB")):
                    date = ortho_fn[:-4]
                    os.makedirs(os.path.join(output_dir, f"plot{plot_id}", date, "raster_data", "UAV1-RGB"), exist_ok=True)
                    clip =  get_clip_plot_ms(
                            os.path.join(parent_dir, "raster_data", "UAV1-RGB", ortho_fn),
                            plot_poly,
                            os.path.join(output_dir, f"plot{plot_id}", date, "raster_data", "UAV1-RGB", ortho_fn)
                            )

        if uav2_rgb:
            if check_dir(parent_dir, "raster_data", "UAV2-RGB", output_dir=output_dir):
                for ortho_fn in os.listdir(os.path.join(parent_dir, "raster_data", "UAV2-RGB")):
                    date = ortho_fn[:-4]
                    os.makedirs(os.path.join(output_dir, f"plot{plot_id}", date, "raster_data", "UAV2-RGB"), exist_ok=True)
                    clip =  get_clip_plot_ms(
                            os.path.join(parent_dir, "raster_data", "UAV2-RGB", ortho_fn),
                            plot_poly,
                            os.path.join(output_dir, f"plot{plot_id}", date, "raster_data", "UAV2-RGB", ortho_fn)
                            )

        if uav3_rgb:
            if check_dir(parent_dir, "raster_data", "UAV3-RGB", output_dir=output_dir):
                for ortho_fn in os.listdir(os.path.join(parent_dir, "raster_data", "UAV3-RGB")):
                    date = ortho_fn.split("_")[0]
                    os.makedirs(os.path.join(output_dir, f"plot{plot_id}", date, "raster_data", "UAV3-RGB"), exist_ok=True)
                    clip =  get_clip_plot_ms(
                            os.path.join(parent_dir, "raster_data", "UAV3-RGB", ortho_fn),
                            plot_poly,
                            os.path.join(output_dir, f"plot{plot_id}", date, "raster_data", "UAV3-RGB", ortho_fn)
                            )

        if uav3_ms:
            if check_dir(parent_dir, "raster_data", "UAV3-MS", output_dir=output_dir):
                for ortho_fn in os.listdir(os.path.join(parent_dir, "raster_data", "UAV3-MS")):
                    if ortho_fn[-4:]==".tif":
                        date = ortho_fn[:-4]
                        os.makedirs(os.path.join(output_dir, f"plot{plot_id}", date, "raster_data", "UAV3-MS"), exist_ok=True)
                        clip =  get_clip_plot_ms(
                                os.path.join(parent_dir, "raster_data", "UAV3-MS", ortho_fn),
                                plot_poly,
                                os.path.join(output_dir, f"plot{plot_id}", date, "raster_data", "UAV3-MS", ortho_fn)
                            )

    # point clouds 
    if check_dir(parent_dir, "point_clouds"):
        # TODO change this to match the new data structure 
        d_buffer_r= 1.1
        plot_poly = get_plot_poly(shp_fp, plot_id)
        plot_poly = scale(plot_poly, xfact=d_buffer_r, yfact=d_buffer_r, origin='center')
        min_x, min_y, max_x, max_y = plot_poly.bounds
        min_z, max_z = (221, 226)  # clip points in z that are noise
        if uav1_rgb:
            if is_generate_uav1_rgb:
                if check_dir(parent_dir, "point_clouds", "UAV1-RGB", output_dir=output_dir):
                    for date in os.listdir(os.path.join(parent_dir, "point_clouds", "UAV1-RGB")):
                        pc_dir = os.path.join(parent_dir, "point_clouds", "UAV1-RGB", date)
                        os.makedirs(os.path.join(output_dir, f"plot{plot_id}"), exist_ok=True) # FIXME , date, "point_clouds", "UAV1-RGB"), exist_ok=True)
                        pc_list = get_pc_list(pc_dir, plot_poly)
                        new_x = np.array([])
                        new_y = np.array([])
                        new_z = np.array([])
                        new_r = np.array([])
                        new_g = np.array([])
                        new_b = np.array([])
                        for pc_fn in pc_list:
                            pc_fp = os.path.join(pc_dir, pc_fn)
                            pc_las = laspy.read(pc_fp)
                            new_x = np.concatenate([new_x, np.array(pc_las.x)])
                            new_y = np.concatenate([new_y, np.array(pc_las.y)])
                            new_z = np.concatenate([new_z, np.array(pc_las.z)])
                            new_r = np.concatenate([new_r, pc_las.red])
                            new_g = np.concatenate([new_g, pc_las.green])
                            new_b = np.concatenate([new_b, pc_las.blue])
    
                        # clip to polygon
                        # clip into rectangle in UTM
                        mask = new_x > min_x
                        mask = np.logical_and(mask, new_y > min_y)
                        mask = np.logical_and(mask, new_x < max_x)
                        mask = np.logical_and(mask, new_y < max_y)
                        mask = np.logical_and(mask, new_z < max_z)
                        mask = np.logical_and(mask, new_z > min_z)

                        new_x = new_x[mask]
                        new_y = new_y[mask]
                        new_z = new_z[mask]
                        new_r = new_r[mask]
                        new_g = new_g[mask]
                        new_b = new_b[mask]

                        # clip to exact polygon 
                        pts = points(new_x, new_y)
                        mask = contains(plot_poly, pts)
                        new_x = new_x[mask]
                        new_y = new_y[mask]
                        new_z = new_z[mask]
                        new_r = new_r[mask]
                        new_g = new_g[mask]
                        new_b = new_b[mask]

                        header = laspy.LasHeader(point_format=2, version="1.2")
                        if len(new_x) == 0:  # no points to write 
                            print(f"no points for UAV1-RGB on {date} of plot{plot_id}")
                            continue
                        header.offsets = np.array([new_x.min(), new_y.min(), new_z.min()])
                        header.scales = pc_las.header.scale
                        with laspy.open(os.path.join(output_dir, f"plot{plot_id}", f"{date}.las"), mode="w", header=header) as writer:
                            point_record = laspy.ScaleAwarePointRecord.zeros(len(new_x), header=header)
                            point_record.x = new_x
                            point_record.y = new_y
                            point_record.z = new_z
                            point_record.red = new_r
                            point_record.green = new_g
                            point_record.blue = new_b
                            
                            writer.write_points(point_record)
            else:  # just copy files over
                if check_dir(parent_dir, "point_clouds", "UAV1-RGB", output_dir=output_dir):
                    parent_dir_uav1_rgb = os.path.join(parent_dir, "point_clouds", "UAV1-RGB", f"plot{plot_id}")
                    if os.path.isdir(parent_dir_uav1_rgb):
                        for date_pc in os.listdir(parent_dir_uav1_rgb):
                            new_dir = os.path.join(output_dir, f"plot{plot_id}",  date_pc.split(".")[0], "point_clouds", "UAV1-RGB")
                            os.makedirs(new_dir, exist_ok=True)
                            shutil.copy2(
                                os.path.join(parent_dir_uav1_rgb, date_pc),
                                os.path.join(new_dir, date_pc),
                            )

        if uav2_lidar:
            if check_dir(parent_dir, "point_clouds", "UAV2-Lidar", output_dir=output_dir):
                for pc_fn in os.listdir(os.path.join(parent_dir, "point_clouds", "UAV2-Lidar")):
                    pc_fp = os.path.join(parent_dir, "point_clouds", "UAV2-Lidar", pc_fn)
                    date = pc_fn.split(".")[0]
                    os.makedirs(os.path.join(output_dir,  f"plot{plot_id}", date, "point_clouds", "UAV2-Lidar"), exist_ok=True)
                    las = laspy.read(pc_fp)
                    las = las[las.x > min_x]
                    las = las[las.y > min_y]
                    las = las[las.x < max_x]
                    las = las[las.y < max_y]

                    # crop tighty to polygon 
                    pts = points(las.x, las.y)
                    mask = contains(plot_poly, pts)
                    las = las[mask]
 
                    # laspy to o3d, and SOR
                    las_points = np.vstack((las.x, las.y, las.z)).transpose()
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(las_points)
                    if date[3] == "8":
                        std_ratio=1.0
                    else:
                        std_ratio=2.0
                    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=6,
                            std_ratio=std_ratio)
                    las = las[ind]
                    las.update_header() 
                    las.write(os.path.join(output_dir, f"plot{plot_id}", date, "point_clouds", "UAV2-Lidar", f"{date}.las"))

        if ugv_lmi:
            if check_dir(parent_dir, "point_clouds", "UGV-LMI", output_dir=output_dir):
                parent_dir_ugvlmi = os.path.join(parent_dir, "point_clouds", "UGV-LMI", f"plot{plot_id}")
                if os.path.isdir(parent_dir_ugvlmi):
                    for date_pc in os.listdir(parent_dir_ugvlmi):
                        new_dir = os.path.join(output_dir, f"plot{plot_id}",  date_pc.split(".")[0], "point_clouds", "UGV-LMI")
                        os.makedirs(new_dir, exist_ok=True)
                        shutil.copy2(
                            os.path.join(parent_dir_ugvlmi, date_pc),
                            os.path.join(new_dir, date_pc),
                        )

        if ugv_ouster:
            if check_dir(parent_dir, "point_clouds", "UGV-Ouster", output_dir=output_dir):
                parent_dir_ugvlmi = os.path.join(parent_dir, "point_clouds", "UGV-Ouster", f"plot{plot_id}")
                if os.path.isdir(parent_dir_ugvlmi):
                    # open ugv area shape 
                    shp = gpd.read_file("/home/linn/ipb/leaf_scanner/UGV_area/UGV_area2.shp")
                    polygon = shp.geometry.iloc[0]
                    for date_pc in os.listdir(parent_dir_ugvlmi):
                        new_dir = os.path.join(output_dir, f"plot{plot_id}",  date_pc.split(".")[0], "point_clouds", "UGV-Ouster")
                        os.makedirs(new_dir, exist_ok=True)

                        las = laspy.read(os.path.join(parent_dir_ugvlmi, date_pc))
                        pts = points(las.x, las.y)
                        mask = contains(polygon, pts)
                        las = las[mask]

                        # laspy to o3d, and SOR
                        las_points = np.vstack((las.x, las.y, las.z)).transpose()
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(las_points)
                        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=6,                                                                                    std_ratio=1.0)
                        las = las[ind]
                        las.write(os.path.join(new_dir, date_pc))

                       # shutil.copy2(
                       #     os.path.join(parent_dir_ugvlmi, date_pc),
                       #     os.path.join(new_dir, date_pc),
                       # )


    # metadata 
    if check_dir(parent_dir, "LAI_biomass_and_metadata"):
        # just copy data over as is
        shutil.copytree(
                os.path.join(parent_dir, "LAI_biomass_and_metadata"),
                os.path.join(output_dir, "LAI_biomass_and_metadata"),
                copy_function=shutil.copy2,
                dirs_exist_ok=True
                )


if __name__ == '__main__':
    main()

