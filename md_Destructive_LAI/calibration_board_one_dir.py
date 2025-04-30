#!/usr/bin/env python3
"""Get calib.txt from a single dir containing only images of the calibration board
"""
import os
import statistics

import numpy as np
import yaml
import click
import cv2
from tqdm import tqdm

from image_preprocessing import  calibrate


@click.command()
@click.option(
        "--checkerboard_dir",
        "-c",
        type=str,
        help ="path to the dir containing the checkerboard images"
        )
@click.option(
        "--output_dir",
        "-o",
        type=str,
        help ="path to the dir to contain the calib.txt"
        )
@click.option(
        "--calib_fp",
        type=str,
        help ="path to the calibration.yaml used to crop the image"
        )
def main(checkerboard_dir, output_dir,calib_fp):
    calib_file = open(calib_fp, 'r')  # TODO use with block to automatically close
    calib_yaml = yaml.unsafe_load(calib_file)
    roi_coords =  calib_yaml['crop_rect']
    x, y, w, h = roi_coords

    scales_list = []
    for img_name in tqdm(os.listdir(checkerboard_dir)):
        calib_img = cv2.imread(os.path.join(checkerboard_dir, img_name))
        calib_frame = calib_img[y: y + h, x: x + w]
        is_found=False

        for i in range(8, 3, -1):
            if is_found:
                is_found=False
                break

            for j in range(7, 3, -1):
                scale = calibrate(calib_frame, gray_thresh=1000, is_brute_trial=True,
                draw_fp=os.path.join(output_dir, img_name),
                width_corners=i,
                height_corners=j,
                )

                if scale > 0:
                    scales_list.append(scale)
                    print(i,j)
                    is_found=True
                    break

    if len(scales_list) > 0:
        ave_scale = statistics.mean(scales_list)
        print(f"saving to {output_dir}")
        np.savetxt(os.path.join(output_dir, "calib.txt"), np.array([ave_scale]))
    else:
        print("Error: no calibration board found!")


if __name__ == '__main__':
    main()
