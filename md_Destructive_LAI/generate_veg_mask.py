"""A script to generate veg masks for manual correction

"""

import os
import click
import pdb
import shutil
import concurrent.futures

import yaml
import numpy as np
import cv2
from tqdm import tqdm
from blend_modes import multiply

from process_images import generate_veg_mask


@click.command()
@click.option('--img_dir', required=True, help="path to dir of images")
@click.option('--out_dir', required=True, help="path to output dir")
@click.option("--calib_fp",type=str,help ="path to the calibration.yaml used to crop the image")
def main(img_dir, out_dir, calib_fp):
     process_one_dir(img_dir, out_dir, calib_fp)

def process_one_dir(dir_fp, out_dir, calib_fp):
    print(dir_fp)
    # read from yaml written by gui
    calib_file = open(calib_fp, 'r')  # TODO use with block to automatically close
    calib_yaml = yaml.unsafe_load(calib_file)
    roi_coords =  calib_yaml['crop_rect']
    crop_x, crop_y, crop_w, crop_h = roi_coords
    lower_veg_bound_hue  =  calib_yaml['lower_hue']
    lower_veg_bound_saturation  =  calib_yaml['lower_saturation']
    lower_veg_bound_value  =  calib_yaml['lower_value']
    lower_veg_bound_hsv = np.array([lower_veg_bound_hue, lower_veg_bound_saturation, lower_veg_bound_value])

    upper_veg_bound_hue  =  calib_yaml['upper_hue']
    upper_veg_bound_saturation  =  calib_yaml['upper_saturation']
    upper_veg_bound_value  =  calib_yaml['upper_value']
    upper_veg_bound_hsv = np.array([upper_veg_bound_hue, upper_veg_bound_saturation, upper_veg_bound_value])
    calib_file.close()

    output_dir = os.path.join(out_dir, "labeller", "semantics")
    os.makedirs(output_dir, exist_ok=True)
    out_vis_dir = os.path.join(out_dir, "vis_vm")
    os.makedirs(out_vis_dir, exist_ok=True)
    pics_dir = os.path.join(dir_fp)

    for pics_name in tqdm(os.listdir(pics_dir)):
        frame_ori = cv2.imread(os.path.join(pics_dir, pics_name))

        # crop image
        frame = frame_ori[crop_y: crop_y + crop_h, crop_x: crop_x + crop_w]

        # get veg mask 
        veg_mask = generate_veg_mask(frame, lower_veg_bound_hsv, upper_veg_bound_hsv)

        # save in image labeller format
        label_format = np.zeros((veg_mask.shape[0], veg_mask.shape[1], 3), dtype=np.uint16)
        label_format[:,:,0] = veg_mask.astype(np.uint16) / 255 
        label_format[veg_mask!=0,1] = 1
        cv2.imwrite(os.path.join(output_dir,pics_name), label_format)

        # write cropped image too
        cv2.imwrite(os.path.join(output_dir,"..",pics_name), frame)

        # write visualisation
        frame=cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2RGBA)
        overlaid = multiply(frame.astype(float), np.expand_dims(veg_mask, -1).astype(float)/255 * np.array([0,0,255,255]), 0.5)
        cv2.imwrite(os.path.join(out_vis_dir,pics_name), overlaid)


if __name__ == "__main__":
    main()
