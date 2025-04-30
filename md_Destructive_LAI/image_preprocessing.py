import click
import os
import cv2
import numpy as np
import shutil
import yaml
import math
import pdb
import copy

from pyzbar.pyzbar import decode
from tqdm import tqdm

from utils import get_exR, get_pinks
from tracker import Tracker, ROI

def parse_qrcode_list(checkerboard_list, checkerboard_fp_list):
    qr_tag = None
    for checkerboard_frame in checkerboard_list:
        qr_tag = parse_qrcode(checkerboard_frame)
        if qr_tag != "no_name":
            break
    if qr_tag == "no_name" or qr_tag is None:
        print(f"QR tag cannot be read from {checkerboard_fp_list[0]} to {checkerboard_fp_list[-1]}.")
    return qr_tag


def parse_qrcode(image):
    # Decode the QR code(s) in the image
    decoded_objects = decode(image)

    if len(decoded_objects) == 1:
        obj = decoded_objects[0]
        qr_data = obj.data.decode('utf-8')
        qr_type = obj.type

        print(f"QR Code Type: {qr_type}")
        print(f"QR Code Data: {qr_data}")
        return qr_data
    elif len(decoded_objects) > 1:
        raise ValueError("More than one QR code detected in the image.")
    else:
        print("No QR codes found in the image. Returning default")
        return "no_name"
 
def get_pinktag_thresh(image_clean, ExRed_thresh=200, is_debug=False):
    """
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    frame_threshold = cv2.inRange(img_hsv, (137, 0, 0), (180, 255, 254))  # FIXME hardcoded values here
    is_pink = (frame_threshold == 255).sum() > num_px_thresh
    """

    """
    exR = get_exR(image_clean)
    exR_bool = exR > ExRed_thresh
    """

    exR_bool = get_pinks(image_clean)
    return exR_bool.astype(np.uint8)* 255

def check_is_qr(image, is_debug=False):
    return len(decode(image)) > 0

def is_close_tol(np_array, value, tolerance):
    np_value = np.ones(np_array.shape) * value
    diff = np_array - np_value 
    abs_diff = np.absolute(diff)
    is_close = abs_diff < tolerance

    return is_close

def majority_vote(np_bool_arr):
    vote_true = np_bool_arr.sum()
    return vote_true > len(np_bool_arr)/2


def check_is_checkerb(image, hough_threshold=100, angle_tolerance_rad=0.1, min_lines = 7,is_debug=False):
    """also detects partial checkerboards
    """
    image_dirty=copy.deepcopy(image)
    is_gridlike = False
    is_enoughlines = False

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise (adjust the kernel size as needed)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection to find edges in the image
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # Use Hough Line Transform to find lines in the image
    lines = cv2.HoughLines(edges, 1, np.pi / 180, hough_threshold)

    if lines is not None:
        # look for perpendicular and parallel hough lines
        thetas = lines[:,0,1]
        is_hori_0 = is_close_tol(thetas, 0, angle_tolerance_rad)
        is_hori_pi = is_close_tol(thetas, np.pi, angle_tolerance_rad)
        is_hori = np.logical_or(is_hori_0, is_hori_pi)
        is_vert = is_close_tol(thetas, np.pi/2, angle_tolerance_rad)
        is_gridlike_np = np.logical_or(is_hori, is_vert)
        is_gridlike = majority_vote(is_gridlike_np) and is_hori.any() and is_vert.any()

        if is_debug:
            # draw found lines
            for line in lines:
                rho = line[0][0]
                theta = line[0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                cv2.line(image_dirty, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
            cv2.imwrite("find_calib.png" , image_dirty)
        
        is_enoughlines = len(lines) >= min_lines

    return is_gridlike and is_enoughlines


def detect_checkerboard(image, pattern_size=(7, 7), threshold=100, is_debug=False):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise (adjust the kernel size as needed)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection to find edges in the image
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # Use Hough Line Transform to find lines in the image
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)

    if lines is not None:
        # Check if there are enough lines (approximately horizontal and vertical)
        if len(lines) >= pattern_size[0] + pattern_size[1]:
            print("Found {} lines".format(len(lines)))
            if is_debug:
                # draw found lines
                for line in lines:
                    rho = line[0][0]
                    theta = line[0][1]
                    a = math.cos(theta)
                    b = math.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                    pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                    cv2.line(image, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
                cv2.imwrite("find_calib.png" , image)
            return True
    return False

def calibrate(frame, width_corners=8, height_corners=7, gray_thresh=500000, is_brute_trial=False, draw_fp=None):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scale = -1
    if (gray < 30).sum() > gray_thresh or is_brute_trial:
        ret, corners = cv2.findChessboardCorners(
            gray, (width_corners, height_corners), None,
        )
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            # cv2.drawChessboardCorners(frame, (width_corners,height_corners), corners2, ret)
            # cv2.imshow('frame', frame)
            # cv2.waitKey(500)

            # # calibrate camera
            # objpoints = [] # 3d point in real world space
            # imgpoints = [] # 2d points in image plane.
            # objp = np.zeros((height_corners*width_corners,3), np.float32)
            # objp[:,:2] = np.mgrid[0:width_corners,0:height_corners].T.reshape(-1,2)
            # objpoints.append(objp)
            # imgpoints.append(corners2)
            # import ipdb;ipdb.set_trace()  # fmt: skip
            # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            tl = corners2[0]
            tr = corners2[width_corners - 1]
            bl = corners2[-width_corners]
            br = corners2[-1]
            width = np.linalg.norm(tl - tr)
            width2 = np.linalg.norm(bl - br)
            width_scale = 0.025 / (((width + width2) / 2) / (width_corners - 1))
            heigth = np.linalg.norm(tl - bl)
            heigth2 = np.linalg.norm(tr - br)
            height_scale = 0.025 / (((heigth + heigth2) / 2) / (height_corners - 1))
            scale = (width_scale + height_scale) / 2
            if not(draw_fp is None):
                cv2.drawChessboardCorners(frame, (width_corners,height_corners), corners2, ret)
                cv2.imwrite(draw_fp, frame)
    # pdb.set_trace()
    return scale
    
@click.command()
@click.option(
    "--data_folder",
    "-d",
    type=str,
    help="path to the data folder",
)
@click.option(
    "--output_folder",
    type=str,
    help="path to the output folder",
)
@click.option('--debug', is_flag=True)
@click.option('--skip_sanity', is_flag=True)
def main(data_folder, output_folder, debug, skip_sanity):
    ispdb = False
    assert data_folder != output_folder, "data_folder and output_folder must be different"
    
    # iterate over dates
    for date in os.listdir(data_folder):
        
        # if date != "20230706-08":  # DBG
        #     continue  # DBG

        print("Processing date: {}".format(date))
        date_folder = os.path.join(data_folder, date)
        calib_fp = os.path.join(output_folder, date, "calibration.yaml")

        calib_file = open(calib_fp, 'r')  # TODO use with block to automatically close
        calib_yaml = yaml.unsafe_load(calib_file)

        roi_coords =  calib_yaml['crop_rect']


        for session in os.listdir(date_folder):
            session_folder = os.path.join(date_folder, session)
            if not os.path.isdir(session_folder):
                print(f"Skipping {session_folder} because its not a dir.")
                continue
            print(f"Working on dir {session_folder}")
            
            # img_folder = os.path.join(data_folder, "pics")
            frame_list = os.listdir(session_folder)
            frame_ids = [int(x.split("_")[2]) for x in frame_list]
            frame_list = [x for _, x in sorted(zip(frame_ids, frame_list))]
            
            os.makedirs(output_folder, exist_ok=True)
            
            plot_name = None
            scale = -1
            
            
            checkerboard_frames = []
            checkerboard_fp_list = []

            vx=250
            birth_roi = ROI(0,0,vx,-1)
            pink_tag_tracker = Tracker(vx = vx, vy = 5, min_num_px_overlap = 100, max_lifetime=10, birthing_ROI=birth_roi)
            is_pink_tag = False
            tracker_min_age = 2
            pink_tag_count = 0

            # is_stop = True 
            is_stop = False 
            pics_folder = os.path.join( output_folder, date, "unsorted")
            os.makedirs(pics_folder, exist_ok=True)
            for frame_path in tqdm(frame_list):
                """ DBG
                if "950" in frame_path:
                    is_stop = False
                if is_stop:
                    continue
                """
                print("Processing frame: {}".format(frame_path))
                # load frame
                frame = cv2.imread(os.path.join(session_folder, frame_path))
                if frame is None:
                    print(f"Error processing frame {frame_path}, skipping to next frame.")
                    corrupt_pics_folder=os.path.join(output_folder, date, "corrupt")
                    os.makedirs(corrupt_pics_folder, exist_ok=True)
                    shutil.copy(os.path.join(session_folder, frame_path), corrupt_pics_folder)
                    continue

                x, y, w, h = roi_coords
                frame = frame[y: y + h, x: x + w]
                
                # check if checkerboard is visible
                # contains_checkb = detect_checkerboard(frame, pattern_size=(8,9), is_debug=debug)

                contains_checkb = check_is_qr(frame,is_debug=debug) or check_is_checkerb(frame, is_debug=debug)
                pinktag_thresh_frame = get_pinktag_thresh(frame, is_debug=debug)

                time_now = int(frame_path.split("_")[2])
                birthed_list, killed_list = pink_tag_tracker.update(pinktag_thresh_frame, time_now, frame_path)

                if len(pink_tag_tracker.tracks_dict) == 1:
                    is_potential_pinktag=True
                elif len(pink_tag_tracker.tracks_dict) > 1:
                    print("Two or more pink tags in tracking...")
                    is_potential_pinktag=True
                    """
                    is_potential_pinktag = False
                    pink_tracker_keylist = list(pink_tag_tracker.tracks_dict)
                    for obj_key in pink_tracker_keylist:
                        pink_tag_tracker.kill_object(obj_key)
                    """
                else:
                    is_potential_pinktag = False

                # finished reading sequence of pink tags
                if len(killed_list) > 0 :
                    if killed_list[0].age >= tracker_min_age:  # make sure that its actually a pink tag
                        pink_tag_count += 1
                        plot_folder = os.path.join(output_folder, date, f"{session}_{plot_name}_pink_tag_{pink_tag_count}")
                        pink_tag_dir = os.path.join(plot_folder, "pink_tags")  # create pinktag dir 
                        os.makedirs(pink_tag_dir, exist_ok=True)
                        np.savetxt(os.path.join(plot_folder, "calib.txt"), np.array([scale]))

                        for frame_fp in killed_list[0].frames_fp_list:
                            if frame_fp is not None:
                                shutil.copy(os.path.join(session_folder, frame_fp), pink_tag_dir)
                        if debug:
                            for obj_i, killed_obj in enumerate(killed_list):
                                killed_obj.vis_pred_curr(os.path.join(pink_tag_dir, f"tracker_{obj_i}_last_frame_{frame_fp}"))
                        
                        # copy pink tags to pinktag dir
                        pics_folder = os.path.join(plot_folder, "pics")
                        # create a new folder for the plot
                        os.makedirs(pics_folder, exist_ok=True)
                    else:  # not actually pink tags, i.e., false positives
                        for frame_fp in killed_list[0].frames_fp_list:
                            shutil.copy(os.path.join(session_folder, frame_fp), pics_folder)

                if len(killed_list) > 1:
                    print("killed more than 1 object... this is weird because there should only be one pink tag")

                if contains_checkb:
                    # accumulate checkerboard frames until a frame without checkerboard is found
                    checkerboard_frames.append(frame)
                    checkerboard_fp_list.append(frame_path)
                    print("Accumulating checkerboard frames: {}".format(len(checkerboard_frames)))
                    continue
                else:
                    if len(checkerboard_frames) > 0:

                        for calib_frame in checkerboard_frames:
                            scale = calibrate(calib_frame) # scale in px/mm
                            if scale > 0:
                                break
                        # read plot name from qr code
                        old_plot_name = plot_name
                        qr_tag = parse_qrcode_list(checkerboard_frames, checkerboard_fp_list)
                        if qr_tag == "no_name" and len(checkerboard_fp_list) < 2:  # the qr is not found so then, its actually not a checkerboard
                            plot_name = old_plot_name
                            for frame_i in checkerboard_fp_list:
                                shutil.copy(os.path.join(session_folder, frame_i), pics_folder)
                            checkerboard_frames = []
                            checkerboard_fp_list = []
                        elif qr_tag == "error":
                            plot_name = old_plot_name
                        else:
                            plot_name = f"{qr_tag}_{checkerboard_fp_list[-1]}"

                        plot_folder = os.path.join(output_folder, date, f"{session}_{plot_name}")
                        pics_folder = os.path.join(plot_folder, "pics")
                        # create a new folder for the plot
                        os.makedirs(pics_folder, exist_ok=True)
                        
                        # save calib
                        np.savetxt(os.path.join(plot_folder, "calib.txt"), np.array([scale]))
                        
                        # save calibration frames to somewhere 
                        # TODO: save when reading instead of looping over list
                        if debug:
                            calib_pics_folder = os.path.join(plot_folder, "calib_pics")
                            os.makedirs(calib_pics_folder, exist_ok=True) 
                            for frame_i in checkerboard_fp_list:
                                shutil.copy(os.path.join(session_folder, frame_i), calib_pics_folder)

                        checkerboard_frames = []
                        checkerboard_fp_list = []
                    
                    if not is_potential_pinktag:
                        # a frame without checkerboard 
                        shutil.copy(os.path.join(session_folder, frame_path), pics_folder)

        calib_file.close()
                    
if __name__ == "__main__":
    main()
