"""A script to calculate LAI

leaves tracked using simple motion model of CV (estimated conveyor belt speed)
and associated using largest pixel overlap 
in case the camera dropped a frame unknowingly, the CV estimates the speed 
as multiples of the user-defined speed.

for short leaves, calculate LAI of all images where the leaves are fully visible, 
get the mean LAI and std and report these

for long leaves (when the leaf is longer than the FOV of the camera),
stich images where the leaf covers at least half the FOV in length,
find transformation between frames using SIFT feature matching with outlier removal based on motion model
only the latest frame is used for the lai calculation to avoid overlapping of incorrect correspondance.

"""

import random
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import torch
from dataclasses import dataclass
from tqdm import tqdm
import time
import copy
from pyzbar.pyzbar import decode
import csv
import concurrent.futures
import configparser

import click
import yaml
from blend_modes import lighten_only, addition, grain_merge, multiply, normal, darken_only

from tracker import Tracker, ROI, estimate_vel, blend_images


@dataclass
class LeafFrame:
    rgb: np.array
    instance_mask: np.array  # instance veg mask
    contour: np.array
    frame_partiality: bool
    transform_from_prev: np.array
    tracker_id: int

def is_touching_top(cat_contours, frame, thres):
    top = cat_contours[:, 1].min() < thres
    return top

def is_touching_bottom(cat_contours, frame, thres):
    height = frame.shape[0]
    bottom = height - cat_contours[:, 1].max() < thres 
    return bottom

def is_vertically_incomplete(cat_contours, frame, thres):
    top = is_touching_top(cat_contours, frame, thres)
    bottom = is_touching_bottom(cat_contours, frame, thres)
    return top or bottom


def is_touching_left(cat_contours, frame, thres):
    left = cat_contours[:, 0].min() < thres
    return left


def is_touching_right(cat_contours, frame, thres):
    width = frame.shape[1]
    right = width - cat_contours[:, 0].max() < thres
    return right

def check_partial(cat_contours, frame, thres=3):
    """Check if contour touches left or right image boarders.

    Args:
        cat_contours (_type_): instance mask
        frame (_type_): _description_
        thres (int, optional): _description_. Defaults to 3.

    Returns:
        is_partial: is partial
    """
    left = is_touching_left(cat_contours, frame, thres)
    right = is_touching_right(cat_contours, frame, thres)
    return left or right

class LeafTracker:
    def __init__(
        self,
        conveyor_velocity,
        scale,
        birth_roi_buffer = 600,
        min_age = 3,
        debug=False,
        step_delta_sy = 5.0,
        delta_sx = 50.0,
        sift_nfts=500
    ) -> None:
        # green has hue value of 60

        self.vel = conveyor_velocity
        self.scale = scale
        self.min_age = min_age  # age by count of tracker to consider a real obj

        self.debug = debug

        self.birthing_roi = ROI(0,0,birth_roi_buffer,-1)
        self.sift_nfts=sift_nfts

        self.delta_sx = delta_sx
        self.step_delta_sy = step_delta_sy
        self.tracker = Tracker(
                vx = self.vel[0],
                vy = self.vel[1],
                min_num_px_overlap = 100,
                max_lifetime = 30,
                timeout = 16,  # max number of frames that was dropped 
                birthing_ROI = self.birthing_roi,
                is_debug = debug
                )


    def add_images(self, frame, final_transform):
        # shift image
        translation_matrix = np.float32(
            [[1, 0, final_transform[0]], [0, 1, final_transform[1]]]
        )
        num_rows, num_cols = frame.shape[:2]
        transl_frame = cv2.warpAffine(
            self.prev_frame, translation_matrix, (num_cols, num_rows)
        )
        overlayed = cv2.addWeighted(transl_frame, 0.5, frame, 0.5, 0.0)
        return overlayed
    
    
    def filter_contours(self, contours, cont_hierarchy):
        parents = cont_hierarchy[0,:,-1] == -1
        valid_contours = [] 
        num_nns = []
        for i in range(len(contours)):
            if parents[i]:
                curr_contour = np.squeeze(contours[i], axis=1)
                valid_contours.append(curr_contour)
                dists = torch.cdist(
                    torch.tensor(self.prev_contour).float(), torch.tensor(curr_contour).float()
                )
                min_dists = dists.min(dim=0)[0]
                valid_associations = min_dists < self.nn_thres
                num_nns.append(valid_associations.sum())
        closest_id = np.argmax(num_nns)
        return valid_contours[closest_id].astype(float)

    def update(self, frame, prev_veg_mask, veg_mask, time_now, frame_path):
        # sometimes a frame was skipped due to firmware issues, this adds a multiple of the distance actually travelled.
        dt = time_now - self.tracker.time_now
        if not prev_veg_mask is None:
            vx, vy = estimate_vel(prev_veg_mask, veg_mask, self.tracker.one_step_vx, self.tracker.one_step_vy, dt)
            print(time_now, vx, vy, dt)
        else:
            vx = self.tracker.one_step_vx
            vy = self.tracker.one_step_vy


        birthed_list, killed_list = self.tracker.update(veg_mask, time_now, frame_path, frame, vx, vy) 

        # bridge tracker_obj to leaf_frame
        leaf_frames_list=[]
        for killed_tracks in killed_list:

            if killed_tracks.age < self.min_age:
                continue
            leaf_frames = []
            prev_time = None

            bgr_i = 0
            kp_old = None
            for killed_obj_timed_state in killed_tracks.timestamped_state_list:
                is_orb = True

                killed_obj_state = killed_obj_timed_state.state

                if killed_obj_state.is_pred:
                    continue

                # TODO this should only be for long leaves
                # remove frames that are at the end or at the beginning of the frame
                # since it is not necessary to track them and they have less features 
                ft_indicies = np.where(killed_obj_state.features)
                min_x = ft_indicies[1].min()
                max_x = ft_indicies[1].max()
                wx = max_x - min_x 
                if wx < killed_obj_state.features.shape[0]/2:
                    is_orb = False


                # get the delta time between frames 
                if prev_time is None:
                    prev_time = killed_obj_timed_state.timestamp

                dt = killed_obj_timed_state.timestamp - prev_time
                transf = (killed_obj_state.vx , killed_obj_state.vy)
                
                bgr_frame = killed_tracks.bgr_list[bgr_i]
                bgr_i +=1  # TODO 
                
                # create contours
                leaf_contour, hierarchy = cv2.findContours(
                    killed_obj_state.features.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
                )

                leaf_contour_cat = np.concatenate(leaf_contour).squeeze(-2).astype(float)

                # check partiality
                is_partial = check_partial(leaf_contour_cat, bgr_frame)


                # use edges instead of just contour
                # mask out all other instances
                bgr_instance_masked  = copy.deepcopy(bgr_frame)
                bgr_instance_masked[~killed_obj_state.features]=0
                bnw_instance_masked = cv2.cvtColor(bgr_instance_masked.astype(np.uint8), cv2.COLOR_BGR2GRAY)
               
                orb = cv2.SIFT_create(
                        nfeatures = self.sift_nfts, # 500,
                        nOctaveLayers=1,
                        contrastThreshold = 0.01,
                        edgeThreshold= 10,)
                kp = orb.detect(bnw_instance_masked,None)
                kp, des = orb.compute(bnw_instance_masked, kp)

                img2 = cv2.drawKeypoints(bnw_instance_masked, kp, None, color=(0,255,0), flags=0)

                kp_np = np.expand_dims(cv2.KeyPoint_convert(kp), 1)

                if not kp_old is None and len(kp_old) > 0:
                    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True) 

                    # remove keypoints which are already out of bounds
                    kp_old_culled = []
                    des_old_culled = []
                    kp_img_h = bgr_frame.shape[0]
                    kp_img_w = bgr_frame.shape[1]
                    for kp_i, des_i in zip(kp_old, des_old):
                        kp_x = kp_i.pt[0]
                        kp_y = kp_i.pt[1]

                        if kp_x < kp_img_w - transf[0]:  # conservatively only remove one step for now
                            if kp_y < kp_img_h - transf[1]:  # conservatively only remove one step for now
                                kp_old_culled.append(kp_i)
                                des_old_culled.append(des_i)

                    if len(des_old_culled) == 0 :
                        print("no keypoints from prev timestep!!")
                    kp_old=kp_old_culled
                    des_old = np.array(des_old_culled)
                    # print("des", des, "des_old", des_old)
                    if len(des_old) > 0 and len(des) > 0:
                        matches_bf = bf.match(des, des_old)
                    else:
                        matches_bf = np.array([])

                    # handle case where there are no matches
                    if len(matches_bf) > 0:
                        matches_bf = sorted(matches_bf, key=lambda x: x.distance)
                        num_max_matches = 50  # TODO make args
                        image_matches_bf = cv2.drawMatches(img2, kp, img_old, kp_old, matches_bf[:num_max_matches], None)

                        # clustering
                        s_list = []
                        for match in matches_bf[:]:
                            p1 = np.array(kp[match.queryIdx].pt)
                            p2 = np.array(kp_old[match.trainIdx].pt)
                            s = np.expand_dims(p2 - p1, -1)
                            s_list.append(s) 

                        # cv2.imwrite(f"orb_{time_now}.png", image_matches_bf)

                        s_np = np.array(s_list)[:,:,0]
                        
                        delta_sx = self.delta_sx  
                        delta_sy = self.step_delta_sy + (dt-1)
                        # use motion model to help with removing outliers
                        min_q = -transf[0] - delta_sx
                        max_q = -transf[0] + delta_sx
                        s_np = s_np[np.logical_and(s_np[:,0] > min_q, s_np[:,0] < max_q)]

                        min_q = -transf[1] - delta_sy
                        max_q = -transf[1] + delta_sy
                        s_np = s_np[np.logical_and(s_np[:,1] > min_q, s_np[:,1] < max_q)]


                        if s_np.shape[0] > 3:
                            upper_quartile = np.percentile(s_np[:,0], 90)
                            lower_quartile = np.percentile(s_np[:,0], 10)
                            min_q = min(upper_quartile, lower_quartile)
                            max_q = max(upper_quartile, lower_quartile)
                            s_np = s_np[np.logical_and(s_np[:,0] > min_q, s_np[:,0] < max_q)]

                        if s_np.shape[0] > 3:
                            upper_quartile = np.percentile(s_np[:,1], 90)
                            lower_quartile = np.percentile(s_np[:,1], 10)
                            min_q = min(upper_quartile, lower_quartile)
                            max_q = max(upper_quartile, lower_quartile)
                            s_np = s_np[np.logical_and(s_np[:,1] > min_q, s_np[:,1] < max_q)]


                        if s_np.shape[0] > 0:
                            s_ave = np.average(s_np, axis=0) 
                        else:
                            # no points survived the filter
                            s_ave = -np.array(transf) 

                        # print(f"{killed_obj_timed_state.timestamp} s_ave {s_ave}")
                    else:
                        s_ave = np.array([-killed_obj_state.vx , -killed_obj_state.vy]) 
                else:
                    s_ave = np.array([0,0]) 

                # print(s_ave)

                des_old = des
                kp_old = kp
                img_old = bnw_instance_masked

                leaf_frame = LeafFrame( 
                                        rgb = bgr_frame,  
                                        instance_mask = killed_obj_state.features,
                                        contour=leaf_contour,
                                        frame_partiality = is_partial,
                                        transform_from_prev =  (-s_ave[0], -s_ave[1]),
                                        tracker_id = killed_tracks.obj_id
                                        )
                leaf_frames.append(leaf_frame)
                prev_time = killed_obj_timed_state.timestamp
            leaf_frames_list.append(leaf_frames)
        return killed_list, leaf_frames_list
        
    def get_tracks_dict(self):
        return self.tracker.tracks_dict

    def update_old(self, frame, contours, cont_hierarchy, stupid_contours, veg_mask):
        cat_contours = np.concatenate(contours).squeeze(-2).astype(float)
        is_partial = check_partial(cat_contours, frame, thres=5)
        if self.debug:
            print("Partial: ", is_partial)
        if len(self.prev_contour) == 0:
            self.prev_contour = cat_contours
            self.prev_stupid_contours = stupid_contours
            self.prev_frame = frame
        else:
            final_transform = copy.deepcopy(self.vel)
            self.prev_contour += self.vel
            end_reached = False
            for step in range(self.icp_steps):
                transl, error, goodness = icp(
                    self.prev_contour, self.prev_stupid_contours, cat_contours, stupid_contours, thres=self.nn_thres
                )

                if goodness < 0.01:
                    self.process_leaf()
                    end_reached = True
                    break
                self.prev_contour += transl
                final_transform += transl
            if not end_reached:
                overlayed = self.add_images(frame, final_transform)
                
                self.prev_contour = self.filter_contours(contours, cont_hierarchy)
                self.prev_frame = overlayed
                leaf_frame = LeafFrame(
                    rgb=frame,
                    instance_mask=veg_mask,
                    contour=cat_contours,
                    viz_contour=contours,
                    frame_partiality=is_partial,
                    transform_from_prev=final_transform,
                )
                self.frame_db.append(leaf_frame)

    def reset(self):
        self.prev_contour = []
        self.prev_frame = []
        self.frame_db = []
        self.is_leaf = True
        
class FrameProcessor:
    def __init__(self, scale, debug=False, pics_dir=None, skip_first_frame_longleaf=True, is_vis=False, skip_last_n_frames_longleaf=0) -> None:
        self.scale = scale
        self.debug = debug
        self.is_vis = is_vis
        self.lais_db = []
        self.obj_id_db = []
        self.lais_std_db = []
        self.pics_dir_fp=pics_dir
        self.is_skip_first_frame_of_longleaf=skip_first_frame_longleaf
        self.skip_last_n_frames_longleaf=skip_last_n_frames_longleaf
        
        # ICP params
        self.icp_steps = 50
        self.nn_thres = 20
        self.border_cut_thres = 3
        self.border_ignore_thres = 20

    def process_leaves(self, leaf_list):
        for leaf_track in leaf_list:
            if len(leaf_track) == 0:
                continue

            # check if leaf was not scanned properly and is out of frame 
            for leaf_track_i in leaf_track:
                cat_con = np.concatenate(leaf_track_i.contour).squeeze(-2).astype(float)  # TODO make this a function
                if is_vertically_incomplete(cat_con, leaf_track_i.instance_mask, 3):
                    # the leaf area is not reliable since the leaf is not fully seen
                    self.lais_db.append(-2)
                    self.obj_id_db.append(leaf_track[0].tracker_id) 
                    self.lais_std_db.append(-2)
                    return

            is_longleaf = ~np.any(~np.array([x.frame_partiality for x in leaf_track]))

            if is_longleaf:
                # check if leaf exists 
                is_enters_left = False
                is_exits_right = False
                for leaf_track_i in leaf_track:  # TODO I am sure you can make this more pythonic 
                    cat_con = np.concatenate(leaf_track_i.contour).squeeze(-2).astype(float)
                    if is_touching_left(
                                        cat_con,
                                        leaf_track_i.instance_mask,  
                                        thres=3):
                        is_enters_left =True
                        break
                for leaf_track_i in leaf_track:
                    cat_con = np.concatenate(leaf_track_i.contour).squeeze(-2).astype(float)
                    if is_touching_right(
                                         cat_con,
                                         leaf_track_i.instance_mask,  
                                         thres=3):
                        is_exits_right =True
                        break

                if not (is_enters_left and is_exits_right):
                    print(f"Leaf {leaf_track[0].tracker_id} might be long leaf but it does not touch both the left and right borders... skipping leaf")

                    return

                is_enters_left = ~np.any(~np.array([x.frame_partiality for x in leaf_track]))

                if self.debug:
                    print(f"Long leaf! obj id: {leaf_track[0].tracker_id}")
                if len(leaf_track) < 3:  
                    print("ERROR: minimum 3 tracked frames needed in long leaf mode! Skipping this instance.")
                    continue
                stiched_frame = self.stich_instance_masks(leaf_track, obj_id = leaf_track[0].tracker_id)
                lai = self.calculate_lai(stiched_frame, scale=self.scale)
                self.lais_db.append(lai)
                self.obj_id_db.append(leaf_track[0].tracker_id) 
                self.lais_std_db.append(-1)   # long leaf only has one lai so no std
            else:
                if self.debug:
                    print("Short leaf!")
                lais = []
                for i, frame in enumerate(leaf_track):
                    if not frame.frame_partiality:
                        lais.append(self.calculate_lai(frame, scale=self.scale))
                        if self.is_vis:
                            frame_rgb = blend_images(frame.rgb, np.expand_dims(frame.instance_mask, -1)*(0,0,255))  # instance mask is used to calc lai now
                            cv2.imwrite(f"{self.pics_dir_fp}/../test_dbg/test_{leaf_track[0].tracker_id}_lai_{i}.png", frame_rgb)
                if self.debug:
                    print("LAIs:", lais)
                lais = np.array(lais)
                std = lais.std()
                if std > 0.01 and self.debug:
                    print("WARNING: std of lai estimation is ", std)
                mean_lai = np.array(lais).mean()
                self.lais_db.append(mean_lai)
                self.obj_id_db.append(leaf_track[0].tracker_id)
                self.lais_std_db.append(std)
                
            # visualize processed thing
            if self.is_vis:
                # select middle frame
                mid_frame = leaf_track[int(len(leaf_track)/2)]
                frame_rgb = mid_frame.rgb
                frame_rgb = blend_images(frame_rgb, np.expand_dims(mid_frame.instance_mask, -1)*(0,0,255))  # instance mask is used to calc lai now

                # cv2.imwrite(f"{self.pics_dir_fp}/../test_dbg/test_{leaf_track[0].tracker_id}_lai.png", frame_rgb)

            if self.debug:
                print("Leaf processed!")

    def contour2pointcloud(self, contour, img_size, z = 0, delete_x_buffer=0):
        leaf_contour_cat = np.concatenate(contour).squeeze(-2).astype(float)

        # remove borders
        border_mask = leaf_contour_cat > delete_x_buffer  # drop points we know was not seen in last frame but only during assoc of curr frame
        border_mask_i = np.logical_and(border_mask[:,0], border_mask[:,1])
        leaf_contour_cat = leaf_contour_cat[border_mask_i]


        # also remove extreme borders 
        border_mask_0 = leaf_contour_cat[:,0] < img_size[0]
        border_mask_1 = leaf_contour_cat[:,1] < img_size[1]
        border_mask_i = np.logical_and(border_mask_0, border_mask_1)
        leaf_contour_cat = leaf_contour_cat[border_mask_i]

        z_np = np.ones((leaf_contour_cat.shape[0],1)) * z 
        leaf_contour_cat = np.concatenate((leaf_contour_cat, z_np), axis=1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(leaf_contour_cat)

        return pcd

    def pc2px_np(self, pc):
        pc_np = np.asarray(pc.points)
        # remove z 
        pc_np = pc_np[:,0:-1].astype(np.int_)
        return pc_np

    def stich_instance_masks(self, frame_db, icp_threshold = 20, obj_id = -1):  # TODO expose icp threshold args
        transforms_db = []

        is_first_frame = True
        for frame in frame_db:
            # get contours of instance
            if is_first_frame:
                prev_pcd = self.contour2pointcloud(frame.contour, frame.rgb.shape)
                is_first_frame = False
                cum_pts_np = np.asarray(prev_pcd.points)
                glob_tf = np.eye(4)
                continue
            new_contours = frame.contour
            new_pcd = self.contour2pointcloud(new_contours, frame.rgb.shape, delete_x_buffer=200)

            # estimated transform from motion model
            est_transf = frame.transform_from_prev
            transf_init = np.eye(4)
            transf_init[0,3] = -est_transf[0]
            transf_init[1,3] = -est_transf[1]

            # TODO remove this code since its not in use anymore
            source_temp = copy.deepcopy(new_pcd)
            target_temp = copy.deepcopy(prev_pcd)
            target_temp.estimate_normals()
            loss = o3d.pipelines.registration.TukeyLoss(k=0.1)
            reg_p2p = o3d.pipelines.registration.registration_icp(
                            source_temp, target_temp, icp_threshold, transf_init,
                            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                            o3d.pipelines.registration.ICPConvergenceCriteria(
                                max_iteration=0,
                                relative_fitness = 1e-20,
                                relative_rmse = 1e-20
                                ),
                            )
            transforms_db.append(reg_p2p.transformation)
            glob_tf = np.matmul(reg_p2p.transformation, glob_tf)
            curr_global = np.asarray(source_temp.transform(glob_tf).points)
            cum_pts_np = np.vstack((cum_pts_np, curr_global))

            prev_pcd = self.contour2pointcloud(new_contours, frame.rgb.shape) 

        cum_pc = o3d.geometry.PointCloud()
        cum_pc.points = o3d.utility.Vector3dVector(cum_pts_np)

       # TODO im sure there is a numpy fn of this
        tf_old = np.eye(4)
        global_transforms = []
        for tf in transforms_db:
            tf_old = np.matmul(tf,tf_old)
            global_transforms.append(tf_old)
        global_transforms = np.array(global_transforms)
       
        min_x = int(np.min(global_transforms[:,0,3]))
        max_x = int(np.max(global_transforms[:,0,3]))
        min_y = int(np.min(global_transforms[:,1,3]))
        max_y = int(np.max(global_transforms[:,1,3]))
            
        sample_frame = frame_db[0].rgb
        global_w = max(-min_x ,0) + sample_frame.shape[1]
        global_h = max(-min_y ,0) + sample_frame.shape[0]
        
        # transform to new origin
        min_transf = np.eye(4)
        min_transf[0,3] = -min_x
        min_transf[1,3] = -min_y
        global_transforms = np.matmul(min_transf, global_transforms)
        cum_pc.transform(min_transf)

        global_bgr = np.ones((global_h, global_w, 3)) *255
        stiched_instance_mask = np.zeros((global_h, global_w), dtype=np.uint8) 
        
        if self.is_skip_first_frame_of_longleaf:
            start_frame = 1
        else:
            start_frame = 0
            first_transf= np.expand_dims(np.eye(4), axis=0)
            first_transf[0,0,3]=global_w - sample_frame.shape[1]
            first_transf[0,1,3]=global_h - sample_frame.shape[0]
            global_transforms=np.vstack([first_transf, global_transforms])

        if self.skip_last_n_frames_longleaf > 0:
            new_frame_db = frame_db[:-self.skip_last_n_frames_longleaf]
            new_global_transforms = global_transforms[:-self.skip_last_n_frames_longleaf]
            # check if empty 
            if len(new_frame_db) > 0:
                frame_db = new_frame_db
                global_transforms = new_global_transforms

        for frame, homo_transf in zip(frame_db[start_frame:], global_transforms):
            cv2_trans = np.eye(3)
            cv2_trans[0:2, 0:2] = homo_transf[0:2,0:2]
            cv2_trans[0,2] = homo_transf[0,3]
            cv2_trans[1,2] = homo_transf[1,3]
            transformed = cv2.warpPerspective(frame.rgb,cv2_trans, (global_w, global_h), flags=cv2.INTER_CUBIC)
            transformed[transformed == 0] = 255
            global_bgr_preped = cv2.cvtColor(global_bgr.astype(np.uint8), cv2.COLOR_RGB2RGBA).astype(float)
            transformed_preped = cv2.cvtColor(transformed.astype(np.uint8), cv2.COLOR_RGB2RGBA).astype(float)
            global_bgr = darken_only(global_bgr_preped , transformed_preped, 0.7).astype(np.uint8)
            
            new_instance_mask = frame.instance_mask.astype(np.uint8)
            new_instance_mask[new_instance_mask == 0 ] = 125
            new_instance_mask[new_instance_mask == 1 ] = 255
            instance_transformed = cv2.warpPerspective(new_instance_mask ,cv2_trans, (global_w, global_h), flags=cv2.INTER_CUBIC)
            empty_mask = instance_transformed!=0
            stiched_instance_mask[empty_mask] = instance_transformed[empty_mask] # take the latest slice to avoid double counting in the case where transforms are wrong

        stiched_instance_mask[stiched_instance_mask==125] = 0

        if self.is_vis:
            global_bgr_vis = copy.deepcopy(global_bgr)
            # draw pointclouds too
            cum_pxnp = self.pc2px_np(cum_pc)
            cum_pxnp_x = cum_pxnp[:,0]
            cum_pxnp_y = cum_pxnp[:,1]

            global_bgr_vis[cum_pxnp_y, cum_pxnp_x] = (255,0,0,255)  # cum pcd in blue
            cv2.imwrite(f"{self.pics_dir_fp}/../test_dbg/test_{obj_id}_lai_ll.png", global_bgr_vis)
            mit_mask = multiply(global_bgr_vis.astype(float), np.expand_dims(stiched_instance_mask, -1).astype(float)/255 * np.array([0,0,255,255]), 0.5)
            cv2.imwrite(f"{self.pics_dir_fp}/../test_dbg/test_{obj_id}_lai_bb.png", mit_mask.astype(np.uint8))

        stiched_leaf_contours, hierarchy = cv2.findContours(
            stiched_instance_mask*255, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        stiched_instance_mask = stiched_instance_mask != 0  # change to bool 

        # create stiched instance mask
        old_frame = frame_db[0]
        stiched_frame = LeafFrame(
                rgb = global_bgr,
                instance_mask = stiched_instance_mask,
                contour = stiched_leaf_contours,
                frame_partiality = False,
                transform_from_prev = old_frame.transform_from_prev,
                tracker_id = old_frame.tracker_id
                )
        return stiched_frame 


    def icp(self, prev_contour, prev_stupid_contours, curr_contour, stupid_contours, thres, debug=False):
        # Find contours:
        contours, hierarchy = cv2.findContours(
            frame.instance_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        
        contours = [x[x[:,0,0] > self.border_cut_thres] for x in contours]
        stupid_contours = [x[:,0,0] <= self.border_ignore_thres for x in contours]
                
        dists = torch.cdist(
            torch.tensor(prev_contour).float(), torch.tensor(curr_contour).float()
        )
        neighbors = torch.argmin(dists, dim=0)
        min_dists = dists.min(dim=0)[0]
        # filter duplicate associations
        duplicate_mask = torch.zeros_like(neighbors, dtype=torch.bool)
        unique_neighbs = neighbors.unique()
        for neighb in unique_neighbs:
            mask = neighbors == neighb
            
            tmp_mask = torch.zeros(mask.sum(), dtype=torch.bool)
            tmp_mask[min_dists[mask].argmin()] = 1
            duplicate_mask[mask] = tmp_mask
            
        valid_associations = min_dists < thres
        valid_associations = torch.logical_and(valid_associations, duplicate_mask)
        transl = (curr_contour - prev_contour[neighbors])[valid_associations].mean(0)
        
        # curr->prev
        eval_dim = 1
        neighbors = torch.argmin(dists, dim=eval_dim)
        min_dists = dists.min(dim=eval_dim)[0]
        # filter duplicate associations
        duplicate_mask = torch.zeros_like(neighbors, dtype=torch.bool)
        unique_neighbs = neighbors.unique()
        for neighb in unique_neighbs:
            mask = neighbors == neighb
            
            tmp_mask = torch.zeros(mask.sum(), dtype=torch.bool)
            tmp_mask[min_dists[mask].argmin()] = 1
            duplicate_mask[mask] = tmp_mask
            
        valid_associations = min_dists < thres
        valid_associations = torch.logical_and(valid_associations, duplicate_mask)
        
        transl += ((prev_contour - curr_contour[neighbors])[valid_associations].mean(0))*(-1)
        
        error = min_dists[valid_associations].mean().item()

        if debug:
            # debug viz
            plt.scatter(prev_contour[:, 0], prev_contour[:, 1])
            plt.scatter(curr_contour[:, 0], curr_contour[:, 1])
            starts = prev_contour[valid_associations]
            ends = curr_contour[neighbors][valid_associations]
            for p in range(len(starts)):
                plt.plot((starts[p][0], ends[p][0]), (starts[p][1], ends[p][1]))

            plt.show()

        goodness = valid_associations.sum() / len(valid_associations)
        return transl, error, goodness


    def calculate_lai(self, frame, scale):
        lai = (frame.instance_mask > 0).sum() * scale
        return lai


def calibrate(frame, width_corners=10, height_corners=7):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if (gray < 30).sum() > 500000:
        ret, corners = cv2.findChessboardCorners(
            gray, (width_corners, height_corners), None
        )
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            tl = corners2[0]
            tr = corners2[width_corners - 1]
            bl = corners2[-width_corners]
            br = corners2[-1]
            width = np.linalg.norm(tl - tr)
            width2 = np.linalg.norm(bl - br)
            width_scale = 0.0212 / (((width + width2) / 2) / (width_corners - 1))
            heigth = np.linalg.norm(tl - bl)
            heigth2 = np.linalg.norm(tr - br)
            height_scale = 0.0212 / (((heigth + heigth2) / 2) / (height_corners - 1))
            scale = (width_scale + height_scale) / 2
            return scale
    return False

def detect_checkerboard(image, pattern_size=(7, 7), threshold=100):
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
            return True
    return False

    
def generate_veg_mask(frame, lower_veg_bound, upper_veg_bound):
    # convert to hsv
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    veg_mask = cv2.inRange(hsv_frame, lower_veg_bound, upper_veg_bound)
    return veg_mask

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
    
def get_roi_coords(image, top, right, bottom, left):
    # compute the roi coordinates by cutting top, right, bottom, left pixels of the image
    x1 = left
    y1 = top
    x2 = image.shape[1] - right
    y2 = image.shape[0] - bottom
    return (x1, y1, x2, y2)


@click.command()
@click.option(
    "--data_folder",
    "-d",
    type=str,
    help="path to the data folder",
    default="./data/leaves",
)
@click.option(
    "--output_folder",
    "-o",
    type=str,
    help="path to the output folder",
    default="./",
)
@click.option('--debug', is_flag=True)
@click.option('--is_vis', "--vis", is_flag=True)
@click.option(
        '--vm_dir', 
        type=str,
        default="",
        help ="path to dir")
@click.option(
        "--calib_yaml",
        type=str,
        default="./data/calibration.yaml",
        help ="path to the calibration.yaml used to crop the image")

@click.option(
        "--calib_txt",
        type=str,
        default="./data/calib.txt",
        help ="path to the calib.txt for camera calib")
@click.option(
        "--tracker_cfg",
        "-t",
        type=str,
        default="./data/tracker_hyps.cfg",
        help ="path to the tracker configuration")
def main(data_folder, output_folder, debug, is_vis, vm_dir, calib_yaml, calib_txt, tracker_cfg):
    cfg_dict = locals()

    pics_dir = data_folder
    output_csv_fp = os.path.join(output_folder, f"la.csv")
    gui_calib_fp = calib_yaml
    scale_calib_fp = calib_txt

    process_one_dir(pics_dir, output_csv_fp, gui_calib_fp, scale_calib_fp, debug, is_vis, vm_dir, cfg_dict, tracker_cfg)


def process_one_dir(pics_dir_fp, output_csv_fp, calib_fp, scale_calib_fp, debug, is_vis, vm_dir, cfg_dict, tracker_cfg_fp):
    cfg_dict_onedir = locals()
    print(f"processing dir: {pics_dir_fp}...")
    if is_vis:
        os.makedirs(f"{pics_dir_fp}/../test_dbg", exist_ok=True)
        os.makedirs(f"{pics_dir_fp}/../vis_frames", exist_ok=True)

    # for posterior, write config to file
    cfg_f = open(f"{pics_dir_fp}/../cmd_args.txt", "a")
    yaml.dump(cfg_dict_onedir, cfg_f, allow_unicode=True)
    cfg_f.close()

    # read tracker params from cfg file
    tracker_config = configparser.ConfigParser()
    tracker_config.read(tracker_cfg_fp)

    vel = np.array([
        tracker_config['TRACKER'].getfloat('conveyor_vel_x'), 
        tracker_config['TRACKER'].getfloat('conveyor_vel_y')])

    frame_list = os.listdir(pics_dir_fp)

    frame_ids = [int(x.split("_")[2]) for x in frame_list]
    frame_list = [x for _, x in sorted(zip(frame_ids, frame_list))]

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

    # read scale from txt
    scale_calib = open(scale_calib_fp, 'r')
    scale = float(scale_calib.readline()) 
    if scale < 0:
        print(f"Error: scale of mm/px read from {scale_calib_fp} is less than zero at {scale}. using default scale instead...")
        scale = 1.647780615736605585e-04
    scale = scale ** 2  # scale in area

    scale_calib.close()

    leaf_tracker = LeafTracker(
        conveyor_velocity=vel,
        scale = scale,
        debug=debug,
        birth_roi_buffer=tracker_config["TRACKER"].getint('birth_roi_buffer'),
        min_age=tracker_config["TRACKER"].getint('min_age'),
        delta_sx = tracker_config["TRACKER"].getfloat('delta_sx'),
        step_delta_sy = tracker_config["TRACKER"].getfloat('step_delta_sy'),
        sift_nfts = tracker_config["LONG_LEAF_STICHING"].getint('sift_nfts')
    )

    # frame_processor calculates the lai and also adds it to its db
    if  tracker_config.has_option('TRACKER',"skip_last_n_frames_longleaf"):
        skip_last_n_frames_longleaf=tracker_config['TRACKER'].getint('skip_last_n_frames_longleaf')
    else:
        skip_last_n_frames_longleaf=0
    frame_processor = FrameProcessor(
        scale=scale,
        debug=debug,
        is_vis=is_vis,
        pics_dir = pics_dir_fp,
        skip_first_frame_longleaf= tracker_config['TRACKER'].getboolean('skip_first_frame_longleaf'), # DBG only for when there are a lot of corrupt frames
        skip_last_n_frames_longleaf=skip_last_n_frames_longleaf
    )

    prev_vm = None  # used to estimate vel 
    for frame_path in tqdm(frame_list):

        # load frame
        frame_ori = cv2.imread(os.path.join(pics_dir_fp, frame_path))

        # crop based on GUI code 
        frame = frame_ori[crop_y: crop_y + crop_h, crop_x: crop_x + crop_w]
        
        if vm_dir != "":
            veg_mask_read = cv2.imread(os.path.join(vm_dir, frame_path), cv2.IMREAD_UNCHANGED)
            if veg_mask_read is None:
                print("Error! no vegetation mask found")
            veg_mask = veg_mask_read[:,:,1].astype(np.uint8) * 255  
        else:
            veg_mask = generate_veg_mask(frame, lower_veg_bound_hsv, upper_veg_bound_hsv)

        time_now = int(frame_path.split("_")[2]) 

        # return list of leaf tracks and list of dead tracker_obj
        completed_tracks, leaf_frames = leaf_tracker.update(frame=frame, prev_veg_mask = prev_vm, veg_mask=veg_mask, time_now=time_now, frame_path=frame_path)
        prev_vm  = copy.deepcopy(veg_mask)

        if is_vis:
            # per frame vis 
            cv2.imwrite(f"{pics_dir_fp}/../vis_frames/vis_{frame_path}", leaf_tracker.tracker.vis_curr_frame())
        for track in completed_tracks:
        # register each track as a leaf 
            if is_vis:
                track.vis_pred_curr(f"{pics_dir_fp}/../test_dbg/test_{track.obj_id}.png")

        # postprocess completed tracks to get the lai
        frame_processor.process_leaves(leaf_frames)  
      
    csv_f = open(output_csv_fp, 'w')
    write = csv.writer(csv_f)
    write.writerow(["tracker_id", "lai_mean", "lai_std"])
    write.writerows(np.array([frame_processor.obj_id_db, frame_processor.lais_db , frame_processor.lais_std_db]).T.tolist())

    csv_f.close()
    print(f"finished processing dir: {pics_dir_fp}.")

if __name__ == "__main__":
    main()
