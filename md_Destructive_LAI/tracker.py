#!/usr/bin/env python3
"""Class header for tracker stuff
"""
import pdb
from collections import namedtuple
import cv2
import numpy as np
import copy
import random

from utils import get_exR
from scipy.optimize import linear_sum_assignment


def blend_images(image1: np.ndarray, image2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """ Blend two images 
  
    Args:
        image1 (np.ndarray): 1st image of shape [H x W x 3]
        image2 (np.ndarray): 2nd image of shape [H x W x 3]
        alpha (float, optional): strength of blending for 1st image. Defaults to 0.5.
  
    Returns:
        np.ndarray: blended image of shape [H x W x 3]
    """
    assert alpha <= 1.0
    assert alpha >= 0.0
    assert image1.shape == image2.shape
  
    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)
  
    blended = alpha * image1 + (1 - alpha) * image2
    blended = np.round(blended).astype(np.uint8)
  
    return blended


ROI = namedtuple("ROI", "min_x, min_y, max_x, max_y")

class Tracker:
    """Tracker to track stuffs
    """

    def __init__(self, vx, vy, min_num_px_overlap, max_lifetime=4, timeout = 30,  birthing_ROI=None, is_debug=False):
        self.is_debug = is_debug
        self.tracks_dict = {}
        self.max_lifetime = max_lifetime
        self.min_num_px_overlap = min_num_px_overlap 
        self.obj_id_count = -1
        self.time_now = -1
        self.min_obj_size = 500  # TODO make args
        self.bgr_frame = None
        self.timeout = timeout

        # motion model stuff
        self.one_step_vx = vx
        self.one_step_vy = vy

        if birthing_ROI is None:
            self.birthing_ROI = ROI(0, 0, -1, -1)
        else:
            self.birthing_ROI = birthing_ROI

    def update(self, frame, time_now, frame_fp=None, frame_bgr=None, vx=0, vy=0):
        killed_list = self.cull_objects(time_now)
        self.update_time_now(time_now, frame=frame, vx=vx, vy=vy)
        if not frame_bgr is None:
            self.bgr_frame = frame_bgr
        birthed_list = self.add_observation(frame, frame_fp, frame_bgr, vx, vy)
        return birthed_list, killed_list

    def update_time_now(self, time_now, frame_fp=None, frame=None, vx=0, vy=0):
        self.time_now = time_now
        for obj_id in self.tracks_dict:
            # apply motion model
            new_timedstate= TimestampedState(self.time_now, self.motion_model(self.time_now, self.tracks_dict[obj_id], 0, 0))
            self.tracks_dict[obj_id].add_timestamped_state(new_timedstate, frame_fp, frame)
            self.tracks_dict[obj_id].update_age(self.time_now)

    def add_observation(self, frame, frame_fp=None, frame_bgr=None, vx=0, vy=0):
        """Update observation to tracker
        DOES NOT KILL OBJECTS, ONLY ADDS
        """
        obs_objects = self.get_objects_in_frame(frame)
        unassociated_obs = self.associate(obs_objects, frame, frame_fp, frame_bgr, vx, vy)
        birthed_list = self.safely_birth(unassociated_obs, frame, frame_fp, frame_bgr)
        return birthed_list

    def is_obj(self, frame, min_pxs = 0):
        # for pink tag vs bg
        is_obj = frame.sum() > 0  # remove bg 

        # remove small objects as well 
        if is_obj:
            is_obj = (frame > 0).sum()  > min_pxs
        return is_obj

    def get_objects_in_frame(self, frame):
        obs_list = []
        # frame is already thresholded in this case so i dont need to do it again but i might need to change this in the future
        num_components, labels, stats, centroids = cv2.connectedComponentsWithStats(frame)
        if num_components > 1: # if there is more than one component (i.e., not just the bg component)
            for i in range(num_components):
                label_bool_mask = labels == i
                if self.is_obj(frame[label_bool_mask], min_pxs=10):
                    obs_state = State(
                        cx=centroids[i][0],
                        cy=centroids[i][1],
                        features = label_bool_mask
                        )
                    obs_list.append(obs_state)
        if self.is_debug:
            cv2.imwrite(f"obs_{len(self.tracks_dict)}.png", frame.astype(np.uint8)*255)

        return obs_list

    def motion_model(self, time_now, obj, vel_x, vel_y):
        prev_state = obj.get_state()
        prev_feat = prev_state.features
        delta_time = time_now - obj.timestamped_state_list[-1].timestamp
        new_cx = prev_state.cx + vel_x * delta_time
        new_cy = prev_state.cy + vel_y * delta_time

        new_features = move_frame(prev_feat, vel_x, vel_y, delta_time)

        if self.is_debug:
            cv2.imwrite(f"prev_{len(self.tracks_dict)}.png", prev_feat.astype(np.uint8)*255)
            cv2.imwrite(f"new_{len(self.tracks_dict)}.png", new_features.astype(np.uint8)*255)

        pred_state = State(
                        cx=new_cx,
                        cy=new_cy,
                        features = new_features,
                        is_pred = True,
                        vx = vel_x,
                        vy = vel_y
                )

        return pred_state

    def vis_curr_frame(self):
        img = copy.deepcopy(self.bgr_frame)
        for obj_id in self.tracks_dict:
            track = self.tracks_dict[obj_id]  

            # draw bb 
            tl, overlay_mask = draw_bb(track.get_state().features, img, track.clr) 
            img = cv2.putText(img, str(obj_id), tl, cv2.FONT_HERSHEY_SIMPLEX , 1, track.clr)

        return img


    def overlap(self, worker, job):
        overlap = np.logical_and(worker.get_state().features, job.features)  # overlap of pred and curr obs
        cost = np.bitwise_not(overlap).sum() 
        return cost

    def SIFT_matches(self, worker, job, max_matches=1000, vx=0, vy=0):
        curr_bgr = self.bgr_frame
        past_bgr = worker.bgr_list[-1]  

        past_mask = worker.get_state().features
        curr_mask = job.features

        if curr_mask.sum() < self.min_num_px_overlap:
            return max_matches

        curr_instance_bgr = np.zeros(curr_bgr.shape, dtype=np.uint8)
        curr_instance_bgr[curr_mask] = curr_bgr[curr_mask]

        past_instance_bgr = np.zeros(past_bgr.shape, dtype=np.uint8)
        past_instance_bgr[past_mask.astype(bool)] = past_bgr[past_mask.astype(bool)]

        sift = cv2.SIFT_create(
                        nfeatures = 500,
                        contrastThreshold = 0.01,
                        edgeThreshold= 100,)

        curr_kp = sift.detect(curr_instance_bgr,None)
        curr_kp, curr_des = sift.compute(curr_instance_bgr, curr_kp)

        past_kp = sift.detect(past_instance_bgr,None)
        past_kp, past_des = sift.compute(past_instance_bgr, past_kp)

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        if curr_des is None or past_des is None:
            return max_matches

        matches_bf = bf.match(curr_des, past_des)
        
        image_matches_bf = cv2.drawMatches(curr_bgr, curr_kp, past_bgr, past_kp, matches_bf, None)
        # cv2.imwrite(f"{self.time_now}_test.png", image_matches_bf)

        s_list = []
        for match in matches_bf[:]:
            p1 = np.array(curr_kp[match.queryIdx].pt)
            p2 = np.array(past_kp[match.trainIdx].pt)
            s = np.expand_dims(p2 - p1, -1)
            s_list.append(s) 
        s_np = np.array(s_list)[:,:,0]

        # TODO make into a function
        delta_sx = 200 
        delta_sy = 50 

        # use motion model to help with removing outliers
        min_q = -vx - delta_sx
        max_q = -vx + delta_sx
        s_np = s_np[np.logical_and(s_np[:,0] > min_q, s_np[:,0] < max_q)]
    
        min_q = -vy - delta_sy
        max_q = -vy + delta_sy
        s_np = s_np[np.logical_and(s_np[:,1] > min_q, s_np[:,1] < max_q)]

        cost = max(0, max_matches - len(s_np))
        # print("cost", cost)

        return cost


    def associate(self, obs_list, frame, frame_fp, frame_bgr=None, vx=0, vy=0):
        """does association and update of given obs to preds
        returns unassociated objects
        DOES NOT RUN PREDS, only associates
        """
    
        # create HA cost table
        num_workers = len(self.tracks_dict)
        num_jobs = len(obs_list)
        
        # TODO make it a dict from the start
        obs_dict= {}
        for obs_i, obs in enumerate(obs_list):
            obs_dict[str(obs_i)] = obs

        if num_workers <= 0:
            return obs_dict

        cost_mat = np.zeros((num_workers, num_jobs))
        worker_key_list = list(self.tracks_dict)
        max_matches = 1000

        for w_i, worker_key in enumerate(worker_key_list):
            for j_i, job in enumerate(obs_list):
                worker=self.tracks_dict[worker_key]
                # cost = self.overlap(worker, job)
                cost = self.SIFT_matches(worker, job, max_matches, vx, vy)
                cost_mat[w_i, j_i] = cost

        row_indices, col_indices = linear_sum_assignment(cost_mat)

        # Extract the optimal assignment
        assignments = [(row, col) for row, col in zip(row_indices, col_indices)]
        
        for assignment in assignments:
            w_i = assignment[0]
            j_i = assignment[1]

            # check if min threshold is met 
            # min_cost = job.features.shape[0]*job.features.shape[1] - self.min_num_px_overlap
            min_matches = 1  # min num of sift assocs   # TODO config for hyperparam
            max_cost = max_matches - min_matches
            if cost_mat[w_i, j_i] < max_cost:
                # associate the object. update the track with new observation
                obs = obs_dict[str(j_i)]
                track = self.tracks_dict[worker_key_list[w_i]]

                obs.vx = vx
                obs.vy = vy
                t_state = TimestampedState(self.time_now, obs)
                track.add_timestamped_state(t_state, frame_fp, frame, frame_bgr)

                obs_dict.pop(str(j_i))  # pop from obs_dict

        return obs_dict

    def safely_birth(self, unassociated_obs, frame, frame_fp, frame_bgr=None):
        birthed_list = []
        # check if the obj is in the birthing ROI
        for obs_i_key, obs in unassociated_obs.items():
            if obs.is_inside_roi(self.birthing_ROI) and obs.size() > self.min_obj_size:
                # actually create and birth obj 
                obj_id = self.next_obj_id()
                clr = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                obj = self.obs2obj(obs, self.time_now, str(obj_id), frame, frame_fp, frame_bgr, obj_id, clr)
                self.tracks_dict[obj_id] = obj
                birthed_list.append(copy.deepcopy(obj))
        return birthed_list

    def obs2obj(self, obs, time, name_str, frame, frame_fp, frame_bgr=None, obj_id=-1, clr=(255,0,255)):
        timed_state = TimestampedState(time, obs)
        obj = Tracker_obj(timed_state, name_str, frame, frame_fp, self.is_debug,frame_bgr=frame_bgr, obj_id=obj_id,clr=clr)
        return obj

    def next_obj_id(self):
        self.obj_id_count +=1
        return self.obj_id_count

    def kill_object(self, obj_key):
        if self.is_debug:
            print(f"killing obj with id {obj_key}")
        kill_obj_copy = copy.deepcopy(self.tracks_dict[obj_key])
        self.tracks_dict.pop(obj_key)

        return kill_obj_copy

    def cull_objects(self, timenow):
        tracks_dict_keys = list(self.tracks_dict)
        killed_list = []
        dt = Tracker_obj.get_delta_time(self.time_now, timenow)

        if dt >= self.timeout:
            for obj_id in tracks_dict_keys:
                killed_obj = self.kill_object(obj_id)
                killed_list.append(killed_obj)
        else:
            for obj_id in tracks_dict_keys:
                obj = self.tracks_dict[obj_id]
                if obj.age > self.max_lifetime:
                    killed_obj = self.kill_object(obj_id)
                    killed_list.append(killed_obj)
                else:
                    # also check if obj has been associated or not since the last timestep. if it has been lost for too long, kill it.
                    last_track = self.tracks_dict[obj_id].get_state()
                    if last_track.is_pred:
                        killed_obj = self.kill_object(obj_id)
                        killed_list.append(killed_obj)
                    else:
                        # check if object is still in frame 
                        fts = self.tracks_dict[obj_id].get_state().features
                        min_x = min(np.where(fts)[1])
                        if dt * self.one_step_vx + min_x >  fts.shape[1]:  # object has gone out of frame
                            # print(f"obj {obj_id} should be out of bounds so we are killing it.")
                            killed_obj = self.kill_object(obj_id)
                            killed_list.append(killed_obj)
                            # TODO also check bounds in y for code generalisability

        return killed_list


# State = namedtuple("State", "cx, cy, features")
class State:
    def __init__(self, cx, cy, features, is_pred=False, vx=0 , vy=0):
        self.cx = cx
        self.cy = cy
        self.features = features
        self.is_pred = is_pred
        self.vx = vx 
        self.vy = vy

    def is_inside_roi(self, roi):
        # TODO this is specific to the pink tag case and needs to be changed for the leaf case 
        roi_mask = np.zeros(self.features.shape)
        roi_mask[roi.min_y:roi.max_y, roi.min_x:roi.max_x]=1
        is_inside = np.logical_and(self.features,roi_mask).any()
        return is_inside

    def size(self):
        return self.features.sum()


    def __str__(self):
        return f"cx: {self.cx}, cy:{self.cy}, features:{self.features}, is_pred:{self.is_pred}"

def move_frame(prev_feat_bool, vel_x, vel_y, delta_time):
    # move the features based on vx and vy
    translation_mat = np.float32([
                [1, 0, vel_x * delta_time],
                [0, 1, vel_y * delta_time]
                ])
    new_features = cv2.warpAffine(prev_feat_bool.astype(np.uint8), translation_mat, (prev_feat_bool.shape[1], prev_feat_bool.shape[0]))
    return new_features 

def estimate_vel(prev_frame, curr_frame, step_vel_x, step_vel_y, delta_time):
    """
    step_vel is the speed per step
    """
    # estimate the number of steps
    frame_width = curr_frame.shape[1]
    max_steps =  int(frame_width // step_vel_x) + 1
    best_step = delta_time
    best_overlap = 0
    buffer_steps = 0  # how far back in time we can look
    start_steps = max(1, delta_time-buffer_steps)
    for step in range(start_steps, max_steps):
        prev_frame_update = move_frame(prev_frame, step_vel_x, step_vel_y, step)  # move frame 
        curr_frame_update = move_frame(curr_frame, -step_vel_x, -step_vel_y, step)  # remove part of the curr frame that cannot be seen in the prev_frame
        curr_frame_update = move_frame(curr_frame_update, step_vel_x, step_vel_y, step)
        overlap = overlap_count(prev_frame_update, curr_frame_update)   
        if overlap > best_overlap:
            best_step = step
            best_overlap = overlap

    vx = best_step * step_vel_x
    vy = best_step * step_vel_y
    return vx, vy

def overlap_count(frame1, frame2):
    count = np.logical_and(frame1, frame2).sum()
    # IoU
    union = np.logical_or(frame1, frame2).sum()
    if union > 0:
        iou = count / union
    else:
        iou = 0 
    return iou

TimestampedState = namedtuple("TimestampedState", "timestamp, state")

def draw_bb(img_mask, img_prev, bb_clr):
    if img_mask.sum() > 0: 
        where_x, where_y = np.where(img_mask)
        max_x = where_x.max()
        min_x = where_x.min()
        max_y = where_y.max()
        min_y = where_y.min()
        img_prev = cv2.rectangle(img_prev, (min_y, min_x), (max_y, max_x), bb_clr, thickness=5)
    else:
        min_y = min_x = 0
    return (min_y, min_x), img_prev


class Tracker_obj:
    """Single object being tracked
    """
    def __init__(self, timestamped_init_state, name_str, frame, frame_fp=None, is_debug=False, frame_bgr=None, obj_id=-1, clr = (255,0,255)):
        self.name_str = name_str
        self.is_debug = is_debug

        self.timestamped_state_list = []  # stores existing states
        self.age = 0
        self.obj_id = obj_id

        self.frames_list = []  # for the full image the object is in. 
        self.frames_fp_list = []  # for the full image the object is in. 

        self.dt = 0  # time difference between last two obs frames. for transforms needed by icp 
        self.clr = clr

        self.add_timestamped_state(timestamped_init_state, frame_fp, None)
        self.bgr_list = []
        if frame_bgr is not None:
            self.bgr_list.append(frame_bgr)


    def vis_pred_curr(self, png_fp, show=False):
        obj_clr = self.clr
        img_prev = np.zeros( self.timestamped_state_list[0].state.features.shape)
        tl, img_prev = draw_bb(self.timestamped_state_list[0].state.features, img_prev, (255))
        img_prev = cv2.putText(img_prev, str(self.obj_id), tl, cv2.FONT_HERSHEY_SIMPLEX , 1, (255))

        for i in range(len(self.timestamped_state_list) -1):
            ft = self.timestamped_state_list[i+1].state.features
            img_curr = ft.astype(np.uint8) * 255

            # add bounding box 
            tl, img_prev = draw_bb(ft, img_prev, (255))

            # add text 
            if self.timestamped_state_list[i+1].state.is_pred:
                text = f"{self.obj_id}_pred"
                img_prev = blend_images(img_prev, img_curr)
            else:
                text = str(self.obj_id)
            img_prev = img_prev.astype(np.uint8)
            img_prev = cv2.putText(img_prev, text, tl, cv2.FONT_HERSHEY_SIMPLEX , 1, (255))
    
        # print also the filenames
        fps = ""
        y = 50
        for fp in self.frames_fp_list:
            img_prev = cv2.putText(img_prev, fp, (100,y), cv2.FONT_HERSHEY_SIMPLEX , 1, (255))
            y += 50

        ft_bgr = (np.expand_dims(np.ones(img_prev.shape),-1) * obj_clr ).astype(np.uint8)
        ft_alpha = np.expand_dims(img_prev,-1) / 255.

        # overlay the bgr images
        if len(self.bgr_list) > 0 :
            img_prev_bgr = np.zeros(self.bgr_list[0].shape)
            for frame_bgr in self.bgr_list:
                img_prev_bgr = blend_images(frame_bgr, img_prev_bgr)

            # combine the bgr wt the features
            # black_mask = img_prev == 0
            # ft_bgr[black_mask] = img_prev_bgr[black_mask]
            img_out = np.multiply(ft_bgr, ft_alpha) +  np.multiply(img_prev_bgr, ( 1. - ft_alpha)).astype(np.uint8)
        if show:
            cv2.imshow(png_fp, img_out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            cv2.imwrite(png_fp, img_out)
        return 

    def add_timestamped_state(self, timestamped_state, fp, frame, frame_bgr=None):
        self.timestamped_state_list.append(timestamped_state)
        self.update_age(timestamped_state.timestamp)
        if fp is not None:
            self.frames_fp_list.append(fp)
        self.frames_list.append(frame)
        if frame_bgr is not None:
            self.bgr_list.append(frame_bgr)
        self.dt = timestamped_state.timestamp - self.timestamped_state_list[-1].timestamp


    def update_age(self, time_now):
        self.age = Tracker_obj.get_delta_time(self.timestamped_state_list[0].timestamp, time_now)

    def get_delta_time(start_time, end_time):
        """Generic time count between two times
        """
        return end_time - start_time

    def get_last_delta_time(self):
        return self.dt

    def get_state(self):
        return self.timestamped_state_list[-1].state

def test_classes():
    test_state = State(
                        cx=0,
                        cy=0,
                        features = None
                      )
    test_timestampedstate = TimestampedState(timestamp = 0, state = test_state)

    test_tracker_obj = Tracker_obj(
                                    timestamped_init_state = test_timestampedstate,
                                    name_str = "test_obj",
                                    frame = None,
                                    frame_fp = frames_fp_list[0],
                                    is_debug = True
                                  )

