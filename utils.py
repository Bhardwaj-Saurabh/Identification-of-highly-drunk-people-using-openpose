import numpy as np
from numba import jit
import itertools
from scipy.spatial import distance

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return izip(a, b)

def Updated_keypoint():
    keypoints = np.array(datum.poseKeypoints)
    if keypoints.all() != None:
        scores = keypoints[:,:,2]
        idx = []
        for i in range(len(scores)):
            nose = scores[i][0]
            RAnkle = scores[i][11]
            LAnkle = scores[i][14]
            RShoulder = scores[i][2]
            LShoulder = scores[i][5]

            if nose < 0.1  and RAnkle < 0.5 and LAnkle < 0.5 and RShoulder < 0.2\
             and LShoulder < 0.2:
                idx.append(i)

        update_kp = np.delete(keypoints, idx, 0)
        return update_kp
    else:
        return None

def pose2box(poses):
    global seen_bodyparts
    """
    Parameters
    ----------
    poses: ndarray of human 2D poses [People * BodyPart]
    Returns
    ----------
    boxes: ndarray of containing boxes [People * [x1,y1,x2,y2]]
    """
    
    boxes = []
    for person in poses:
        x = []
        y = []
        seen_bodyparts = person[np.where((person[:,0] != 0) | (person[:,1] != 0))]
        for i, j in seen_bodyparts:
            x.append(i)
            y.append(j)

        #print(nose, RShoulder, LShoulder, RAnkle, LAnkle)

        x1 = int(min(x))
        x2 = int(max(x))
        y1 = int(min(y))
        y2 = int(max(y))
        
        if x1 > 0 and x2 > 0 and y1 > 0 and y2 > 0:
            box = [x1, y1 -20, x2, y2 + 5]
            boxes.append(box)
    return np.array(boxes)

def updated_boxes(boxes, starting_line, frame_height):

    update_box = []
    for i in range(len(boxes)):
        x1,y1,x2,y2 = boxes[i]
        h = y2 - y1
        if h > 50 and y2 > starting_line:
            update_box.append(boxes[i])

    return update_box

def calc_motion_efficiency(centre):
    steps = len(centre)
    disp = distance.euclidean(centre[0], centre[-1])
    #print(disp)
    moving_length = 0

    for step in range(0, steps-1):
        dist_step = distance.euclidean(centre[step], centre[step + 1])
        #print(dist_step)
        moving_length += dist_step

    #print(moving_length)
    
    motion_efficiency = disp/moving_length
    return motion_efficiency

def cal_S(lambd_pt, T_mh, T_ml, sigma_m, sm_p):

    if lambd_pt >= T_mh:
        sm = sm_p + lambd_pt
    elif lambd_pt < T_ml:
        sm = sm_p - sigma_m
    else:
        sm = sm_p

    if sm >= 0:
        return sm
    else:
        return 0

def distancia_midpoints(mid1, mid2):
    return np.linalg.norm(np.array(mid1)-np.array(mid2))

def pose2midpoint(pose):
    """
    Parameters
    ----------
    poses: ndarray of human 2D pose [BodyPart]
    Returns
    ----------
    boxes: pose midpint [x,y]
    """
    box = poses2boxes([pose])[0]
    midpoint = [np.mean([box[0],box[2]]), np.mean([box[1],box[3]])]
    return np.array(midpoint)

@jit
def iou(bb_test,bb_gt):
    """
    Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
        + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    return(o)