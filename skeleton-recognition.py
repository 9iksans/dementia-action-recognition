from cubemos.core.nativewrapper import CM_TargetComputeDevice
from cubemos.core.nativewrapper import initialise_logging, CM_LogLevel
from cubemos.skeleton_tracking.nativewrapper import Api, SkeletonKeypoints
import cv2
import argparse
import csv
import os
import numpy as np
import platform
from pprint import pprint

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
keypoint_ids = [
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
]

keypoint_ex= [
    (0, 0),
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 4),
    (5, 5),
    (6, 6),
    (7, 7),
    (8, 8),
    (9, 9),
    (10, 10),
    (11, 11),
    (12, 12),
    (13, 13),
    (14, 14),
    (15, 15),
    (16, 16),
    (17, 17),
]

def default_license_dir():
    if platform.system() == "Windows":
        return os.path.join(os.environ["LOCALAPPDATA"], "Cubemos", "SkeletonTracking", "license")
    elif platform.system() == "Linux":
        return os.path.join(os.environ["HOME"], ".cubemos", "skeleton_tracking", "license")
    else:
        raise Exception("{} is not supported".format(platform.system()))


sdk_path = os.environ["CUBEMOS_SKEL_SDK"]

# if args.verbose:
#         initialise_logging(sdk_path, CM_LogLevel.CM_LL_DEBUG, True, default_log_dir())

api = Api(default_license_dir())
model_path = os.path.join(
    sdk_path, "models", "skeleton-tracking", "fp32", "skeleton-tracking.cubemos"
)

api.load_model(CM_TargetComputeDevice.CM_CPU, model_path)

def get_valid_limbs(keypoint_ids, skeleton, confidence_threshold):
    limbs = [
        (tuple(map(int, skeleton.joints[i])), tuple(map(int, skeleton.joints[v])))
        for (i, v) in keypoint_ids
        if skeleton.confidences[i] >= confidence_threshold
        and skeleton.confidences[v] >= confidence_threshold
    ]
    valid_limbs = [
        limb
        for limb in limbs
        if limb[0][0] >= 0 and limb[0][1] >= 0 and limb[1][0] >= 0 and limb[1][1] >= 0
    ]
    return valid_limbs

def render_result(skeletons, img, confidence_threshold):
    skeleton_color = (100, 254, 213)
    fields = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17"]
    for index, skeleton in enumerate(skeletons):
        limbs = get_valid_limbs(keypoint_ids, skeleton, confidence_threshold)
        for limb in limbs:
            cv2.line(
                img, limb[0], limb[1], skeleton_color, thickness=2, lineType=cv2.LINE_AA
                
            )

def export_csv(skeleton,img,confidence_threshold):
    stackarray = ()
    for index, skeleton in enumerate(skeletons):
        limbs = get_valid_limbs(keypoint_ex, skeleton, confidence_threshold)
        for limb in limbs:
            stackarray = stackarray + limb[0]
        print(np.asarray(stackarray))
        with open(r'foo.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(np.asarray(stackarray))
            
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # perform inference
    skeletons = api.estimate_keypoints(frame, 192)

    new_skeletons = api.estimate_keypoints(frame, 192)
    new_skeletons = api.update_tracking_id(skeletons, new_skeletons)

    render_result(skeletons, frame, 0.5)
    export_csv(skeletons, frame, 0.5)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#cek
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
