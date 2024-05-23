"""
to deal with (apparent) symmetries of tshirts, we order the keypoints in a consistent way.

cf. https://arxiv.org/abs/2401.01734
"""

import numpy as np

TSHIRT_KEYPOINTS = [
    "shoulder_left",
    "neck_left",
    "neck_right",
    "shoulder_right",
    "sleeve_right_top",
    "sleeve_right_bottom",
    "armpit_right",
    "waist_right",
    "waist_left",
    "armpit_left",
    "sleeve_left_bottom",
    "sleeve_left_top",
]


def order_tshirt_keypoints(keypoints_2D: np.ndarray, bbox: tuple):

    # left == side of which the waist kp is closest to the bottom left corner of the bbox in 2D.
    # simply serves to break symmetries and find adjacent keypoints, does not correspond with human notion of left and right,
    # which is determind in 3D. This can be determiend later in the pipeline if desired, once the 2D keypoints are lifted to 3D somehow.
    # we use waist kp as this one has been estimated to be least deformed by real pipelines.

    x_min, y_min, width, height = bbox

    bottom_left_bbox_corner = (x_min, y_min + height)

    waist_left_idx = TSHIRT_KEYPOINTS.index("waist_left")
    waist_right_idx = TSHIRT_KEYPOINTS.index("waist_right")
    waist_left_2D = keypoints_2D[waist_left_idx]
    waist_right_2D = keypoints_2D[waist_right_idx]

    distance_waist_left = np.linalg.norm(np.array(waist_left_2D) - np.array(bottom_left_bbox_corner))
    distance_waist_right = np.linalg.norm(np.array(waist_right_2D) - np.array(bottom_left_bbox_corner))

    if distance_waist_left > distance_waist_right:
        should_tshirt_be_flipped = True
    else:
        should_tshirt_be_flipped = False
    # print(f"should_tshirt_be_flipped: {should_tshirt_be_flipped}")
    if should_tshirt_be_flipped:
        for idx, keypoint in enumerate(TSHIRT_KEYPOINTS):
            if "left" in keypoint:
                right_idx = TSHIRT_KEYPOINTS.index(keypoint.replace("left", "right"))
                # print(f"swapping {keypoint} with {TSHIRT_KEYPOINTS[right_idx]}")
                # swap the rows in the numpy array, cannot do this as with lists
                # https://stackoverflow.com/questions/21288044/row-exchange-in-numpy
                keypoints_2D[[idx, right_idx]] = keypoints_2D[[right_idx, idx]]

    return keypoints_2D
