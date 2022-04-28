import numpy as np
from enum import IntEnum


class FilterMethod(IntEnum):
    RECTIFICATION = 0
    QUAD = 1
    PNP = 2


class Image:
    def __init__(self, mat, kp=None, desc=None):
        self.mat = mat
        self.kps = kp
        self.desc = desc  # descriptors
        self.inliers_kps_idx = np.zeros(1)
        self.quad_inliers_kps_idx = np.zeros(1)
        self.pnp_inliers_kps_idx = np.zeros(1)
        self.inliers_filter_funcs = [self.get_rectification_inliers_kps, self.get_quad_inliers_kps,
                                     self.get_pnp_inliers_kps]
        self.inliers_idx_filter_funcs = [self.get_rectification_inliers_idx,
                                         self.get_quad_inliers_kps_idx,
                                         self.get_pnp_inliers_kps_idx]
        self.outliers_filter_funcs = 1

    def __is_rgb(self):
        return self.mat.ndim == 3

    def get_inliers_kps(self, filter_kind: FilterMethod):
        return self.inliers_filter_funcs[filter_kind]()

    def get_inliers_kps_idx(self, filter_kind: FilterMethod):
        return self.inliers_idx_filter_funcs[filter_kind]()

    def get_ouliers_kps(self):
        return self.kps[self.get_outliers_idx()]

    def get_rectification_inliers_kps(self):
        return self.kps[self.inliers_kps_idx]

    def get_rectification_inliers_idx(self):
        return self.inliers_kps_idx

    def set_inliers_kps(self, new_inliers_kp_idx):
        self.inliers_kps_idx = new_inliers_kp_idx

    def get_outliers_idx(self):
        all_idx = {i for i in range(len(self.kps))}
        return np.array(all_idx - self.inliers_kps_idx)

    def get_rectification_outliers_kps(self):
        return self.kps[self.get_outliers_idx()]

    def get_quad_inliers_kps(self):
        return self.kps[self.quad_inliers_kps_idx]

    def get_quad_inliers_kps_idx(self):
        return self.quad_inliers_kps_idx

    def set_quad_inliers_kps_idx(self, quad_inliers_kps_idx):
        self.quad_inliers_kps_idx = quad_inliers_kps_idx

    def get_pnp_inliers_kps_idx(self):
        return self.pnp_inliers_kps_idx

    def get_pnp_inliers_kps(self):
        return self.get_quad_inliers_kps()[self.pnp_inliers_kps_idx]

    def set_pnp_inliers_kps_idx(self, pnp_inliers_kps_idx):
        self.pnp_inliers_kps_idx = pnp_inliers_kps_idx


class StereoPair:
    def __init__(self, left_image: Image, right_image: Image, idx, matches=None):
        self.left_image = left_image
        self.right_image = right_image
        self.idx = idx
        self.matches = matches
        self.__rectified_inliers_matches_idx = np.zeros(1)
        self.__quad_inliers_matches_idx = []
        self.pnp_inliers_matches_idx = []
        self.filter_matches_funcs = []
        self.left_right_kps_idx_dict = {}
        # self.filter_method = FilterMethod.NO_FILTER

    def get_rectified_inliers_matches(self):
        return self.matches[self.__rectified_inliers_matches_idx]

    def set_rectified_inliers_matches_idx(self, rectified_inliers_matches_idx):
        self.__rectified_inliers_matches_idx = rectified_inliers_matches_idx

    def get_rectified_outliers_matches_idx(self):
        return list(set(range(len(self.matches))) - set(
            self.__rectified_inliers_matches_idx))

    def set_quad_inliers_matches_idx(self, quad_inliers_matches_idx):
        self.__quad_inliers_matches_idx = quad_inliers_matches_idx

    def get_quad_inliers_matches(self):
        return self.matches[self.__rectified_inliers_matches_idx[self.__quad_inliers_matches_idx]]

    def get_quad_inliers_matches_idx(self):
        return self.__rectified_inliers_matches_idx[self.__quad_inliers_matches_idx]

    def get_pnp_outliers_matches_idx(self):
        return list(set(self.__rectified_inliers_matches_idx) - set(
            self.__rectified_inliers_matches_idx[self.pnp_inliers_matches_idx]))

    def set_left_right_kps_idx_dict(self, left_right_kps_idx_dict):
        self.left_right_kps_idx_dict = left_right_kps_idx_dict

    def get_left_right_kps_idx_dict(self):
        return self.left_right_kps_idx_dict


class Quad:

    def __init__(self, stereo_pair1: StereoPair, stereo_pair2: StereoPair, left_left_matches=None):
        self.stereo_pair1 = stereo_pair1
        self.stereo_pair2 = stereo_pair2
        self.left_left_matches = left_left_matches
        self.relative_trans = np.zeros(1)  # T from pair 1 to pair 2
        self.left_left_kps_idx_dict = {}  # key is keypoint of left image of pair 2 and value is keypoint of left image of pair 1 (of left left matches)

    def set_left_left_matches(self, left_left_matches):
        self.left_left_matches = left_left_matches

    def set_left_left_kps_idx_dict(self, left_left_kps_idx_dict):
        self.left_left_kps_idx_dict = left_left_kps_idx_dict

    def get_left_left_kps_idx_dict(self):
        return self.left_left_kps_idx_dict

    def set_relative_trans(self, transform):
        self.relative_trans = transform

    def get_relative_trans(self):
        return self.relative_trans

class Track:

    def __init__(self, track_id, kp_idx, pair_id):
        self.track_id = track_id
        self.last_kp_idx = kp_idx
        self.last_pair = pair_id

    def set_last_pair_id(self, pair_id):
        self.last_pair = pair_id

    def set_last_kp_idx(self, kp_idx):
        self.last_kp_idx = kp_idx
